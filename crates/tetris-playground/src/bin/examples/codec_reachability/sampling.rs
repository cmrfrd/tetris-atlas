use crate::{
    proof::{Finding, Prover, describe},
    state::{
        Bits, Board, CODEC, COLS, Codec, FULL_BAG, MAX_HOLES, PlayTotals, ROWS, Rng, State, holes,
    },
};
use anyhow::{Result, ensure};
use rayon::prelude::*;

pub fn batch(state: &mut State, prover: &Prover, pool: &rayon::ThreadPool) -> Result<Vec<Finding>> {
    let batch = state.header.totals.batches;
    let work = state.header.workload.clone();
    let mut findings = Vec::new();
    if work.controls && batch == 0 && ROWS == 4 && MAX_HOLES >= 2 {
        for anchor in 0..COLS {
            for other in 0..COLS {
                if anchor.abs_diff(other) < 2 {
                    continue;
                }
                for low in 0..3 {
                    let board = Board::from_limbs(std::array::from_fn(|x| {
                        if x == anchor {
                            15
                        } else if x == other {
                            8 | (1 << low)
                        } else {
                            0
                        }
                    }));
                    if let Ok(id) = CODEC.id_u64(&board) {
                        consider(state, prover, board, id, "control", &mut findings)?;
                    }
                }
            }
        }
    }
    // Generate the entire deterministic input sequence before parallel work.
    let parents: Vec<_> = (0..work.parents)
        .map(|_| state.header.rng[0].next() % Codec::BOARD_COUNT)
        .collect();
    let counts = pool.install(|| {
        parents
            .par_chunks(64)
            .map(|ids| -> Result<(u64, u64, u64)> {
                let mut moves = 0;
                let mut hits = 0;
                let mut fresh = 0;
                for &id in ids {
                    let board = CODEC.board_from_u64(id)?;
                    for &placement in &prover.actions {
                        moves += 1;
                        let mut next = board;
                        next.apply_piece_placement(placement);
                        if next.is_lost() || next.height() as usize > ROWS {
                            continue;
                        }
                        if let Ok(id) = CODEC.id_u64(&next) {
                            ensure!(
                                !state.bits[3].contains(id),
                                "incoming edge contradicts an impossibility proof"
                            );
                            hits += 1;
                            fresh += u64::from(state.bits[0].mark(id));
                        }
                    }
                }
                Ok((moves, hits, fresh))
            })
            .try_reduce(|| (0, 0, 0), |a, b| Ok((a.0 + b.0, a.1 + b.1, a.2 + b.2)))
    })?;
    state.header.totals.parent_samples += work.parents as u64;
    state.header.totals.parent_moves += counts.0;
    state.header.totals.incoming_moves += counts.1;
    state.header.totals.incoming_unique += counts.2;
    let strict = play(
        false,
        batch,
        &work,
        &mut state.header.rng[1],
        &state.bits[1],
        &state.bits[3],
        &mut state.header.totals.strict,
        prover,
        &state.header.watch,
    )?;
    let relaxed = play(
        true,
        batch,
        &work,
        &mut state.header.rng[2],
        &state.bits[2],
        &state.bits[3],
        &mut state.header.totals.relaxed,
        prover,
        &state.header.watch,
    )?;
    state.header.totals.witnesses_saved += (strict.len() + relaxed.len()) as u64;
    findings.extend(strict);
    findings.extend(relaxed);
    for _ in 0..work.candidates {
        let id = state.header.rng[3].next() % Codec::BOARD_COUNT;
        state.header.totals.candidates += 1;
        state.header.totals.candidate_incoming_hits += u64::from(state.bits[0].contains(id));
        let witnessed = state.bits[1].contains(id) || state.bits[2].contains(id);
        state.header.totals.candidate_witness_hits += u64::from(witnessed);
        if witnessed || state.bits[3].contains(id) {
            continue;
        }
        let board = CODEC.board_from_u64(id)?;
        consider(state, prover, board, id, "uniform", &mut findings)?;
    }
    // Raw proof/witness verification is the gate before anything reaches the journal.
    for finding in &findings {
        prover.verify(finding)?;
    }
    state.header.totals.batches += 1;
    Ok(findings)
}

fn consider(
    state: &mut State,
    prover: &Prover,
    board: Board,
    id: u64,
    origin: &str,
    findings: &mut Vec<Finding>,
) -> Result<()> {
    if state.bits[3].contains(id) || !prover.impossible(&board) {
        return Ok(());
    }
    ensure!(
        !state.bits[0].contains(id) && !state.bits[1].contains(id) && !state.bits[2].contains(id),
        "impossibility proof contradicts observations"
    );
    let finding = Finding::Impossible {
        batch: state.header.totals.batches,
        codec_id: id,
        columns: board.as_limbs(),
        origin: origin.to_owned(),
    };
    prover.verify(&finding)?;
    state.bits[3].mark(id);
    state.header.totals.proved_impossible += 1;
    findings.push(finding);
    Ok(())
}

fn play(
    relaxed: bool,
    batch: u64,
    work: &crate::state::Workload,
    rng: &mut Rng,
    seen: &Bits,
    impossible: &Bits,
    totals: &mut PlayTotals,
    prover: &Prover,
    watch: &[u64],
) -> Result<Vec<Finding>> {
    let mut findings = Vec::new();
    let mut choices = Vec::new();
    let mut path = Vec::with_capacity(work.horizon);
    totals.unique += u64::from(seen.mark(0));
    for _ in 0..work.episodes {
        totals.episodes += 1;
        let mut board = Board::new();
        let mut bag = FULL_BAG;
        let mut outside = false;
        path.clear();
        for _ in 0..work.horizon {
            let mut piece_bits = bag;
            for _ in 0..rng.next() % u64::from(bag.count_ones()) {
                piece_bits &= piece_bits - 1;
            }
            let piece = piece_bits.trailing_zeros() as u8;
            choices.clear();
            for &placement in &prover.by_piece[piece as usize] {
                let mut next = board;
                let result = next.apply_piece_placement(placement);
                if next.is_lost() || next.height() as usize > ROWS {
                    continue;
                }
                let h = holes(&next);
                if !relaxed && h as usize > MAX_HOLES {
                    continue;
                }
                choices.push((next, placement, result.lines_cleared, h));
            }
            if choices.is_empty() {
                totals.stopped += 1;
                break;
            }
            let pick = if relaxed && outside && rng.next() % 5 != 0 {
                let least = choices.iter().map(|c| c.3).min().unwrap();
                let n = choices.iter().filter(|c| c.3 == least).count();
                let pick = rng.next() as usize % n;
                choices
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.3 == least)
                    .nth(pick)
                    .unwrap()
                    .0
            } else {
                rng.next() as usize % choices.len()
            };
            let (next, placement, lines, h) = choices[pick];
            board = next;
            path.push(placement);
            bag &= !(1 << piece);
            if bag == 0 {
                bag = FULL_BAG;
            }
            totals.pieces += 1;
            totals.lines += u64::from(lines);
            totals.max_holes = totals.max_holes.max(h);
            let id = if h as usize <= MAX_HOLES {
                if relaxed {
                    CODEC.id_u64(&board).ok()
                } else {
                    Some(CODEC.id_u64(&board)?)
                }
            } else {
                None
            };
            if let Some(id) = id {
                ensure!(
                    !impossible.contains(id),
                    "seed path contradicts an impossibility proof"
                );
                totals.codec_visits += 1;
                let fresh = seen.mark(id);
                totals.unique += u64::from(fresh);
                if outside {
                    totals.returns += 1;
                }
                let budget = if relaxed && !outside {
                    work.witnesses / 2
                } else {
                    work.witnesses
                };
                if (fresh && watch.binary_search(&id).is_ok())
                    || (findings.len() < budget && (fresh || outside))
                {
                    findings.push(Finding::Witness {
                        batch,
                        codec_id: id,
                        columns: board.as_limbs(),
                        relaxed_holes: relaxed,
                        actions: path.iter().copied().map(describe).collect(),
                        remaining_bag: bag,
                    });
                }
                outside = false;
            } else {
                if !outside {
                    totals.excursions += 1;
                }
                outside = true;
            }
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn watched_targets_get_paths_even_with_zero_random_witness_budget() {
        let prover = Prover::new().unwrap();
        let mut watch: Vec<_> = prover
            .actions
            .iter()
            .filter_map(|&p| {
                let mut board = Board::new();
                board.apply_piece_placement(p);
                CODEC.id_u64(&board).ok()
            })
            .collect();
        watch.sort_unstable();
        watch.dedup();
        let work = crate::state::Workload {
            seed: 42,
            parents: 0,
            episodes: 1,
            candidates: 0,
            horizon: 1,
            witnesses: 0,
            controls: false,
            watch: 0,
        };
        let seen = Bits::new(Codec::BOARD_COUNT).unwrap();
        let bad = Bits::new(Codec::BOARD_COUNT).unwrap();
        let found = play(
            false,
            0,
            &work,
            &mut Rng(42),
            &seen,
            &bad,
            &mut PlayTotals::default(),
            &prover,
            &watch,
        )
        .unwrap();
        assert_eq!(found.len(), 1);
        prover.verify(&found[0]).unwrap();
    }
    #[test]
    fn sampled_legal_paths_are_never_certified_impossible() {
        let prover = Prover::new().unwrap();
        let work = crate::state::Workload {
            seed: 42,
            parents: 0,
            episodes: 100,
            candidates: 0,
            horizon: 32,
            witnesses: 100,
            controls: false,
            watch: 0,
        };
        let seen = Bits::new(Codec::BOARD_COUNT).unwrap();
        let impossible = Bits::new(Codec::BOARD_COUNT).unwrap();
        for relaxed in [false, true] {
            let findings = play(
                relaxed,
                0,
                &work,
                &mut Rng(42),
                &seen,
                &impossible,
                &mut PlayTotals::default(),
                &prover,
                &[],
            )
            .unwrap();
            assert!(!findings.is_empty());
            for f in findings {
                prover.verify(&f).unwrap();
                if let Finding::Witness { columns, .. } = f {
                    assert!(!prover.impossible(&Board::from_limbs(columns)));
                }
            }
        }
    }
}
