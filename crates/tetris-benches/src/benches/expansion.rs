//! State expansion and state-key benchmarks.

use crate::fixtures::game as fixtures;
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, DefaultHasher, Hash, Hasher};
use tetris_game::TetrisPiecePlacement;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Successor {
    board: tetris_game::TetrisBoard<10, 20>,
    bag: tetris_game::TetrisPieceBagState,
    piece_index: u8,
    orientation: tetris_game::TetrisPieceOrientation<10>,
    result: tetris_game::PlacementResult,
}

fn successor_storage(input: &[fixtures::AtlasFixture]) -> Vec<Successor> {
    let count = input
        .iter()
        .map(|parent| {
            parent
                .bag
                .iter_pieces()
                .map(|piece| <TetrisPiecePlacement<10>>::all_from_piece(piece).len())
                .sum::<usize>()
        })
        .sum();
    vec![
        Successor {
            board: tetris_game::TetrisBoard::<10, 20>::new(),
            bag: tetris_game::TetrisPieceBagState::full(),
            piece_index: 0,
            orientation: <tetris_game::TetrisPieceOrientation<10>>::from_index(0),
            result: tetris_game::PlacementResult {
                is_lost: false,
                lines_cleared: 0
            },
        };
        count
    ]
}

fn expand(input: &[fixtures::AtlasFixture], output: &mut [Successor]) -> usize {
    let mut count = 0;
    for parent in input {
        for piece in parent.bag.iter_pieces() {
            let mut bag = parent.bag;
            bag.remove(piece);
            if bag.empty() {
                bag.fill();
            }

            let piece_index = piece.index();
            for &placement in <TetrisPiecePlacement<10>>::all_from_piece(piece) {
                let mut board = parent.board;
                let result = board.apply_piece_placement(placement);
                output[count] = Successor {
                    board,
                    bag,
                    piece_index,
                    orientation: placement.orientation,
                    result,
                };
                count += 1;
            }
        }
    }

    count
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "The canonical engine clears at most four lines per placement."
)]
fn successor_digest(output: &[Successor]) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for successor in output {
        let bytes = successor
            .board
            .as_limbs()
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .chain([
                u8::from(successor.bag),
                successor.piece_index,
                successor.orientation.index(),
                successor.result.is_lost as u8,
                successor.result.lines_cleared as u8,
            ]);
        for byte in bytes {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }

    digest
}

type InternMap = HashMap<
    (
        tetris_game::TetrisBoard<10, 20>,
        tetris_game::TetrisPieceBagState,
    ),
    u32,
    BuildHasherDefault<DefaultHasher>,
>;

fn intern(output: &[Successor], map: &mut InternMap, ids: &mut [u32]) {
    assert_eq!(output.len(), ids.len());

    // Include clearing retained storage and interning; exclude allocation.
    map.clear();
    for (successor, id) in output.iter().zip(ids) {
        if successor.result.is_lost {
            *id = u32::MAX;
        } else {
            let next_id =
                u32::try_from(map.len()).expect("graph index must fit its packed representation");
            *id = *map
                .entry((successor.board, successor.bag))
                .or_insert(next_id);
        }
    }
}

tetris_bench! {
    fn bench_atlas_successors(c: &mut Criterion) {
        let input = fixtures::atlas_parents();
        let mut group = c.benchmark_group("atlas");
        group.throughput(Throughput::Elements(input.len() as u64));
        let mut output = successor_storage(&input);
        let count = expand(&input, &mut output);
        assert_eq!(count, output.len());

        // Freeze output semantics across baseline and optimized binaries before timing.
        assert_eq!(successor_digest(&output), 0xed4b_7099_977a_68f8);
        group.bench_with_input(
            BenchmarkId::new("successors_fixed", input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    let count = expand(black_box(input), &mut output);
                    black_box(&output[..count]);
                });
            },
        );
        let mut map = InternMap::with_capacity_and_hasher(output.len(), Default::default());
        let mut ids = vec![0; output.len()];
        group.bench_with_input(
            BenchmarkId::new("expand_intern_fixed", input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    let count = expand(black_box(input), &mut output);
                    intern(&output[..count], &mut map, &mut ids[..count]);
                    black_box((&output[..count], &ids[..count], map.len()));
                });
            },
        );

        group.finish();
    }
}

tetris_bench! {
    fn bench_state_keys(c: &mut Criterion) {
        let keys = fixtures::state_keys();
        let mut group = c.benchmark_group("state_key");
        group.throughput(Throughput::Elements(keys.len() as u64));
        let mut hashes = vec![0u64; keys.len()];
        group.bench_with_input(
            BenchmarkId::new("hash_fixed", keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for (key, out) in black_box(keys).iter().zip(&mut hashes) {
                        let mut hasher = DefaultHasher::new();
                        key.hash(&mut hasher);
                        *out = hasher.finish();
                    }

                    black_box(&hashes);
                });
            },
        );

        let (map, queries) = fixtures::state_lookup(&keys);
        let mut output = vec![None; queries.len()];
        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("lookup_fixed", queries.len()),
            &queries,
            |b, queries| {
                b.iter(|| {
                    let map = black_box(&map);
                    for (key, out) in black_box(queries).iter().zip(&mut output) {
                        *out = map.get(key).copied();
                    }

                    black_box(&output);
                });
            },
        );

        group.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::{TetrisPiece, TetrisPieceBagState};

    #[test]
    fn interning_preserves_keys_loss_markers_and_resets_ids() {
        let parents = fixtures::atlas_parents();
        let mut successors = successor_storage(&parents);
        expand(&parents, &mut successors);
        let a = *successors.iter().find(|s| !s.result.is_lost).expect("fixture must contain the requested successor in interning_preserves_keys_loss_markers_and_resets_ids");
        let mut other_metadata = a;
        other_metadata.piece_index = (a.piece_index + 1) % 7;
        let mut other_board = a;
        other_board.board.set_bit(0, 19);
        assert_ne!(a.board, other_board.board);

        let mut other_bag = a;
        other_bag.bag.remove(a.bag.iter_pieces().next().expect("fixture collection must contain an element in interning_preserves_keys_loss_markers_and_resets_ids"));
        assert_ne!(a.bag, other_bag.bag);

        let topout = fixtures::placement_boundaries()
            .into_iter()
            .find(|f| f.name == "topout")
            .expect("fixture must contain the requested successor in interning_preserves_keys_loss_markers_and_resets_ids");
        let mut lost = a;
        lost.board = topout.input.board;
        lost.result = lost.board.apply_piece_placement(topout.input.placement);
        assert!(lost.result.is_lost);

        let input = [a, a, other_metadata, other_board, other_bag, lost, lost];
        let mut map = InternMap::default();
        let mut ids = [0; 7];
        intern(&input, &mut map, &mut ids);
        assert_eq!(ids, [0, 0, 0, 1, 2, u32::MAX, u32::MAX]);
        assert_eq!(map.len(), 3);

        for (key, id) in &map {
            let source = input[ids.iter().position(|value| value == id).expect("fixture must contain the requested identifier in interning_preserves_keys_loss_markers_and_resets_ids")];
            assert_eq!(*key, (source.board, source.bag));
        }

        // Reuse must discard old keys and restart numbering, including after all-lost input.
        intern(&[lost, lost], &mut map, &mut ids[..2]);
        assert_eq!(ids[..2], [u32::MAX; 2]);
        assert!(map.is_empty());
        intern(&[other_board, a], &mut map, &mut ids[..2]);
        assert_eq!(ids[..2], [0, 1]);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn atlas_successors_match_exhaustive_reference() {
        let input = fixtures::atlas_parents();
        assert_eq!(input, fixtures::atlas_parents());
        assert_eq!(input.len(), 128);

        for size in 1..=7 {
            assert!(input.iter().any(|p| p.bag.count() == size));
        }

        let mut output = successor_storage(&input);
        let count = expand(&input, &mut output);
        assert_eq!(count, output.len());

        let mut reference = Vec::new();
        for parent in &input {
            for piece in TetrisPiece::all() {
                if !parent.bag.contains(piece) {
                    continue;
                }

                let mask = u8::from(parent.bag) & !piece.mask();
                let bag = if mask == 0 {
                    tetris_game::TetrisPieceBagState::full()
                } else {
                    TetrisPieceBagState::try_from(mask).expect("valid bag mask")
                };

                for &placement in <TetrisPiecePlacement<10>>::all_from_piece(piece) {
                    let mut board = parent.board;
                    let result = board.apply_piece_placement(placement);
                    reference.push(Successor {
                        board,
                        bag,
                        piece_index: piece.index(),
                        orientation: placement.orientation,
                        result,
                    });
                }
            }
        }

        assert_eq!(output, reference);

        let digest = successor_digest(&output);
        println!(
            "atlas_successors={} atlas_digest={digest:016x}",
            output.len()
        );
    }
}
