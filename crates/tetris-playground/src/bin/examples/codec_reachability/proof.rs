use crate::state::{
    Board, CODEC, COLS, Columns, FULL_BAG, MAX_HOLES, ROWS, holes, nonempty_rows, row,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tetris_game::TetrisPiecePlacement;

pub fn actions() -> Vec<TetrisPiecePlacement<COLS>> {
    TetrisPiecePlacement::<COLS>::ALL_PLACEMENTS
        .iter()
        .copied()
        .filter(|p| {
            usize::from(p.orientation.column.inner())
                + usize::from(p.piece.width(p.orientation.rotation))
                <= COLS
        })
        .collect()
}

pub fn describe(p: TetrisPiecePlacement<COLS>) -> [u8; 3] {
    [
        p.piece.index(),
        p.orientation.rotation.index(),
        p.orientation.column.inner(),
    ]
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Finding {
    Impossible {
        batch: u64,
        codec_id: u64,
        columns: Columns,
        origin: String,
    },
    Witness {
        batch: u64,
        codec_id: u64,
        columns: Columns,
        relaxed_holes: bool,
        actions: Vec<[u8; 3]>,
        remaining_bag: u8,
    },
}

pub struct Footprint {
    pub placement: TetrisPiecePlacement<COLS>,
    pub cells: Columns,
    pub height: usize,
}

pub struct Prover {
    pub actions: Vec<TetrisPiecePlacement<COLS>>,
    pub footprints: Vec<Footprint>,
    pub by_piece: [Vec<TetrisPiecePlacement<COLS>>; tetris_game::constants::NUM_TETRIS_PIECES],
}

impl Prover {
    pub fn new() -> Result<Self> {
        let actions = actions();
        ensure!(COLS > 4, "this proof needs width > four cells");
        ensure!(
            ROWS <= 28,
            "this proof needs room for a transient four-cell piece in u32 columns"
        );

        let mut footprints = Vec::new();
        for &placement in &actions {
            let mut piece = Board::new();
            piece.apply_piece_placement(placement);
            ensure!(piece.count() == 4, "proof requires four-cell tetrominoes");

            for y in 0..piece.height() as usize {
                let r = row(&piece, y);
                ensure!(
                    (r & !(r << 1)).count_ones() == 1,
                    "proof requires contiguous piece rows"
                );
            }
            footprints.push(Footprint {
                placement,
                cells: piece.as_limbs().to_owned(),
                height: piece.height() as usize,
            });
        }

        let by_piece = std::array::from_fn(|i| {
            actions
                .iter()
                .copied()
                .filter(|p| p.piece.index() as usize == i)
                .collect()
        });

        Ok(Self {
            actions,
            footprints,
            by_piece,
        })
    }

    pub fn eligible(&self, board: &Board) -> bool {
        if board.height() as usize != ROWS {
            return false;
        }

        let top = row(board, ROWS - 1);
        (top & !(top << 1)).count_ones() > 1
    }

    // Reference implementation. No candidate is proved by sampling absence.
    pub fn has_predecessor_reference(&self, board: &Board) -> bool {
        for footprint in &self.footprints {
            if footprint.height > ROWS {
                continue;
            }

            for dy in 0..=ROWS - footprint.height {
                let cells = footprint.cells.map(|c| c << dy);
                if !cells
                    .iter()
                    .zip(board.as_limbs().to_owned())
                    .all(|(q, b)| q & b == *q)
                {
                    continue;
                }

                let before = Board::from_limbs(std::array::from_fn(|x| {
                    board.as_limbs().to_owned()[x] & !cells[x]
                }));
                if !nonempty_rows(&before) {
                    continue;
                }

                let mut next = before;
                let result = next.apply_piece_placement(footprint.placement);
                if result.lines_cleared == 0 && next == *board {
                    return true;
                }
            }
        }

        false
    }

    pub fn impossible(&self, board: &Board) -> bool {
        self.eligible(board) && !self.has_predecessor_reference(board)
    }

    pub fn verify(&self, finding: &Finding) -> Result<()> {
        match finding {
            Finding::Impossible { columns, .. } => {
                let board = Board::from_limbs(*columns);
                ensure!(
                    !board.is_lost() && board.height() as usize <= ROWS,
                    "invalid proof board height"
                );
                ensure!(
                    self.eligible(&board) && !self.has_predecessor_reference(&board),
                    "impossibility proof failed"
                );
            }
            Finding::Witness {
                columns,
                relaxed_holes,
                actions,
                remaining_bag,
                ..
            } => {
                ensure!(!actions.is_empty(), "empty witness");

                let mut board = Board::new();
                let mut bag = FULL_BAG;
                for action in actions {
                    let p = self
                        .actions
                        .iter()
                        .find(|&&p| describe(p) == *action)
                        .context("unknown piece/rotation/column")?;
                    let bit = 1 << p.piece.index();
                    ensure!(bag & bit != 0, "illegal bag repetition");
                    board.apply_piece_placement(*p);
                    ensure!(
                        !board.is_lost() && board.height() as usize <= ROWS,
                        "witness exceeds height cap"
                    );

                    if !relaxed_holes {
                        ensure!(
                            holes(&board) as usize <= MAX_HOLES,
                            "strict witness exceeds hole cap"
                        );
                        CODEC
                            .id_u64(&board)
                            .context("strict witness prefix rejected by current codec")?;
                    }
                    bag &= !bit;
                    if bag == 0 {
                        bag = FULL_BAG;
                    }
                }
                ensure!(
                    board.as_limbs() == columns && bag == *remaining_bag,
                    "witness result mismatch"
                );
                ensure!(
                    holes(&board) as usize <= MAX_HOLES,
                    "witness target exceeds hole cap"
                );

                if !relaxed_holes {
                    CODEC
                        .id_u64(&board)
                        .context("strict witness target rejected by current codec")?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixtures use bounded board dimensions and case counts, or deliberately select low bits from deterministic random words."
)]
mod tests {
    use super::*;

    #[test]
    fn accepts_known_impossibility_and_rejects_reachable_board() {
        let p = Prover::new()
            .expect("valid fixture in accepts_known_impossibility_and_rejects_reachable_board");
        let bad = Board::from_limbs(std::array::from_fn(|x| {
            if x == 0 {
                15
            } else if x == 2 {
                9
            } else {
                0
            }
        }));
        assert_eq!(
            CODEC.id_u64(&bad),
            Err(tetris_codec::Error::ImpossibleBoard)
        );
        assert!(p.impossible(&bad));

        let mut good = Board::new();
        good.apply_piece_placement(p.actions[0]);
        assert!(!p.impossible(&good));
    }

    #[test]
    fn witness_checks_bag_and_raw_board_not_old_id() {
        let p =
            Prover::new().expect("valid fixture in witness_checks_bag_and_raw_board_not_old_id");
        let action = p.actions[0];
        let mut board = Board::new();
        board.apply_piece_placement(action);
        let f = Finding::Witness {
            batch: 0,
            codec_id: u64::MAX,
            columns: board.as_limbs().to_owned(),
            relaxed_holes: false,
            actions: vec![describe(action)],
            remaining_bag: FULL_BAG & !(1 << action.piece.index()),
        };
        p.verify(&f)
            .expect("valid fixture in witness_checks_bag_and_raw_board_not_old_id");
        let mut bad = f.clone();
        if let Finding::Witness { actions, .. } = &mut bad {
            actions.push(describe(action));
        }

        assert!(p.verify(&bad).is_err());
    }
}
