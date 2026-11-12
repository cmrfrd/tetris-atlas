//! Separate narrow-board codec coverage with its own Criterion collection.
use criterion::Criterion;
use tetris_game::{TetrisGameConfig, TetrisPieceBagState};

/// A narrow full-bag game that exercises two-byte IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SixColumnTetris;

impl TetrisGameConfig for SixColumnTetris {
    const ROWS: usize = 20;
    const COLS: usize = 6;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
}

tetris_bench! {
    /// Registers narrow-board coverage independently of Standard Tetris.
    fn bench_six_column_board_codec(c: &mut Criterion) {
        codec_grid!(c, SixColumnTetris, "board_codec_six_columns", "six_columns";
            3 => [2],
        );
    }
}
