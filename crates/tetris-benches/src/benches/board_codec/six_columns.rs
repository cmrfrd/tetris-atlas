//! Separate narrow-board codec coverage with its own Criterion collection.
use crate::fixtures::codec::SixColumnTetris;
use criterion::Criterion;

tetris_bench! {
    /// Registers narrow-board coverage independently of Standard Tetris.
    fn bench_six_column_board_codec(c: &mut Criterion) {
        codec_grid!(c, SixColumnTetris, "board_codec_six_columns", "six_columns";
            3 => [2],
        );
    }
}
