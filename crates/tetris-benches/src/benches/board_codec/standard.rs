//! Ten-column Standard Tetris codecs: a small height/hole grid and representative limits.

use criterion::Criterion;
use tetris_game::StandardTetris;

tetris_bench! {
    /// Registers the fixed Standard Tetris configuration grid.
    fn bench_standard_board_codec(c: &mut Criterion) {
        codec_grid!(c, StandardTetris, "board_codec", "standard";
            4 => [0, 1, 2],
            5 => [0, 1],
            6 => [0, 1],
            2 => [0, 4],
            20 => [0],
        );
    }
}
