//! Ten-column Standard Tetris codecs: a small height/hole grid and representative limits.
use criterion::Criterion;
use tetris_game::StandardTetris;

tetris_bench! {
    /// Registers the fixed Standard Tetris grid and the historical shared fixture.
    fn bench_standard_board_codec(c: &mut Criterion) {
        codec_grid!(c, StandardTetris, "board_codec", "standard";
            4 => [0, 1],
            5 => [0, 1],
            6 => [0, 1],
            2 => [0, 4],
            20 => [0],
        );
        super::runner::bench_config::<StandardTetris, 4, 2>(
            c,
            "board_codec",
            "standard_4_2_reachable_shared",
            &super::fixtures::REACHABLE_ROWS,
        );
    }
}
