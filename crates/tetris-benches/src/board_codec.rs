//! Representative public-API codec benchmarks with stable raw-board fixtures.
use criterion::Criterion;
use tetris_game::{StandardTetris, TetrisGameConfig, TetrisPieceBagState};

mod fixtures;
mod runner;

use runner::{SAMPLES, bench_config};

/// A narrow full-bag game that exercises two-byte IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SixColumnTetris;

impl TetrisGameConfig for SixColumnTetris {
    const ROWS: usize = 20;
    const COLS: usize = 6;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
}

tetris_bench! {
    /// Registers the same representative configurations on every normal benchmark run.
    fn bench_board_codec(c: &mut Criterion) {
        bench_config::<StandardTetris, 1, 0>(c, "standard_1_0", &fixtures::legal_rows::<StandardTetris, 1, 0>());
        bench_config::<StandardTetris, 4, 1>(c, "standard_4_1", &fixtures::legal_rows::<StandardTetris, 4, 1>());
        bench_config::<StandardTetris, 5, 1>(c, "standard_5_1", &fixtures::legal_rows::<StandardTetris, 5, 1>());
        bench_config::<StandardTetris, 6, 1>(c, "standard_6_1", &fixtures::legal_rows::<StandardTetris, 6, 1>());
        bench_config::<StandardTetris, 4, 2>(c, "standard_4_2_reachable_shared", &fixtures::REACHABLE_ROWS);
        bench_config::<StandardTetris, 2, 4>(c, "standard_2_4", &fixtures::legal_rows::<StandardTetris, 2, 4>());
        bench_config::<SixColumnTetris, 3, 2>(c, "six_columns_3_2", &fixtures::legal_rows::<SixColumnTetris, 3, 2>());
        bench_config::<StandardTetris, 20, 0>(c, "standard_20_0", &fixtures::legal_rows::<StandardTetris, 20, 0>());
    }
}
