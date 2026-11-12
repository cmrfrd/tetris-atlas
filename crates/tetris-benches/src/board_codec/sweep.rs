//! One separately compiled configuration, using the permanent suite's fixtures and kernels.
use criterion::Criterion;
use tetris_codec::{BoardCodec, GameDimensions, IdConfig};
use tetris_game::StandardTetris;

#[path = "fixtures.rs"]
mod fixtures;
#[path = "runner.rs"]
mod runner;

use runner::{SAMPLES, bench_config};

/// Checks and benchmarks one ten-column height/hole configuration.
pub fn run<const R: usize, const H: usize>(criterion: &mut Criterion)
where
    GameDimensions<StandardTetris, R, H>: IdConfig,
{
    assert!((2..=6).contains(&R) && H <= 5, "outside the bounded sweep");
    let codec = BoardCodec::<StandardTetris, R, H>::new();
    let mut seed = 0xb04d_c0de_0002u64;
    for sample in 0..1024 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let id = match sample {
            0 => 0,
            1 => codec.len() - 1,
            _ => seed % codec.len(),
        };
        let board = codec.board_from_u64(id).expect("sampled ID decodes");
        let rows = codec.rows_from_u64(id).expect("sampled ID decodes to rows");
        assert_eq!(codec.id_u64(&board), Ok(id));
        assert_eq!(codec.id_rows_u64(&rows), Ok(id));
    }
    assert!(codec.board_from_u64(codec.len()).is_err());
    assert!(codec.rows_from_u64(codec.len()).is_err());
    let rows = if R == 4 && H == 2 {
        fixtures::REACHABLE_ROWS
            .iter()
            .map(|row| std::array::from_fn(|y| row[y]))
            .collect()
    } else {
        fixtures::legal_rows::<StandardTetris, R, H>()
    };
    let name = if R == 4 && H == 2 {
        "standard_4_2_reachable_shared".to_owned()
    } else {
        format!("standard_{R}_{H}")
    };
    bench_config::<StandardTetris, R, H>(criterion, &name, &rows);
}
