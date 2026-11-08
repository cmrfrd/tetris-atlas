//! Piece, board, placement, and game replay benchmarks.

use super::fixtures;
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use tetris_game::{Major, TetrisBoard, TetrisGame, TetrisPiecePlacement};

fn placement_boundaries() -> Vec<super::fixtures::PlacementBoundary> {
    let fixtures = super::fixtures::placement_boundaries();
    for fixture in &fixtures {
        let mut board = fixture.input.board;
        let result = board.apply_piece_placement(fixture.input.placement);
        assert_eq!(result.lines_cleared, fixture.lines_cleared);
        assert_eq!(result.is_lost, fixture.is_lost);
    }
    fixtures
}

fn metadata(
    c: &mut Criterion,
    name: &str,
    input: &[TetrisPiecePlacement<10>],
    operation: impl Fn(TetrisPiecePlacement<10>) -> u8,
) {
    let mut output = vec![0; input.len()];
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(input.len() as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(input.len()),
        &input,
        |b, input| {
            b.iter(|| {
                for (&placement, out) in black_box(input).iter().zip(&mut output) {
                    *out = operation(placement);
                }
                black_box(&output);
            });
        },
    );
    group.finish();
}

tetris_bench! {
    fn bench_is_lost(c: &mut Criterion) {
        let mut group = c.benchmark_group("engine/is_lost_fixed");
        for (name, input) in fixtures::board_cases() {
            let mut output = vec![false; input.len()];
            group.throughput(Throughput::Elements(input.len() as u64));
            group.bench_with_input(name, &input, |b, input| {
                b.iter(|| {
                    for (board, out) in black_box(input).iter().zip(&mut output) {
                        *out = board.is_lost();
                    }
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
}

tetris_bench! {
    fn bench_count(c: &mut Criterion) {
        let mut group = c.benchmark_group("engine/count_fixed");
        for (name, input) in fixtures::board_cases() {
            let mut output = vec![0; input.len()];
            group.throughput(Throughput::Elements(input.len() as u64));
            group.bench_with_input(name, &input, |b, input| {
                b.iter(|| {
                    for (board, out) in black_box(input).iter().zip(&mut output) {
                        *out = board.count();
                    }
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
}

tetris_bench! {
    fn bench_clear_and_collapse_rows(c: &mut Criterion) {
        let boards = fixtures::rows();
        let mut output = vec![([0u32; 10], 0u32); boards.len()];
        let mut group = c.benchmark_group("engine/clear_fixed");
        group.throughput(Throughput::Elements(boards.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(boards.len()),
            &boards,
            |b, boards| {
                b.iter(|| {
                    for (&board, out) in black_box(boards).iter().zip(&mut output) {
                        let mut board = board;
                        let cleared = board.clear_and_collapse_rows();
                        *out = (*board.as_limbs(), cleared);
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bench_clear(c: &mut Criterion) {
        let mut group = c.benchmark_group("engine/copy_reset_fixed");
        for (name, input) in fixtures::board_cases() {
            let mut output = input.clone();
            group.throughput(Throughput::Elements(input.len() as u64));
            group.bench_with_input(name, &input, |b, input| {
                b.iter(|| {
                    // Include restoring inputs; reuse storage without allocating timed batches.
                    output.copy_from_slice(black_box(input));
                    for board in black_box(&mut output) {
                        board.clear();
                    }
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
}

tetris_bench! {
    fn bench_apply_piece_placement(c: &mut Criterion) {
        let input = fixtures::placements();
        let mut output = vec![([0u32; 10], 0u32, false); input.len()];
        let mut group = c.benchmark_group("engine/apply_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (f, out) in black_box(input).iter().zip(&mut output) {
                        let mut board = f.board;
                        let result = board.apply_piece_placement(f.placement);
                        *out = (
                            *board.as_limbs(),
                            result.lines_cleared,
                            result.is_lost,
                        );
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bench_from_cell_array(c: &mut Criterion) {
        let mut group = c.benchmark_group("engine/from_cells_fixed");
        for (name, boards) in fixtures::board_cases()
            .into_iter()
            .filter(|(name, _)| *name != "lost")
        {
            let input = boards
                .iter()
                .map(|board| board.to_cell_array(Major::Row))
                .collect::<Vec<_>>();
            let mut output = vec![TetrisBoard::<10, 20>::new(); input.len()];
            group.throughput(Throughput::Elements(input.len() as u64));
            group.bench_with_input(name, &input, |b, input| {
                b.iter(|| {
                    for (cells, out) in black_box(input).iter().zip(&mut output) {
                        *out = TetrisBoard::<10, 20>::from_cell_array(*cells, Major::Row);
                    }
                    black_box(&output);
                });
            });
        }
        group.finish();
    }
}

tetris_bench! {
    fn bench_metadata_fixed(c: &mut Criterion) {
        let input = <TetrisPiecePlacement<10>>::ALL_PLACEMENTS.repeat(64);
        metadata(c, "piece/index_fixed", &input, |p| p.piece.index());
        metadata(c, "piece/rotation_count_fixed", &input, |p| {
            p.piece.num_rotations()
        });
        metadata(c, "piece/width_fixed", &input, |p| {
            p.piece.width(p.orientation.rotation)
        });
        metadata(c, "piece/height_fixed", &input, |p| {
            p.piece.height(p.orientation.rotation)
        });
        metadata(c, "placement/index_fixed", &input, |p| p.index());
    }
}

tetris_bench! {
    fn bench_fixed_replay(c: &mut Criterion) {
        let input = fixtures::replays();
        let mut output = input
            .iter()
            .map(|fixture| fixture.expected)
            .collect::<Vec<_>>();
        let mut group = c.benchmark_group("engine/replay_fixed");
        group.throughput(Throughput::Elements(
            input.iter().map(|f| f.actions.len() as u64).sum(),
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (fixture, out) in black_box(input).iter().zip(&mut output) {
                        let mut game: TetrisGame = <TetrisGame>::new_with_seed(fixture.seed);
                        for &action in &fixture.actions {
                            game.apply_placement(action);
                        }
                        *out = game;
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bench_placement_boundaries(c: &mut Criterion) {
        let mut group = c.benchmark_group("engine/apply_boundaries");
        group.throughput(Throughput::Elements(1));
        for fixture in placement_boundaries() {
            group.bench_with_input(fixture.name, &fixture.input, |b, input| {
                b.iter(|| {
                    let mut board = black_box(input.board);
                    let result = board.apply_piece_placement(black_box(input.placement));
                    black_box((board, result));
                });
            });
        }
        group.finish();
    }
}

// Reset the seeded generator outside timing and reuse output storage.
fn sample_fixed<R, T>(c: &mut Criterion, name: &str, initial: T)
where
    R: rand::RngExt + rand::SeedableRng,
    T: Copy,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    let mut output = vec![initial; 10_000];
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(output.len() as u64));
    group.bench_function("10000", |b| {
        b.iter_batched(
            || R::seed_from_u64(0xDEAD_BEEF_CAFE),
            |mut rng| {
                for out in &mut output {
                    *out = rand::RngExt::random(&mut rng);
                }
                black_box((&output, rng));
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

tetris_bench! {
    fn bench_sampling(c: &mut Criterion) {
        use rand::rngs::{SmallRng, StdRng};
        use tetris_game::TetrisPiece;

        sample_fixed::<SmallRng, _>(c, "piece/sample_fixed/SmallRng", TetrisPiece::O);
        sample_fixed::<StdRng, _>(c, "piece/sample_fixed/StdRng", TetrisPiece::O);
        sample_fixed::<SmallRng, _>(c, "placement/sample_fixed/SmallRng", <TetrisPiecePlacement<10>>::from_index(0));
        sample_fixed::<StdRng, _>(c, "placement/sample_fixed/StdRng", <TetrisPiecePlacement<10>>::from_index(0));
    }
}
