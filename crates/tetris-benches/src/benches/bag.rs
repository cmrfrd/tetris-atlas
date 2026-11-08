//! Piece-bag drawing and set-operation benchmarks.

use super::fixtures;
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use tetris_game::{TetrisGameRng, TetrisPiece, TetrisPieceBag, TetrisPieceBagState};

// Preserve iteration order and membership in the consumed result.
fn encode_pieces(state: TetrisPieceBagState) -> u64 {
    let mut encoded = 0;
    for piece in state.iter_pieces() {
        encoded = (encoded << 4) | 0b1000 | piece.index() as u64;
    }
    encoded
}

tetris_bench! {
    fn bench_bag_state_iteration(c: &mut Criterion) {
        let input: Vec<_> = (0..=TetrisPieceBagState::FULL_MASK)
            .map(|mask| TetrisPieceBagState::try_from(mask).expect("valid mask"))
            .collect();
        let mut output = vec![0; input.len()];
        let mut group = c.benchmark_group("bag_state");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_function("iter_pieces_fixed", |b| {
            b.iter(|| {
                for (&state, out) in black_box(&input).iter().zip(&mut output) {
                    *out = encode_pieces(state);
                }
                black_box(&output);
            });
        });
        group.throughput(Throughput::Elements(1));
        let mut index = 0;
        group.bench_function("iter_pieces_scalar", |b| {
            b.iter(|| {
                let state = black_box(input[index]);
                index = (index + 1) % input.len();
                black_box(encode_pieces(state));
            });
        });
        group.finish();
    }
}

tetris_bench! {
    fn bench_bag_draw_stream_consumed(c: &mut Criterion) {
        let mut rng = TetrisGameRng::new(42);
        let mut bag = TetrisPieceBag::from_bag_state(TetrisPieceBagState::full());
        bag.shuffle(&mut rng);
        let mut state = fixtures::BagFixture { bag, rng };
        let mut group = c.benchmark_group("bag/draw_stream_consumed");
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(1), &1, |b, _| {
            b.iter(|| {
                black_box(state.draw(TetrisPieceBagState::full()));
            })
        });
        group.finish();
    }
}

tetris_bench! {
    fn bench_bag_phases(c: &mut Criterion) {
        for (name, piece_set) in [("seven", 0x7f), ("three", 0x13), ("one", 0x02)] {
            for (phase, empty) in [
                ("draw_existing_fixed", false),
                ("refill_and_draw_fixed", true),
            ] {
                let input = fixtures::bags(piece_set, empty);
                let refill = TetrisPieceBagState::try_from(piece_set).expect("valid mask");
                let mut output = vec![(input[0], TetrisPiece::O); input.len()];
                let mut group = c.benchmark_group(format!("bag/{phase}"));
                group.throughput(Throughput::Elements(input.len() as u64));
                group.bench_with_input(BenchmarkId::from_parameter(name), &input, |b, input| {
                    b.iter(|| {
                        // Snapshot copying is included; allocation and fixture construction are not.
                        for (template, out) in black_box(input).iter().zip(&mut output) {
                            let mut state = *template;
                            let piece = state.draw(refill);
                            *out = (state, piece);
                        }
                        black_box(&output);
                    });
                });
                group.finish();
            }
        }
    }
}

tetris_bench! {
    fn bench_bag_state_operations(c: &mut Criterion) {
        let input = fixtures::bag_states();
        let mut group = c.benchmark_group("bag_state");
        group.throughput(Throughput::Elements(input.len() as u64));
        let mut popped = vec![(input[0].0, Some(input[0].1)); input.len()];
        let mut removed = vec![input[0].0; input.len()];
        let mut contains = vec![false; input.len()];
        group.bench_with_input(
            BenchmarkId::new("pop_fixed", input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (&(mut state, _), out) in black_box(input).iter().zip(&mut popped) {
                        let piece = state.pop();
                        *out = (state, piece);
                    }
                    black_box(&popped);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("remove_fixed", input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (&(mut state, piece), out) in black_box(input).iter().zip(&mut removed) {
                        state.remove(piece);
                        *out = state;
                    }
                    black_box(&removed);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("contains_fixed", input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (&(state, piece), out) in black_box(input).iter().zip(&mut contains) {
                        *out = state.contains(piece);
                    }
                    black_box(&contains);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bench_bag_state_pop_scalar(c: &mut Criterion) {
        let mut group = c.benchmark_group("bag_state");
        group.throughput(Throughput::Elements(1));
        group.bench_function("pop_scalar", |b| {
            let mut mask = 0u8;
            b.iter(|| {
                mask = mask.wrapping_add(1) & tetris_game::TetrisPieceBagState::FULL_MASK;
                let mut state = tetris_game::TetrisPieceBagState::try_from(black_box(mask)).expect("valid bag mask");
                let piece = state.pop();
                // Consuming each result separately prevents cross-call SIMD, unlike pop_fixed.
                black_box((state, piece));
            });
        });
        group.finish();
    }
}

// Cover every remaining count and requested piece with fixed shuffled bags.
fn removal_inputs() -> Vec<(TetrisPieceBag, TetrisPiece)> {
    let mut input = Vec::new();
    for seed in 1..=128 {
        let mut rng = TetrisGameRng::new(seed);
        let mut bag = TetrisPieceBag::from_bag_state(TetrisPieceBagState::full());
        bag.shuffle(&mut rng);
        loop {
            for piece in TetrisPiece::all() {
                input.push((bag, piece));
            }
            if bag.count() == 0 {
                break;
            }
            let _ = bag.draw();
        }
    }
    input
}

tetris_bench! {
    fn bench_bag_remove(c: &mut Criterion) {
        let input = removal_inputs();
        let mut output = vec![(input[0].0, false); input.len()];
        let mut group = c.benchmark_group("bag/remove_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(input.len()), &input, |b, input| {
            b.iter(|| {
                // Copy each template so removals cannot exhaust bags during warm-up.
                for (&(mut bag, piece), out) in black_box(input).iter().zip(&mut output) {
                    let removed = bag.remove(piece);
                    *out = (bag, removed);
                }
                black_box(&output);
            });
        });
        group.finish();
    }
}

tetris_bench! {
    fn bench_bag_count_scalar(c: &mut Criterion) {
        let input = removal_inputs();
        let mut index = 0;
        let mut group = c.benchmark_group("bag");
        group.throughput(Throughput::Elements(1));
        group.bench_function("count_scalar", |b| {
            b.iter(|| {
                let bag = black_box(input[index].0);
                index += 1;
                if index == input.len() { index = 0; }
                black_box(bag.count());
            });
        });
        group.finish();
    }
}
