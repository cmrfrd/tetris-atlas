//! Bit manipulation and transpose kernels. Output storage is reused.

use crate::fixtures::{
    NUM_ELEMS,
    utils::{bit_inputs, byte_inputs, mask_inputs, transpose_inputs},
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use tetris_utils::transpose_bits;

tetris_bench! {
    fn bench_bits_to_byte(c: &mut Criterion) {
        let input = bit_inputs();
        let mut output = vec![0; input.len()];
        let mut group = c.benchmark_group("bits/to_byte_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (bits, out) in black_box(input).iter().zip(&mut output) {
                        *out = tetris_utils::bits_to_byte(bits);
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bench_byte_to_bits(c: &mut Criterion) {
        let input = byte_inputs();
        let mut output = vec![[0; 8]; input.len()];
        let mut group = c.benchmark_group("bits/from_byte_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (&byte, out) in black_box(input).iter().zip(&mut output) {
                        *out = tetris_utils::byte_to_bits(byte);
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bitmask_as_slice(c: &mut Criterion) {
        let input = mask_inputs();
        let mut output = vec![[0; 40]; input.len()];
        let mut group = c.benchmark_group("bits/bitmask_slice_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (&bits, out) in black_box(input).iter().zip(&mut output) {
                        *out = tetris_utils::BitMask::<40>::new_from_u64(bits).as_slice();
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    fn bench_transpose_bits(c: &mut Criterion) {
        let (input, wide, tall) = transpose_inputs();
        let mut group = c.benchmark_group("bits/transpose_fixed");
        group.throughput(Throughput::Elements(NUM_ELEMS as u64));
        let mut output = vec![[0; 25]; input.len()];
        group.bench_with_input("10x20", &input, |b, input| {
            b.iter(|| {
                // Input copying is included; allocation and disposal are excluded.
                output.copy_from_slice(black_box(input));
                for matrix in &mut output {
                    transpose_bits::<10, 20, 25>(matrix);
                }
                black_box(&output);
            });
        });
        let input = wide;
        let mut output = vec![[0; 40]; input.len()];
        group.bench_with_input("10x32", &input, |b, input| {
            b.iter(|| {
                // Input copying is included; allocation and disposal are excluded.
                output.copy_from_slice(black_box(input));
                for matrix in &mut output {
                    transpose_bits::<10, 32, 40>(matrix);
                }
                black_box(&output);
            });
        });
        let input = tall;
        let mut output = vec![[0; 40]; input.len()];
        group.bench_with_input("32x10", &input, |b, input| {
            b.iter(|| {
                // Input copying is included; allocation and disposal are excluded.
                output.copy_from_slice(black_box(input));
                for matrix in &mut output {
                    transpose_bits::<32, 10, 40>(matrix);
                }
                black_box(&output);
            });
        });
        group.finish();
    }
}
