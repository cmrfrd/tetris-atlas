//! Shared public-API codec kernels for representative benchmarks and the bounded sweep.
use criterion::{Criterion, Throughput, black_box};
use tetris_codec::{BoardCodec, GameDimensions, IdConfig};
use tetris_game::{TetrisBoard, TetrisGameConfig};

pub(super) const SAMPLES: usize = 256;

/// Times both public representations after checking the fixed fixtures and all inverses.
pub(super) fn bench_config<C: TetrisGameConfig, const R: usize, const H: usize>(
    c: &mut Criterion,
    name: &str,
    rows: &[[u16; R]],
) where
    GameDimensions<C, R, H>: IdConfig,
    [(); C::COLS]:,
    [(); C::ROWS]:,
{
    let codec = BoardCodec::<C, R, H>::new();
    assert_eq!(rows.len(), SAMPLES);
    let ids: Vec<_> = rows
        .iter()
        .map(|r| codec.id_rows(r).expect("legal fixture encodes"))
        .collect();
    let mut boards = Vec::with_capacity(rows.len());
    for row in rows {
        let mut columns = [0u32; C::COLS];
        for (x, col) in columns.iter_mut().enumerate() {
            for (y, &mask) in row.iter().enumerate() {
                *col |= u32::from((mask >> x) & 1) << y;
            }
        }
        boards.push(TetrisBoard::<{ C::COLS }, { C::ROWS }>::from_limbs(columns));
    }
    for ((row, board), &id) in rows.iter().zip(&boards).zip(&ids) {
        assert_eq!(codec.id(board), Ok(id));
        assert_eq!(codec.board_from_id(id), Ok(*board));
        assert_eq!(codec.rows_from_id(id), Ok(*row));
    }
    let fixture_hash = rows
        .iter()
        .flatten()
        .flat_map(|r| r.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        });
    eprintln!(
        "CODEC_FIXTURE {name} samples={} hash={fixture_hash:016x} states={} bits={} id_bytes={} node_bytes={}",
        rows.len(),
        codec.len(),
        BoardCodec::<C, R, H>::ID_BITS,
        BoardCodec::<C, R, H>::ID_BYTES,
        codec.table_payload_bytes()
    );
    let mut group = c.benchmark_group(format!("board_codec/{name}"));
    group.throughput(Throughput::Elements(SAMPLES as u64));
    group.bench_function("encode_rows_256", |b| {
        b.iter(|| {
            for row in rows {
                black_box(
                    black_box(&codec)
                        .id_rows(black_box(row))
                        .expect("fixture encodes"),
                );
            }
        })
    });
    group.bench_function("encode_native_256", |b| {
        b.iter(|| {
            for board in &boards {
                black_box(
                    black_box(&codec)
                        .id(black_box(board))
                        .expect("fixture encodes"),
                );
            }
        })
    });
    group.bench_function("decode_rows_256", |b| {
        b.iter(|| {
            for &id in &ids {
                black_box(
                    black_box(&codec)
                        .rows_from_id(black_box(id))
                        .expect("fixture decodes"),
                );
            }
        })
    });
    group.bench_function("decode_native_256", |b| {
        b.iter(|| {
            for &id in &ids {
                black_box(
                    black_box(&codec)
                        .board_from_id(black_box(id))
                        .expect("fixture decodes"),
                );
            }
        })
    });
    group.finish();
}
