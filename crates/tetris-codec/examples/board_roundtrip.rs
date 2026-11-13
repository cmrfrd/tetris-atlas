#![feature(generic_const_exprs)]
#![allow(
    long_running_const_eval,
    reason = "Bounded codec construction exceeds the default const evaluation threshold."
)]

use tetris_codec::BoardCodec;
use tetris_game::{StandardTetris, TetrisBoard, TetrisGameConfig};

type Codec = BoardCodec<StandardTetris, 4, 2>;
const CODEC: Codec = Codec::new();

fn main() -> Result<(), tetris_codec::Error> {
    let board: TetrisBoard<{ StandardTetris::COLS }, { StandardTetris::ROWS }> =
        TetrisBoard::from_limbs([3, 3, 0, 0, 0, 0, 0, 0, 0, 0]);

    let id: u32 = CODEC.id(&board)?;
    assert_eq!(CODEC.board_from_id(id)?, board);

    println!(
        "count={} bits={} bytes={} tables={} id={id}",
        Codec::BOARD_COUNT,
        Codec::ID_BITS,
        Codec::ID_BYTES,
        CODEC.table_payload_bytes()
    );

    Ok(())
}
