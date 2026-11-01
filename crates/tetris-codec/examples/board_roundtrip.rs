#![feature(generic_const_exprs)]
#![allow(incomplete_features, long_running_const_eval)]
use tetris_codec::BoardCodec;
use tetris_game::{StandardTetris, TetrisBoard};
type Codec = BoardCodec<StandardTetris, 4, 2>;
const CODEC: Codec = Codec::new();
fn main() -> Result<(), tetris_codec::Error> {
    let board = TetrisBoard::<StandardTetris>::from_limbs([3, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
    let id = CODEC.id(&board)?;
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
