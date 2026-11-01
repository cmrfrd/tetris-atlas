//! Compile-time integer selection. Table generation does not choose Rust ID types.
use crate::{CompiledConfig, Dimensions};
use std::{fmt::Debug, hash::Hash};
use tetris_game::TetrisGameConfig;

/// Smallest supported integer storage width for a nonempty dense ID range.
#[doc(hidden)]
pub const fn id_bytes(board_count: u64) -> usize {
    assert!(board_count > 0, "a codec must contain the empty board");
    let max_id = board_count - 1;
    if max_id <= u16::MAX as u64 {
        2
    } else if max_id <= u32::MAX as u64 {
        4
    } else {
        8
    }
}

#[doc(hidden)]
pub struct IdWidth<const BYTES: usize>;
#[doc(hidden)]
pub trait IdStorage {
    type Id: Copy + Debug + Eq + Ord + Hash + TryFrom<u64> + Into<u64>;
}

// Rust macros cannot evaluate BOARD_COUNT. Const evaluation chooses the width;
// this macro supplies the three type mappings, entirely within this crate.
macro_rules! integer_storage {
    ($($bytes:literal => $integer:ty),+ $(,)?) => {
        $(impl IdStorage for IdWidth<$bytes> {
            type Id = $integer;
        })+
    };
}
integer_storage!(2 => u16, 4 => u32, 8 => u64);

/// Dimensions with an automatically selected `u16`, `u32`, or `u64` ID.
/// This is implemented automatically from `CompiledConfig::BOARD_COUNT`.
pub trait IdConfig: CompiledConfig {
    type Id: Copy + Debug + Eq + Ord + Hash + TryFrom<u64> + Into<u64>;
}

impl<T: CompiledConfig> IdConfig for T
where
    IdWidth<{ id_bytes(T::BOARD_COUNT) }>: IdStorage,
{
    type Id = <IdWidth<{ id_bytes(T::BOARD_COUNT) }> as IdStorage>::Id;
}

/// The codec's native integer ID type, selected from its const-evaluated board count.
pub type BoardId<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> =
    DimensionId<{ C::COLS }, ROWS, MAX_HOLES>;

pub type DimensionId<const COLS: usize, const ROWS: usize, const MAX_HOLES: usize> =
    <Dimensions<COLS, ROWS, MAX_HOLES> as IdConfig>::Id;

/// Name a codec's automatically selected ID type.
///
/// ```
/// use tetris_codec::board_id_type;
/// use tetris_game::StandardTetris;
/// let id: board_id_type!(StandardTetris, 1, 0) = 0u16;
/// assert_eq!(std::mem::size_of_val(&id), 2);
/// ```
///
/// The default decoder requires the selected type; use `board_from_u64` for an
/// external `u64` value, which is range-checked before decoding.
///
/// ```compile_fail
/// use tetris_codec::BoardCodec;
/// use tetris_game::StandardTetris;
/// let codec = BoardCodec::<StandardTetris, 1, 0>::new();
/// let external_id: u64 = 0;
/// codec.board_from_id(external_id).unwrap();
/// ```
#[macro_export]
macro_rules! board_id_type {
    ($game:ty, $rows:expr, $max_holes:expr $(,)?) => {
        $crate::BoardId<$game, { $rows }, { $max_holes }>
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoardCodec, test_support::Game};
    use tetris_game::StandardTetris;

    #[test]
    fn integer_selection_handles_exact_capacity_boundaries() {
        use crate::id::id_bytes;
        for (count, bytes) in [
            (1, 2),
            (u16::MAX as u64, 2),
            (u16::MAX as u64 + 1, 2),
            (u16::MAX as u64 + 2, 4),
            (u32::MAX as u64, 4),
            (u32::MAX as u64 + 1, 4),
            (u32::MAX as u64 + 2, 8),
            (u64::MAX, 8),
        ] {
            assert_eq!(id_bytes(count), bytes);
        }
    }

    #[test]
    fn id_alias_and_macro_resolve_to_real_integer_types() {
        type Standard = BoardCodec<StandardTetris, 4, 2>;
        const CODEC: Standard = Standard::new();
        let small: BoardId<Game<6, 12>, 3, 2> = 159u16;
        let medium: board_id_type!(StandardTetris, 4, 2) = 2559u32;
        let large: BoardId<StandardTetris, 20, 0> = 3_219_940_489_100u64;
        let small_ids = [small; 3];
        let medium_ids = [medium; 3];
        let large_ids = [large; 3];
        assert_eq!(std::mem::size_of_val(&small_ids), 6);
        assert_eq!(std::mem::size_of_val(&medium_ids), 12);
        assert_eq!(std::mem::size_of_val(&large_ids), 24);
        assert_eq!(BoardCodec::<Game<6, 12>, 3, 2>::ID_BYTES, 2);
        assert_eq!(Standard::ID_BYTES, 4);
        assert_eq!(BoardCodec::<StandardTetris, 20, 0>::ID_BYTES, 8);
        // The game height is invalid, but selecting/naming the ID type must not
        // evaluate the constructor's guarded BOARD_COUNT constant.
        let _: BoardId<Game<10, 2>, 4, 2> = medium;
        let id: u32 = CODEC.id_rows(&[3, 3, 0, 0]).unwrap();
        assert_eq!(CODEC.rows_from_id(id), Ok([3, 3, 0, 0]));
    }
}
