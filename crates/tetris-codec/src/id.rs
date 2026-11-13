//! Compile-time integer selection. Table generation does not choose Rust ID types.
//!
//! The width predicate and bridge types must be public because they occur in the
//! public `IdConfig::Id` definition. This module stays private; callers use `BoardId`.

use crate::{CompiledConfig, GameDimensions};
use std::{fmt::Debug, hash::Hash};
use tetris_game::TetrisGameConfig;

/// Returns the smallest integer width in bytes that fits every ID in `0..board_count`.
///
/// Supported widths are 1, 2, 4, and 8 bytes. Selection uses the largest ID,
/// `board_count - 1`, so a count of 256 still fits in one byte.
///
/// # Panics
///
/// Panics if `board_count` is zero: every codec must contain the empty board.
pub const fn id_bytes(board_count: u64) -> usize {
    assert!(board_count > 0, "a codec must contain the empty board");

    let max_id = board_count - 1;
    if max_id <= u8::MAX as u64 {
        1
    } else if max_id <= u16::MAX as u64 {
        2
    } else if max_id <= u32::MAX as u64 {
        4
    } else {
        8
    }
}

/// Type-level bridge from a const-evaluated byte width to an integer type.
///
/// [`IdStorage`] maps widths 1, 2, 4, and 8 to their unsigned integers. This bridge
/// must be public because it occurs in [`IdConfig`]'s associated type definition;
/// the containing module remains private. Callers use [`BoardId`] instead.
pub struct IdWidth<const BYTES: usize>;

/// Maps the four supported [`IdWidth`] values to `u8`, `u16`, `u32`, or `u64`.
///
/// This public bridge is required by [`IdConfig`]'s associated type definition.
/// Const evaluation selects the width; these implementations supply the Rust type.
pub trait IdStorage {
    /// Unsigned integer with exactly the selected number of bytes.
    type Id: Copy + Debug + Eq + Ord + Hash + TryFrom<u64> + Into<u64>;
}

// Rust macros cannot evaluate BOARD_COUNT. Const evaluation chooses the width;
// this macro supplies the four type mappings, entirely within this crate.
macro_rules! integer_storage {
    ($($bytes:literal => $integer:ty),+ $(,)?) => {
        $(impl IdStorage for IdWidth<$bytes> {
            type Id = $integer;
        })+
    };
}

integer_storage!(1 => u8, 2 => u16, 4 => u32, 8 => u64);

/// Dimensions with an automatically selected `u8`, `u16`, `u32`, or `u64` ID.
/// Implemented automatically from [`CompiledConfig::BOARD_COUNT`]. The const count
/// selects a byte width, and the crate's type-level mapping selects the integer.
/// Callers do not implement this trait or configure a storage width themselves.
pub trait IdConfig: CompiledConfig {
    /// Smallest supported unsigned integer that fits every admitted board ID.
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
    <GameDimensions<C, ROWS, MAX_HOLES> as IdConfig>::Id;

/// Name a codec's automatically selected ID type.
///
/// ```
/// use tetris_codec::board_id_type;
/// use tetris_game::StandardTetris;
/// let id: board_id_type!(StandardTetris, 2, 0) = 0u16;
/// assert_eq!(std::mem::size_of_val(&id), 2);
/// ```
///
/// ID aliases also reject one-row configurations at compile time:
///
/// ```compile_fail
/// use tetris_codec::board_id_type;
/// use tetris_game::StandardTetris;
/// let _: board_id_type!(StandardTetris, 1, 0) = 0;
/// ```
///
/// The default decoder requires the selected type; use `board_from_u64` for an
/// external `u64` value, which is range-checked before decoding.
///
/// ```compile_fail
/// use tetris_codec::BoardCodec;
/// use tetris_game::StandardTetris;
/// let codec = BoardCodec::<StandardTetris, 2, 0>::new();
/// let external_id: u64 = 0;
/// codec.board_from_id(external_id).expect("valid fixture in id_bytes");
/// ```
#[macro_export]
macro_rules! board_id_type {
    ($game:ty, $rows:expr, $max_holes:expr $(,)?) => {
        $crate::BoardId<$game, { $rows }, { $max_holes }>
    };
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixtures use bounded board dimensions and case counts, or deliberately select low bits from deterministic random words."
)]
mod tests {
    use super::*;
    use crate::{BoardCodec, board::tests::Game};
    use tetris_game::StandardTetris;

    #[test]
    fn integer_selection_handles_exact_capacity_boundaries() {
        use crate::id::id_bytes;
        for (count, bytes) in [
            (1, 1),
            (u8::MAX as u64, 1),
            (u8::MAX as u64 + 1, 1),
            (u8::MAX as u64 + 2, 2),
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

        let id: u32 = CODEC
            .id_rows(&[3, 3, 0, 0])
            .expect("valid fixture in id_alias_and_macro_resolve_to_real_integer_types");
        assert_eq!(CODEC.rows_from_id(id), Ok([3, 3, 0, 0]));
    }
}
