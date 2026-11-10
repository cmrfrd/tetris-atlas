use tetris_macros::inline_conditioned;

use crate::{IdConfig, private, rules::occupancy_modulus};
use std::{fmt, marker::PhantomData};
use tetris_game::{TetrisBoard, TetrisGameConfig};

/// Checks the top-row support restriction at the capped-history boundary.
///
/// At widths >=4, at most one top-row run may lack support underneath;
/// that unsupported run can have at most three cells. Floor support is handled
/// by the caller, which skips this rule for a one-row cap and widths below four.
#[inline_conditioned]
pub(crate) const fn top_supported(top: u16, below: u16, cols: usize) -> bool {
    let mut unsupported = 0;
    let mut x = 0;
    while x < cols {
        if top & (1 << x) == 0 {
            x += 1;
            continue;
        }
        let start = x;
        let mut supported = false;
        while x < cols && top & (1 << x) != 0 {
            supported |= below & (1 << x) != 0;
            x += 1;
        }
        if !supported {
            unsupported += 1;
            if unsupported > 1 || x - start > 3 {
                return false;
            }
        }
    }
    true
}

/// All failures preserve the caller's board and codec.
/// Reports invalid configurations, malformed boards and out-of-range IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidConfiguration,
    HeightExceeded,
    InvalidRowMask,
    EmptyInternalRow,
    FullRow,
    OccupancyResidue,
    TooManyHoles,
    UnsupportedTopRun,
    ImpossibleBoard,
    InvalidId,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidConfiguration => {
                "codec cap must be 1..=28 and fit a game with at most 28 rows"
            }
            Self::HeightExceeded => "board exceeds the codec height cap",
            Self::InvalidRowMask => "row has occupied cells outside C::COLS",
            Self::EmptyInternalRow => "board has an empty row below an occupied row",
            Self::FullRow => "board has an uncleared full row",
            Self::OccupancyResidue => "occupied cells violate the tetromino/line-clear residue",
            Self::TooManyHoles => "board exceeds the hole cap",
            Self::UnsupportedTopRun => "top row violates the capped-history support restriction",
            Self::ImpossibleBoard => "board violates the capped-history reachability restrictions",
            Self::InvalidId => "ID is outside the codec's dense range",
        })
    }
}

impl std::error::Error for Error {}

/// Type-level geometry and piece-set policy sharing a compiled carrier.
/// The default policy assumes the canonical full seven-piece bag.
pub struct Dimensions<
    const COLS: usize,
    const ROWS: usize,
    const MAX_HOLES: usize,
    const FULL_BAG: bool = true,
>;

/// Returns whether the configured bag contains every canonical piece.
#[doc(hidden)]
#[inline_conditioned]
pub const fn is_full_bag(pieces: tetris_game::TetrisPieceBagState) -> bool {
    pieces.count() as usize == tetris_game::constants::NUM_TETRIS_PIECES
}

/// Compiled dimensions derived from the game configuration.
pub type GameDimensions<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> =
    Dimensions<{ C::COLS }, ROWS, MAX_HOLES, { is_full_bag(C::PIECE_SET) }>;

/// A compiled carrier with a board count and shared rule storage.
pub trait CompiledConfig: private::Sealed {
    const BOARD_COUNT: u64;
    #[doc(hidden)]
    const TABLES: &'static Tables;
}

#[derive(Clone, Copy)]
pub(crate) struct SmallNode {
    pub zero: u16,
    pub one: u16,
    pub weight: u16,
}

#[derive(Clone, Copy)]
pub(crate) struct MediumNode {
    pub zero: u16,
    pub one: u16,
    pub weight: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct WideNode {
    pub zero: u32,
    pub one: u32,
    pub weight: u64,
}

trait Decision: Copy {
    /// Returns the zero-branch child index.
    fn zero(self) -> u32;

    /// Returns the one-branch child index.
    fn one(self) -> u32;

    /// Returns the zero-branch completion count.
    fn weight(self) -> u64;
}

macro_rules! access {
    ($($ty:ty),+) => {
        $(impl Decision for $ty {
            /// Returns the zero-branch child index.
            #[inline_conditioned(always)]
            fn zero(self) -> u32 {
                u32::from(self.zero)
            }

            /// Returns the one-branch child index.
            #[inline_conditioned(always)]
            fn one(self) -> u32 {
                u32::from(self.one)
            }

            /// Returns the zero-branch completion count.
            #[inline_conditioned(always)]
            fn weight(self) -> u64 {
                u64::from(self.weight)
            }
        })+
    };
}

access!(SmallNode, MediumNode, WideNode);

#[derive(Clone, Copy)]
pub(crate) enum Nodes {
    Small(&'static [SmallNode]),
    Medium(&'static [MediumNode]),
    Wide(&'static [WideNode]),
}

impl Nodes {
    /// Returns the retained node-record payload in bytes.
    #[inline_conditioned]
    fn payload_bytes(self) -> usize {
        match self {
            Self::Small(n) => std::mem::size_of_val(n),
            Self::Medium(n) => std::mem::size_of_val(n),
            Self::Wide(n) => std::mem::size_of_val(n),
        }
    }
}

/// Stores compiled completion counts and packed decision records.
#[doc(hidden)]
pub struct Tables {
    pub(crate) holes: usize,
    pub(crate) root: u32,
    pub(crate) count: u64,
    pub(crate) nodes: Nodes,
    pub(crate) levels: &'static [u16],
}

/// Traverses completion counts to rank the supplied board bits.
#[inline_conditioned]
fn rank_nodes<T: Decision>(
    nodes: &[T],
    levels: &[u16],
    root: u32,
    total: u64,
    vars: usize,
    mut bit: impl FnMut(usize) -> bool,
) -> Option<u64> {
    let mut edge = root;
    let mut rank = 0;
    if levels.is_empty() {
        // A nontrivial occupancy modulus makes every live bit significant.
        for level in 0..vars {
            if edge == 0 {
                return None;
            }
            let node = nodes[edge as usize];
            if bit(level) {
                rank += node.weight();
                edge = node.one();
            } else {
                edge = node.zero();
            }
        }
        return (edge == 1).then_some(rank);
    }
    let mut count = total;
    for level in 0..vars {
        if count == 0 {
            return None;
        }
        let node = nodes[edge as usize];
        let decision = edge > 1 && levels[edge as usize] as usize == level;
        let zero = if decision { node.weight() } else { count / 2 };
        if bit(level) {
            rank += zero;
            count -= zero;
            if decision {
                edge = node.one();
            }
        } else {
            count = zero;
            if decision {
                edge = node.zero();
            }
        }
    }
    (edge == 1 && count == 1).then_some(rank)
}

/// Decodes a dense ID by emitting its board bits to the callback.
#[inline_conditioned]
fn select_nodes<T: Decision>(
    nodes: &[T],
    levels: &[u16],
    root: u32,
    total: u64,
    mut rank: u64,
    vars: usize,
    mut bit: impl FnMut(usize, bool),
) -> Result<(), Error> {
    if rank >= total {
        return Err(Error::InvalidId);
    }
    let mut edge = root;
    if levels.is_empty() {
        for level in 0..vars {
            let node = nodes[edge as usize];
            let one = rank >= node.weight();
            if one {
                rank -= node.weight();
                edge = node.one();
            } else {
                edge = node.zero();
            }
            bit(level, one);
        }
        return if edge == 1 && rank == 0 {
            Ok(())
        } else {
            Err(Error::InvalidId)
        };
    }
    let mut count = total;
    for level in 0..vars {
        let node = nodes[edge as usize];
        let decision = edge > 1 && levels[edge as usize] as usize == level;
        let zero = if decision { node.weight() } else { count / 2 };
        let one = rank >= zero;
        if one {
            rank -= zero;
            count -= zero;
            if decision {
                edge = node.one();
            }
        } else {
            count = zero;
            if decision {
                edge = node.zero();
            }
        }
        bit(level, one);
    }
    if edge == 1 && rank == 0 && count == 1 {
        Ok(())
    } else {
        Err(Error::InvalidId)
    }
}

impl Tables {
    /// Ranks board bits using the configured node representation.
    #[inline_conditioned]
    fn rank(&self, bit: impl FnMut(usize) -> bool, vars: usize) -> Option<u64> {
        match self.nodes {
            Nodes::Small(n) => rank_nodes(n, self.levels, self.root, self.count, vars, bit),
            Nodes::Medium(n) => rank_nodes(n, self.levels, self.root, self.count, vars, bit),
            Nodes::Wide(n) => rank_nodes(n, self.levels, self.root, self.count, vars, bit),
        }
    }

    /// Emits the board bits selected by a range-checked dense ID.
    #[inline_conditioned]
    fn select(&self, rank: u64, vars: usize, bit: impl FnMut(usize, bool)) -> Result<(), Error> {
        match self.nodes {
            Nodes::Small(n) => select_nodes(n, self.levels, self.root, self.count, rank, vars, bit),
            Nodes::Medium(n) => {
                select_nodes(n, self.levels, self.root, self.count, rank, vars, bit)
            }
            Nodes::Wide(n) => select_nodes(n, self.levels, self.root, self.count, rank, vars, bit),
        }
    }
}

/// Dense IDs. Arithmetic configurations order by hole metadata and column heights;
/// other configurations order lexicographically by column, top cell first.
/// Rules and completion counts are compiled from COLS, ROWS and MAX_HOLES.
/// Eight- through twelve-column, full-seven-bag one-row codecs are exactly seed-reachable. Other
/// configurations remain necessary-condition carriers, rather than exact reachable sets.
/// Codec values are zero-sized; their type supplies the shared compiled tables.
#[derive(Clone, Copy)]
pub struct BoardCodec<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> {
    config: PhantomData<C>,
}

impl<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> BoardCodec<C, ROWS, MAX_HOLES>
where
    GameDimensions<C, ROWS, MAX_HOLES>: CompiledConfig,
{
    /// Tables fixed by the codec's type-level configuration.
    const TABLES: &'static Tables = <GameDimensions<C, ROWS, MAX_HOLES> as CompiledConfig>::TABLES;

    // All four cells must start above the previous board. Near the top of the
    // engine's u32 columns, abs_diff-based placement can overlap existing cells,
    // invalidating even the occupancy residue invariant, not just the last-move rule.
    const VALID_CONFIGURATION: bool = ROWS > 0 && ROWS <= 28 && ROWS <= C::ROWS && C::ROWS <= 28;

    /// Number of boards represented by this configuration.
    pub const BOARD_COUNT: u64 = {
        assert!(
            Self::VALID_CONFIGURATION,
            "codec cap must be 1..=28 and fit a game with at most 28 rows"
        );
        <GameDimensions<C, ROWS, MAX_HOLES> as CompiledConfig>::BOARD_COUNT
    };

    /// Minimum packed width of a dense board ID.
    pub const ID_BITS: u32 = 64 - (Self::BOARD_COUNT - 1).leading_zeros();

    /// Board width supplied by the game configuration.
    pub const COLS: usize = C::COLS;

    /// Cell-count modulus preserved by placements and line clears.
    pub const OCCUPANCY_MODULUS: usize = occupancy_modulus(C::COLS);

    /// Creates a codec whose rules and counts are evaluated at compile time.
    ///
    /// # Panics
    /// Panics if the history cap does not fit the configured game.
    #[inline_conditioned]
    pub const fn new() -> Self {
        assert!(
            Self::VALID_CONFIGURATION,
            "codec cap must be 1..=28 and fit a game with at most 28 rows"
        );
        Self {
            config: PhantomData,
        }
    }

    /// Creates a codec, returning an error for an unsupported game or history cap.
    #[inline_conditioned]
    pub const fn try_new() -> Result<Self, Error> {
        if !Self::VALID_CONFIGURATION {
            return Err(Error::InvalidConfiguration);
        }
        Ok(Self::new())
    }

    /// Returns the whole-byte storage needed for one fixed-size entry per board.
    #[inline_conditioned]
    pub const fn required_bytes(bits_per_entry: u32) -> u128 {
        (Self::BOARD_COUNT as u128 * bits_per_entry as u128).div_ceil(8)
    }

    /// Returns the number of boards in the dense ID range.
    #[inline_conditioned]
    pub const fn len(&self) -> u64 {
        Self::BOARD_COUNT
    }

    /// Returns whether the dense ID range is empty.
    #[inline_conditioned]
    pub const fn is_empty(&self) -> bool {
        Self::BOARD_COUNT == 0
    }

    /// Returns the minimum number of packed bits needed for an ID.
    #[inline_conditioned]
    pub const fn id_bits(&self) -> u32 {
        Self::ID_BITS
    }

    /// Returns the hole budget clamped to its physical maximum.
    #[inline_conditioned]
    pub const fn effective_max_holes(&self) -> usize {
        Self::TABLES.holes
    }

    /// Returns retained node and level bytes, excluding metadata and compiler scratch space.
    #[inline_conditioned]
    pub fn table_payload_bytes(&self) -> usize {
        Self::TABLES.nodes.payload_bytes() + std::mem::size_of_val(Self::TABLES.levels)
    }

    /// Returns the most specific admission error available for these rows.
    #[inline_conditioned]
    fn rejection(&self, rows: &[u16; ROWS]) -> Error {
        self.validate(rows).err().unwrap_or(Error::ImpossibleBoard)
    }

    /// Encodes a native board as a dense ID, rejecting boards outside the configured language.
    #[inline_conditioned]
    pub fn id_u64(&self, board: &TetrisBoard<{ C::COLS }, { C::ROWS }>) -> Result<u64, Error>
    where
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        let cols = board.as_limbs();
        if cols.iter().any(|&col| col >> ROWS != 0) {
            return Err(Error::HeightExceeded);
        }
        let result = if crate::single_row::applies(C::COLS, ROWS, is_full_bag(C::PIECE_SET)) {
            let row = cols
                .iter()
                .enumerate()
                .fold(0u16, |row, (x, &col)| row | (((col & 1) as u16) << x));
            crate::single_row::rank(C::COLS, row)
        } else if C::COLS * ROWS <= 64 {
            let code = cols
                .iter()
                .fold(0u64, |code, &col| (code << ROWS) | u64::from(col));
            if crate::arithmetic::applies(C::COLS, ROWS, MAX_HOLES) {
                crate::arithmetic::rank(ROWS, code)
            } else {
                Self::TABLES.rank(
                    |level| code & (1 << (C::COLS * ROWS - 1 - level)) != 0,
                    C::COLS * ROWS,
                )
            }
        } else {
            Self::TABLES.rank(
                |level| cols[level / ROWS] & (1 << (ROWS - 1 - level % ROWS)) != 0,
                C::COLS * ROWS,
            )
        };
        result.ok_or_else(|| {
            let rows = std::array::from_fn(|y| {
                cols.iter()
                    .enumerate()
                    .fold(0, |row, (x, &col)| row | (((col >> y) & 1) as u16) << x)
            });
            self.rejection(&rows)
        })
    }

    /// Encodes bottom-first rows as a dense ID, rejecting invalid masks or inadmissible boards.
    #[inline_conditioned]
    pub fn id_rows_u64(&self, rows: &[u16; ROWS]) -> Result<u64, Error> {
        let full = (1u16 << C::COLS) - 1;
        if rows.iter().any(|&row| row & !full != 0) {
            return Err(Error::InvalidRowMask);
        }
        if crate::single_row::applies(C::COLS, ROWS, is_full_bag(C::PIECE_SET)) {
            return crate::single_row::rank(C::COLS, rows[0]).ok_or_else(|| self.rejection(rows));
        }
        if crate::arithmetic::applies(C::COLS, ROWS, MAX_HOLES) {
            let code = (0..C::COLS).fold(0u64, |code, x| {
                let col = rows
                    .iter()
                    .enumerate()
                    .fold(0u64, |col, (y, row)| col | (u64::from((row >> x) & 1) << y));
                (code << ROWS) | col
            });
            return crate::arithmetic::rank(ROWS, code).ok_or_else(|| self.rejection(rows));
        }
        Self::TABLES
            .rank(
                |level| rows[ROWS - 1 - level % ROWS] & (1 << (level / ROWS)) != 0,
                C::COLS * ROWS,
            )
            .ok_or_else(|| self.rejection(rows))
    }

    /// Checks the structural rules and returns the occupied height.
    #[inline_conditioned]
    fn validate(&self, rows: &[u16; ROWS]) -> Result<usize, Error> {
        let full = (1u16 << C::COLS) - 1;
        if rows.iter().any(|&r| r & !full != 0) {
            return Err(Error::InvalidRowMask);
        }
        let height = rows.iter().rposition(|&row| row != 0).map_or(0, |i| i + 1);
        if rows[..height].contains(&0) {
            return Err(Error::EmptyInternalRow);
        }
        if rows[..height].contains(&full) {
            return Err(Error::FullRow);
        }
        let occupied: u32 = rows.iter().map(|row| row.count_ones()).sum();
        if !(occupied as usize).is_multiple_of(Self::OCCUPANCY_MODULUS) {
            return Err(Error::OccupancyResidue);
        }
        let mut seen = 0u16;
        let mut holes = 0;
        for &row in rows[..height].iter().rev() {
            holes += (seen & !row).count_ones() as usize;
            seen |= row;
        }
        if holes > Self::TABLES.holes {
            return Err(Error::TooManyHoles);
        }
        if C::COLS >= 4
            && height == ROWS
            && height >= 2
            && !top_supported(rows[height - 1], rows[height - 2], C::COLS)
        {
            return Err(Error::UnsupportedTopRun);
        }
        Ok(height)
    }

    /// Decodes a range-checked ID into the exact native board.
    #[inline_conditioned]
    pub fn board_from_u64(&self, id: u64) -> Result<TetrisBoard<{ C::COLS }, { C::ROWS }>, Error>
    where
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        if crate::single_row::applies(C::COLS, ROWS, is_full_bag(C::PIECE_SET)) {
            let row = crate::single_row::select(C::COLS, id).ok_or(Error::InvalidId)?;
            return Ok(TetrisBoard::<{ C::COLS }, { C::ROWS }>::from_limbs(
                std::array::from_fn(|x| u32::from((row >> x) & 1)),
            ));
        }
        if C::COLS * ROWS <= 64 {
            let mut code = 0u64;
            if crate::arithmetic::applies(C::COLS, ROWS, MAX_HOLES) {
                code = crate::arithmetic::select(ROWS, id).ok_or(Error::InvalidId)?;
            } else {
                Self::TABLES.select(id, C::COLS * ROWS, |_, one| {
                    code = (code << 1) | u64::from(one)
                })?;
            }
            Ok(TetrisBoard::<{ C::COLS }, { C::ROWS }>::from_limbs(
                std::array::from_fn(|x| {
                    u32::try_from((code >> ((C::COLS - 1 - x) * ROWS)) & ((1u64 << ROWS) - 1))
                        .expect("masked codec column fits u32")
                }),
            ))
        } else {
            let mut cols = [0u32; C::COLS];
            Self::TABLES.select(id, C::COLS * ROWS, |level, one| {
                cols[level / ROWS] |= u32::from(one) << (ROWS - 1 - level % ROWS)
            })?;
            Ok(TetrisBoard::<{ C::COLS }, { C::ROWS }>::from_limbs(cols))
        }
    }

    /// Decodes a range-checked ID into bottom-first row masks.
    #[inline_conditioned]
    pub fn rows_from_u64(&self, id: u64) -> Result<[u16; ROWS], Error> {
        if crate::single_row::applies(C::COLS, ROWS, is_full_bag(C::PIECE_SET)) {
            let row = crate::single_row::select(C::COLS, id).ok_or(Error::InvalidId)?;
            return Ok(std::array::from_fn(|_| row));
        }
        if crate::arithmetic::applies(C::COLS, ROWS, MAX_HOLES) {
            let code = crate::arithmetic::select(ROWS, id).ok_or(Error::InvalidId)?;
            return Ok(std::array::from_fn(|y| {
                (0..C::COLS).fold(0u16, |row, x| {
                    row | ((((code >> ((C::COLS - 1 - x) * ROWS + y)) & 1) as u16) << x)
                })
            }));
        }
        let mut rows = [0; ROWS];
        Self::TABLES.select(id, C::COLS * ROWS, |level, one| {
            rows[ROWS - 1 - level % ROWS] |= u16::from(one) << (level / ROWS)
        })?;
        Ok(rows)
    }
}

/// Native integer entry points. Selection depends only on the dimension count,
/// so even formatting a rejected constructor does not evaluate guarded constants.
impl<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> BoardCodec<C, ROWS, MAX_HOLES>
where
    GameDimensions<C, ROWS, MAX_HOLES>: IdConfig,
{
    /// Storage width of the selected integer, independent of packed `ID_BITS`.
    pub const ID_BYTES: usize = std::mem::size_of::<crate::BoardId<C, ROWS, MAX_HOLES>>();

    /// Encodes using this configuration's smallest supported integer type.
    #[inline_conditioned]
    pub fn id(
        &self,
        board: &TetrisBoard<{ C::COLS }, { C::ROWS }>,
    ) -> Result<crate::BoardId<C, ROWS, MAX_HOLES>, Error>
    where
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        Self::narrow_id(self.id_u64(board)?)
    }

    /// Encodes bottom-first rows using the selected integer type.
    #[inline_conditioned]
    pub fn id_rows(&self, rows: &[u16; ROWS]) -> Result<crate::BoardId<C, ROWS, MAX_HOLES>, Error> {
        Self::narrow_id(self.id_rows_u64(rows)?)
    }

    /// Converts a validated ID into the configured native integer type.
    #[inline_conditioned]
    fn narrow_id(id: u64) -> Result<crate::BoardId<C, ROWS, MAX_HOLES>, Error> {
        crate::BoardId::<C, ROWS, MAX_HOLES>::try_from(id).map_err(|_| Error::InvalidId)
    }

    /// Recovers the exact native board from this configuration's integer type.
    #[inline_conditioned]
    pub fn board_from_id(
        &self,
        id: crate::BoardId<C, ROWS, MAX_HOLES>,
    ) -> Result<TetrisBoard<{ C::COLS }, { C::ROWS }>, Error>
    where
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        self.board_from_u64(id.into())
    }

    /// Recovers bottom-first rows from this configuration's integer type.
    #[inline_conditioned]
    pub fn rows_from_id(
        &self,
        id: crate::BoardId<C, ROWS, MAX_HOLES>,
    ) -> Result<[u16; ROWS], Error> {
        self.rows_from_u64(id.into())
    }
}

impl<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> Default
    for BoardCodec<C, ROWS, MAX_HOLES>
where
    GameDimensions<C, ROWS, MAX_HOLES>: CompiledConfig,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> fmt::Debug
    for BoardCodec<C, ROWS, MAX_HOLES>
where
    GameDimensions<C, ROWS, MAX_HOLES>: CompiledConfig,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Result::unwrap_err and Result's Debug instantiate this even when
        // try_new rejected C::ROWS. Do not evaluate the guarded BOARD_COUNT.
        let dimension_count = <GameDimensions<C, ROWS, MAX_HOLES> as CompiledConfig>::BOARD_COUNT;
        f.debug_struct("BoardCodec")
            .field("columns", &C::COLS)
            .field("rows", &ROWS)
            .field("max_holes", &MAX_HOLES)
            .field("board_count", &dimension_count)
            .finish()
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixtures use bounded board dimensions and case counts, or deliberately select low bits from deterministic random words."
)]
pub(crate) mod tests {
    use super::*;
    use crate::fixtures::Game;
    use tetris_game::{StandardTetris, TetrisPiecePlacement};

    type Standard = BoardCodec<StandardTetris, 4, 2>;
    const CODEC: Standard = Standard::new();
    use std::sync::LazyLock;

    /// Checks that typed codec handles retain no per-instance configuration storage.
    #[test]
    fn codec_handles_are_zero_sized() {
        assert_eq!(std::mem::size_of::<Standard>(), 0);
        assert_eq!(std::mem::size_of::<BoardCodec<StandardTetris, 4, 1>>(), 0);
        assert_eq!(std::mem::size_of::<BoardCodec<Game<6, 20>, 3, 2>>(), 0);
        assert_eq!(std::mem::size_of::<BoardCodec<StandardTetris, 20, 0>>(), 0);
    }

    /// A canonical placement applied to an empty standard board.
    struct Footprint {
        placement: TetrisPiecePlacement<10>,
        columns: [u32; 10],
        height: usize,
    }
    static FOOTPRINTS: LazyLock<Vec<Footprint>> = LazyLock::new(|| {
        TetrisPiecePlacement::<10>::ALL_PLACEMENTS
            .iter()
            .copied()
            .map(|placement| {
                let mut b =
                    TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::new();
                b.apply_piece_placement(placement);
                Footprint {
                    placement,
                    columns: b.as_limbs().to_owned(),
                    height: b.height() as usize,
                }
            })
            .collect()
    });

    /// Independent engine oracle. Tests use widths <= the engine's placement inventory.
    pub(crate) fn has_predecessor<C: TetrisGameConfig, const R: usize>(rows: &[u16; R]) -> bool
    where
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        let columns = std::array::from_fn(|x| {
            (0..R).fold(0u32, |v, y| v | (u32::from((rows[y] >> x) & 1) << y))
        });
        let target = TetrisBoard::<{ C::COLS }, { C::ROWS }>::from_limbs(columns);
        for shape in FOOTPRINTS.iter() {
            if shape.height > R
                || shape
                    .columns
                    .iter()
                    .enumerate()
                    .any(|(x, &v)| x >= C::COLS && v != 0)
            {
                continue;
            }
            for dy in 0..=R - shape.height {
                if !(0..C::COLS)
                    .all(|x| (shape.columns[x] << dy) & columns[x] == shape.columns[x] << dy)
                {
                    continue;
                }
                let mut before =
                    TetrisBoard::<{ C::COLS }, { C::ROWS }>::from_limbs(std::array::from_fn(|x| {
                        columns[x] & !(shape.columns[x] << dy)
                    }));
                let cols = before.as_limbs().to_owned();
                if (0..before.height()).any(|y| cols.iter().all(|c| c & (1 << y) == 0)) {
                    continue;
                }
                let result = before.apply_piece_placement(tetris_game::TetrisPiecePlacement {
                    piece: shape.placement.piece,
                    orientation: tetris_game::TetrisPieceOrientation {
                        rotation: shape.placement.orientation.rotation,
                        column: tetris_game::Column::new(
                            shape.placement.orientation.column.inner(),
                        ),
                    },
                });
                if result.lines_cleared == 0 && before == target {
                    return true;
                }
            }
        }
        false
    }

    /// Checks admission with independent cell scans and engine predecessor replay.
    pub(crate) fn admitted<C: TetrisGameConfig, const R: usize, const H: usize>(
        rows: &[u16; R],
    ) -> bool
    where
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        if C::COLS >= 8 && C::COLS <= 12 && R == 1 && C::PIECE_SET.count() == 7 {
            return crate::single_row::tests::single_row_reachable(C::COLS).contains(&rows[0]);
        }
        let full = (1u16 << C::COLS) - 1;
        if rows.iter().any(|&r| r & !full != 0) {
            return false;
        }
        let height = rows.iter().rposition(|&r| r != 0).map_or(0, |y| y + 1);
        if rows[..height].iter().any(|&r| r == 0 || r == full) {
            return false;
        }
        let (mut divisor, mut width) = (4, C::COLS);
        while width != 0 {
            (divisor, width) = (width, divisor % width);
        }
        if rows.iter().map(|r| r.count_ones() as usize).sum::<usize>() % divisor != 0 {
            return false;
        }
        let mut holes = 0;
        for x in 0..C::COLS {
            if let Some(top) = rows.iter().rposition(|r| r & (1 << x) != 0) {
                holes += rows[..top].iter().filter(|r| **r & (1 << x) == 0).count();
            }
        }
        if holes > H {
            return false;
        }
        if C::COLS >= 4 && height == R && R >= 2 {
            let mut unsupported = 0;
            let mut x = 0;
            while x < C::COLS {
                if rows[R - 1] & (1 << x) == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                let mut supported = false;
                while x < C::COLS && rows[R - 1] & (1 << x) != 0 {
                    supported |= rows[R - 2] & (1 << x) != 0;
                    x += 1;
                }
                if !supported {
                    unsupported += 1;
                    if unsupported > 1 || x - start > 3 {
                        return false;
                    }
                }
            }
        }
        !(height == R
            && R <= 28
            && C::COLS > 4
            && (rows[R - 1] & !(rows[R - 1] << 1)).count_ones() > 1
            && !has_predecessor::<C, R>(rows))
    }

    /// Checks the public arithmetic wrappers, typed IDs and retained node payload.
    fn arithmetic_api<const R: usize>(count: u64, bits: u32)
    where
        GameDimensions<StandardTetris, R, 1>: IdConfig,
    {
        let codec = BoardCodec::<StandardTetris, R, 1>::new();
        assert_eq!(codec.len(), count);
        assert_eq!(codec.id_bits(), bits);
        assert_eq!(codec.table_payload_bytes(), 0);
        assert_eq!(BoardCodec::<StandardTetris, R, 1>::ID_BYTES, 4);
        let mut seed = 0xb38a_8f21_0731_c42d;
        for id in [0, 1, count / 2, count - 1]
            .into_iter()
            .chain((0..256).map(|_| random(&mut seed) % count))
        {
            let board = codec
                .board_from_u64(id)
                .expect("valid fixture in arithmetic_api");
            let rows = codec
                .rows_from_u64(id)
                .expect("valid fixture in arithmetic_api");
            assert_eq!(codec.id_u64(&board), Ok(id));
            assert_eq!(codec.id_rows_u64(&rows), Ok(id));
            let typed = codec.id(&board).expect("valid fixture in arithmetic_api");
            assert_eq!(codec.board_from_id(typed), Ok(board));
            let typed_rows = codec
                .id_rows(&rows)
                .expect("valid fixture in arithmetic_api");
            assert_eq!(codec.rows_from_id(typed_rows), Ok(rows));
        }
        assert_eq!(codec.board_from_u64(count), Err(Error::InvalidId));
        assert_eq!(codec.rows_from_u64(count), Err(Error::InvalidId));
        assert_eq!(codec.rows_from_u64(u64::MAX), Err(Error::InvalidId));
        assert_eq!(codec.id_rows_u64(&[1 << 10; R]), Err(Error::InvalidRowMask));
        assert_eq!(
            codec.id_u64(&TetrisBoard::from_limbs([1 << R; 10])),
            Err(Error::HeightExceeded)
        );
    }

    #[test]
    fn public_arithmetic_codecs_use_dense_u32_ids_without_node_payload() {
        arithmetic_api::<4>(58_149_719, 26);
        arithmetic_api::<5>(459_948_915, 29);
        arithmetic_api::<6>(2_551_894_523, 32);
        let codec = BoardCodec::<StandardTetris, 4, 1>::new();
        let impossible = TetrisBoard::from_limbs([11, 0, 15, 0, 3, 3, 0, 0, 1, 0]);
        assert_eq!(codec.id_u64(&impossible), Err(Error::ImpossibleBoard));
    }

    /// Advances the fixed-seed xorshift stream used by mutation tests.
    fn random(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Compares every small raw board with the independent admission oracle.
    fn exhaustive<C: TetrisGameConfig, const R: usize, const H: usize>()
    where
        GameDimensions<C, R, H>: IdConfig,
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        let codec = BoardCodec::<C, R, H>::new();
        let mut seen = vec![false; codec.len() as usize];
        for bits in 0..1u64 << (C::COLS * R) {
            let rows =
                std::array::from_fn(|y| ((bits >> (y * C::COLS)) & ((1 << C::COLS) - 1)) as u16);
            let result = codec.id_rows_u64(&rows);
            assert_eq!(result.is_ok(), admitted::<C, R, H>(&rows), "{rows:?}");
            if let Ok(id) = result {
                assert!(!seen[id as usize], "collision: {rows:?}");
                seen[id as usize] = true;
                assert_eq!(codec.rows_from_u64(id), Ok(rows));
                let b = codec
                    .board_from_u64(id)
                    .expect("valid fixture in exhaustive");
                assert_eq!(codec.id_u64(&b), Ok(id));
            }
        }
        assert!(seen.into_iter().all(|x| x));
        assert_eq!(codec.rows_from_u64(codec.len()), Err(Error::InvalidId));
    }

    #[test]
    /// Compares every small raw board with the independent admission oracle.
    fn exhaustive_small_carriers_match_engine_rule_with_dense_coverage() {
        exhaustive::<Game<2, 20>, 3, 1>();
        exhaustive::<Game<3, 20>, 3, 2>();
        exhaustive::<Game<4, 20>, 2, 1>();
        exhaustive::<Game<5, 20>, 2, 1>();
        exhaustive::<Game<6, 20>, 2, 1>();
        exhaustive::<StandardTetris, 1, 0>();
    }

    /// Checks deterministic ID round-trips and exact board mirrors.
    fn sampled<C: TetrisGameConfig, const R: usize, const H: usize>()
    where
        GameDimensions<C, R, H>: IdConfig,
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        let codec = BoardCodec::<C, R, H>::new();
        let mut seed = 0xb04d_c0de;
        for id in [0, 1, codec.len() / 2, codec.len() - 1]
            .into_iter()
            .chain((0..512).map(|_| random(&mut seed) % codec.len()))
        {
            let rows = codec.rows_from_u64(id).expect("valid fixture in sampled");
            assert!(admitted::<C, R, H>(&rows));
            assert_eq!(codec.id_rows_u64(&rows), Ok(id));
            let native = codec.board_from_u64(id).expect("valid fixture in sampled");
            assert_eq!(codec.id_u64(&native), Ok(id));
            let typed = codec.id(&native).expect("valid fixture in sampled");
            assert_eq!(codec.board_from_id(typed), Ok(native));
            let reflected = std::array::from_fn(|y| rows[y].reverse_bits() >> (16 - C::COLS));
            let mirror = codec
                .id_rows_u64(&reflected)
                .expect("valid fixture in sampled");
            assert_eq!(codec.rows_from_u64(mirror), Ok(reflected));
            assert_eq!(mirror == id, reflected == rows);
        }
    }

    #[test]
    /// Checks deterministic ID round-trips and exact board mirrors.
    fn sampled_roundtrips_include_native_wide_boards_and_exact_mirrors() {
        sampled::<StandardTetris, 4, 2>();
        sampled::<Game<6, 20>, 3, 2>();
        sampled::<Game<7, 20>, 3, 1>();
        sampled::<Game<6, 20>, 4, 2>();
        sampled::<Game<9, 20>, 20, 0>();
        sampled::<StandardTetris, 20, 0>();
    }

    #[test]
    fn compiler_derives_counts_widths_and_compact_tables() {
        assert_eq!(Standard::BOARD_COUNT, 392_431_346);
        assert_eq!(Standard::ID_BITS, 29);
        assert_eq!(Standard::ID_BYTES, 4);
        assert_eq!(BoardCodec::<Game<6, 20>, 3, 2>::BOARD_COUNT, 26_365);
        assert_eq!(
            BoardCodec::<StandardTetris, 20, 0>::BOARD_COUNT,
            3_219_940_489_101
        );
        assert!(CODEC.table_payload_bytes() <= 384_424);
        assert_eq!(std::mem::size_of::<Standard>(), 0);
        assert_eq!(Standard::required_bytes(64), 3_139_450_768);
        assert_eq!(CODEC.id_rows_u64(&[0; 4]), Ok(0));
    }

    #[test]
    fn engine_shapes_satisfy_the_exclusion_proof_premises() {
        for action in TetrisPiecePlacement::<10>::ALL_PLACEMENTS.iter().copied() {
            let width = action.piece.width(action.orientation.rotation);
            let mut footprint = TetrisBoard::<10, 20>::new();
            footprint.apply_piece_placement(TetrisPiecePlacement {
                orientation: tetris_game::TetrisPieceOrientation {
                    column: tetris_game::Column::new(0),
                    ..action.orientation
                },
                ..action
            });
            let columns = &footprint.as_limbs()[..usize::from(width)];
            assert_eq!(columns.iter().map(|c| c.count_ones()).sum::<u32>(), 4);
            let shift = columns
                .iter()
                .map(|c| c.trailing_zeros())
                .min()
                .expect("valid fixture in engine_shapes_satisfy_the_exclusion_proof_premises");
            let occupied = columns.iter().fold(0, |mask, c| mask | (c >> shift));
            let height = 32 - occupied.leading_zeros();
            let row = |y| {
                columns.iter().enumerate().fold(0u16, |mask, (x, c)| {
                    mask | ((((c >> shift) >> y) & 1) as u16) << x
                })
            };
            for y in 0..height {
                let mask = row(y);
                assert_eq!((mask & !(mask << 1)).count_ones(), 1);
            }
            if height > 1 {
                assert_ne!(row(height - 1) & row(height - 2), 0);
            } else {
                assert_eq!(row(0), 0b1111);
            }
        }
    }

    #[test]
    fn all_reachable_three_column_two_row_bag_states_are_admitted() {
        use std::collections::{HashSet, VecDeque};

        type Narrow = Game<3, 20>;
        let codec = BoardCodec::<Narrow, 2, 2>::new();
        let empty = TetrisBoard::<{ Narrow::COLS }, { Narrow::ROWS }>::new();
        let full_bag = u8::from(tetris_game::TetrisPieceBagState::full());
        let mut seen = HashSet::from([(empty.as_limbs().to_owned(), full_bag)]);
        let mut pending = VecDeque::from([(empty, full_bag)]);
        // Independent forward reachability: do not prune parents by codec admission
        // or hole count, which could otherwise hide an unsound carrier constraint.
        while let Some((board, bag)) = pending.pop_front() {
            for action in TetrisPiecePlacement::<3>::ALL_PLACEMENTS.iter().copied() {
                let bit = 1 << action.piece.index();
                if bag & bit == 0
                    || usize::from(action.orientation.column.inner())
                        + usize::from(action.piece.width(action.orientation.rotation))
                        > Narrow::COLS
                {
                    continue;
                }
                let mut next = board;
                next.apply_piece_placement(action);
                if next.is_lost() || next.height() > 2 {
                    continue;
                }
                if next.total_holes() <= 2 {
                    assert!(
                        codec.id_u64(&next).is_ok(),
                        "{:?}",
                        next.as_limbs().to_owned()
                    );
                }
                let remaining = if bag & !bit == 0 {
                    full_bag
                } else {
                    bag & !bit
                };
                if seen.insert((next.as_limbs().to_owned(), remaining)) {
                    pending.push_back((next, remaining));
                }
            }
        }
        assert!(seen.len() > 1);
    }

    /// Enumerates arbitrary-piece-order engine successors under the height cap.
    fn forward_closure<C: TetrisGameConfig, const R: usize, const H: usize>()
    where
        GameDimensions<C, R, H>: IdConfig,
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        use std::collections::HashSet;
        let codec = BoardCodec::<C, R, H>::new();
        let empty = TetrisBoard::<{ C::COLS }, { C::ROWS }>::new();
        let mut seen = HashSet::from([empty.as_limbs().to_owned()]);
        let mut arena = vec![(empty, usize::MAX, usize::MAX)];
        let mut head = 0;
        let mut checks = 0u64;
        let mut checked_boards = 0u64;
        let mut hole_returns = 0u64;
        // Allow every piece at every step: this contains all legal bag histories.
        // Neither the codec nor the target hole budget filters exploration.
        while head < arena.len() {
            let parent = arena[head].0;
            for (index, action) in TetrisPiecePlacement::<{ C::COLS }>::ALL_PLACEMENTS
                .iter()
                .copied()
                .enumerate()
            {
                if usize::from(action.orientation.column.inner())
                    + usize::from(action.piece.width(action.orientation.rotation))
                    > C::COLS
                {
                    continue;
                }
                let mut next = parent;
                next.apply_piece_placement(action);
                if next.is_lost() || next.height() as usize > R {
                    continue;
                }
                let fresh = seen.insert(next.as_limbs().to_owned());
                if next.total_holes() as usize <= H {
                    checks += 1;
                    hole_returns += u64::from(parent.total_holes() as usize > H);
                    if fresh {
                        checked_boards += 1;
                    }
                    let id = codec.id_u64(&next);
                    if id.is_err() {
                        let mut path = vec![index];
                        let mut node = head;
                        while arena[node].1 != usize::MAX {
                            path.push(arena[node].2);
                            node = arena[node].1;
                        }
                        path.reverse();
                        assert!(
                            id.is_ok(),
                            "false exclusion: cols={} R={R} H={H} board={:?} error={id:?} placement_indices={path:?}",
                            C::COLS,
                            next.as_limbs().to_owned()
                        );
                    }
                    assert_eq!(
                        codec.board_from_u64(id.expect("valid fixture in forward_closure")),
                        Ok(next)
                    );
                }
                if fresh {
                    arena.push((next, head, index));
                }
            }
            head += 1;
        }
        eprintln!(
            "FORWARD_CLOSURE cols={} R={R} H={H} reachable_boards={} target_boards={} target_edges={checks} hole_return_edges={hole_returns}",
            C::COLS,
            arena.len(),
            checked_boards + 1
        );
    }

    #[test]
    #[ignore = "complete forward engine closure across multiple small board universes"]
    fn all_piece_order_forward_closures_have_no_false_exclusions() {
        forward_closure::<Game<1, 20>, 3, 2>();
        forward_closure::<Game<2, 20>, 3, 1>();
        forward_closure::<Game<3, 20>, 3, 2>();
        forward_closure::<Game<4, 20>, 3, 2>();
        forward_closure::<Game<5, 20>, 3, 2>();
        forward_closure::<Game<6, 20>, 3, 2>();
        forward_closure::<Game<7, 20>, 3, 1>();
        forward_closure::<StandardTetris, 2, 2>();
    }

    /// Samples legal bag histories while retaining parents above the target hole budget.
    fn legal_histories<C: TetrisGameConfig, const R: usize, const H: usize>()
    where
        GameDimensions<C, R, H>: IdConfig,
        [(); C::COLS]:,
        [(); C::ROWS]:,
    {
        let codec = BoardCodec::<C, R, H>::new();
        let full_bag = u8::from(C::PIECE_SET);
        let mut checks = 0u64;
        let mut at_cap = 0u64;
        let mut clears = 0u64;
        let mut hole_returns = 0u64;
        let mut max_history = 0;
        let mut unique = std::collections::HashSet::new();
        for seed in [
            1,
            7,
            42,
            127,
            0xb04d_c0de,
            0xdead_beef,
            0x1234_5678,
            u64::MAX,
        ] {
            let mut state = seed;
            let mut board = TetrisBoard::<{ C::COLS }, { C::ROWS }>::new();
            let mut bag = full_bag;
            let mut history = Vec::new();
            for _ in 0..16_384 {
                let piece_bit = {
                    let draw = random(&mut state) % u64::from(bag.count_ones());
                    (0..7)
                        .filter(|p| bag & (1 << p) != 0)
                        .nth(draw as usize)
                        .expect("valid fixture in legal_histories")
                };
                let mut successors = Vec::new();
                for (index, action) in TetrisPiecePlacement::<{ C::COLS }>::ALL_PLACEMENTS
                    .iter()
                    .copied()
                    .enumerate()
                {
                    if action.piece.index() as usize != piece_bit
                        || usize::from(action.orientation.column.inner())
                            + usize::from(action.piece.width(action.orientation.rotation))
                            > C::COLS
                    {
                        continue;
                    }
                    let mut next = board;
                    let outcome = next.apply_piece_placement(action);
                    if next.is_lost() || next.height() as usize > R {
                        continue;
                    }
                    if next.total_holes() as usize <= H {
                        checks += 1;
                        at_cap += u64::from(next.height() as usize == R);
                        clears += u64::from(outcome.lines_cleared > 0);
                        hole_returns += u64::from(board.total_holes() as usize > H);
                        unique.insert(next.as_limbs().to_owned());
                        let id = codec.id_u64(&next);
                        assert!(
                            id.is_ok(),
                            "false exclusion: cols={} R={R} H={H} seed={seed} history={history:?} next_index={index} board={:?} error={id:?}",
                            C::COLS,
                            next.as_limbs().to_owned()
                        );
                        assert_eq!(
                            codec.board_from_u64(id.expect("valid fixture in legal_histories")),
                            Ok(next)
                        );
                    }
                    successors.push((next, index));
                }
                if successors.is_empty() {
                    board = TetrisBoard::<{ C::COLS }, { C::ROWS }>::new();
                    bag = full_bag;
                    history.clear();
                    continue;
                }
                // Mix unconstrained choices with low-hole choices. Both are actual
                // legal play; even the latter never filters by codec membership.
                if !random(&mut state).is_multiple_of(3) {
                    let min_holes = successors
                        .iter()
                        .map(|(b, _)| b.total_holes())
                        .min()
                        .expect("valid fixture in legal_histories");
                    successors.retain(|(b, _)| b.total_holes() == min_holes);
                }
                let (next, index) = successors[random(&mut state) as usize % successors.len()];
                history.push(index);
                max_history = max_history.max(history.len());
                board = next;
                bag &= !(1 << piece_bit);
                if bag == 0 {
                    bag = full_bag;
                }
            }
        }
        assert!(checks > 0);
        eprintln!(
            "LEGAL_HISTORIES cols={} R={R} H={H} seeds=8 steps_per_seed=16384 target_edges={checks} unique_targets={} cap_edges={at_cap} clearing_edges={clears} hole_return_edges={hole_returns} max_history={max_history}",
            C::COLS,
            unique.len()
        );
    }

    #[test]
    #[ignore = "fixed-seed canonical engine audit of the arithmetic specializations"]
    fn arithmetic_legal_histories_have_no_false_exclusions() {
        legal_histories::<StandardTetris, 4, 1>();
        legal_histories::<StandardTetris, 5, 1>();
        legal_histories::<StandardTetris, 6, 1>();
    }

    #[test]
    #[ignore = "fixed-seed legal engine histories with unconstrained hole excursions"]
    fn legal_seven_bag_histories_have_no_false_exclusions() {
        legal_histories::<StandardTetris, 4, 2>();
        legal_histories::<Game<6, 20>, 4, 2>();
        legal_histories::<Game<7, 20>, 3, 1>();
        legal_histories::<StandardTetris, 20, 0>();
        legal_histories::<Game<9, 20>, 20, 0>();
    }

    #[test]
    fn three_column_line_clear_can_create_two_unsupported_top_runs() {
        type Narrow = Game<3, 20>;
        let codec = BoardCodec::<Narrow, 2, 2>::new();
        // Two full seven-bags, followed by J from the third bag. Its full
        // three-cell row clears above the history cap and leaves a cell behind.
        let actions = [
            [0, 0, 0],
            [1, 1, 2],
            [2, 0, 0],
            [5, 2, 0],
            [6, 0, 0],
            [3, 1, 1],
            [4, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [5, 2, 0],
            [4, 1, 1],
            [3, 1, 0],
            [6, 0, 0],
            [6, 2, 0],
        ];
        let mut board = TetrisBoard::<{ Narrow::COLS }, { Narrow::ROWS }>::new();
        let mut bag = tetris_game::TetrisPieceBagState::full();
        for action in actions {
            let placement = *TetrisPiecePlacement::<3>::ALL_PLACEMENTS
                .iter()
                .find(|p| {
                    [
                        p.piece.index(),
                        p.orientation.rotation.index(),
                        usize::from(p.orientation.column.inner()) as u8,
                    ] == action
                })
                .expect(
                    "valid fixture in three_column_line_clear_can_create_two_unsupported_top_runs",
                );
            assert!(bag.contains(placement.piece));
            bag.remove(placement.piece);
            if bag.empty() {
                bag = tetris_game::TetrisPieceBagState::full();
            }
            board.apply_piece_placement(placement);
            assert!(!board.is_lost());
            assert!(board.height() <= 2);
            let id = codec.id_u64(&board).expect(
                "valid fixture in three_column_line_clear_can_create_two_unsupported_top_runs",
            );
            assert_eq!(codec.board_from_u64(id), Ok(board));
        }
        assert_eq!(board.as_limbs().to_owned(), [2, 1, 2]);
        assert_eq!(board.total_holes(), 2);
        assert_eq!(codec.len(), 43);
    }

    #[test]
    fn cap_boundary_stacks_and_four_column_self_clears_roundtrip() {
        let vertical = *TetrisPiecePlacement::<4>::ALL_PLACEMENTS
            .iter()
            .find(|p| {
                usize::from(p.orientation.column.inner()) == 0
                    && p.piece.width(p.orientation.rotation) == 1
                    && p.piece.height(p.orientation.rotation) == 4
            })
            .expect("valid fixture in cap_boundary_stacks_and_four_column_self_clears_roundtrip");
        let horizontal = *TetrisPiecePlacement::<4>::ALL_PLACEMENTS
            .iter()
            .find(|p| {
                usize::from(p.orientation.column.inner()) == 0
                    && p.piece.width(p.orientation.rotation) == 4
            })
            .expect("valid fixture in cap_boundary_stacks_and_four_column_self_clears_roundtrip");
        // Arbitrary piece orders are a superset of bag histories. These isolate
        // the exact upper cap and the exceptional above-cap clear at width four.
        let codec = BoardCodec::<Game<4, 28>, 4, 0>::new();
        let mut board = TetrisBoard::<4, 28>::new();
        board.apply_piece_placement(vertical);
        assert_eq!(board.height(), 4);
        let before = board;
        assert_eq!(board.apply_piece_placement(horizontal).lines_cleared, 1);
        assert_eq!(board, before);
        assert_eq!(
            codec.board_from_u64(codec.id_u64(&board).expect(
                "valid fixture in cap_boundary_stacks_and_four_column_self_clears_roundtrip"
            )),
            Ok(board)
        );
        // Exercise spawn non-overlap at the maximum supported cap independently
        // of a large diagram's compile-time construction budget.
        for n in 2..=7 {
            board.apply_piece_placement(vertical);
            assert_eq!(board.height(), n * 4);
            assert_eq!(board.as_limbs().to_owned()[0].count_ones(), n * 4);
        }
        assert_eq!(board.height(), 28);
        let codec = BoardCodec::<StandardTetris, 20, 0>::new();
        let mut board = TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::new();
        for _ in 0..5 {
            board.apply_piece_placement(
                TetrisPiecePlacement::<10>::ALL_PLACEMENTS
                    .iter()
                    .copied()
                    .find(|p| {
                        p.piece == vertical.piece
                            && p.orientation.rotation == vertical.orientation.rotation
                            && p.orientation.column.inner() == 0
                    })
                    .expect("valid fixture in cap_boundary_stacks_and_four_column_self_clears_roundtrip"),
            );
            let id = codec.id_u64(&board).expect(
                "valid fixture in cap_boundary_stacks_and_four_column_self_clears_roundtrip",
            );
            assert_eq!(codec.board_from_u64(id), Ok(board));
        }
        assert_eq!(board.height(), 20);
    }

    #[test]
    fn unsupported_spawn_overlap_heights_are_rejected() {
        assert!(BoardCodec::<Game<1, 28>, 28, 0>::try_new().is_ok());
        assert_eq!(
            BoardCodec::<Game<1, 31>, 28, 0>::try_new()
                .expect_err("invalid fixture must be rejected"),
            Error::InvalidConfiguration
        );
        assert_eq!(
            BoardCodec::<Game<1, 28>, 29, 0>::try_new()
                .expect_err("invalid fixture must be rejected"),
            Error::InvalidConfiguration
        );
        assert_eq!(
            BoardCodec::<Game<1, 28>, 30, 0>::try_new()
                .expect_err("invalid fixture must be rejected"),
            Error::InvalidConfiguration
        );
        assert_eq!(
            BoardCodec::<Game<1, 28>, 31, 0>::try_new()
                .expect_err("invalid fixture must be rejected"),
            Error::InvalidConfiguration
        );
        // The previous engine accepted this spawn-overlap board after a 31-row
        // replay. Current gameplay rejects those dimensions before placement.
        let old = TetrisBoard::<10, 20>::from_limbs([
            8_388_607,
            268_435_455,
            2_147_483_647,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]);
        assert_eq!(old.count(), 82);
        assert_eq!(CODEC.id_u64(&old), Err(Error::HeightExceeded));
    }

    #[test]
    fn invalid_inputs_preserve_precise_errors_and_are_never_truncated() {
        for (rows, error) in [
            ([1024, 0, 0, 0], Error::InvalidRowMask),
            ([0, 3, 0, 0], Error::EmptyInternalRow),
            ([1023, 0, 0, 0], Error::FullRow),
            ([1, 0, 0, 0], Error::OccupancyResidue),
            ([5, 1, 1, 5], Error::ImpossibleBoard),
        ] {
            assert_eq!(CODEC.id_rows_u64(&rows), Err(error));
        }
        for x in 0..10 {
            for y in 4..32 {
                let mut cols = [0; 10];
                cols[x] = 1 << y;
                assert_eq!(
                    CODEC.id_u64(&TetrisBoard::<
                        { StandardTetris::COLS },
                        { StandardTetris::ROWS },
                    >::from_limbs(cols)),
                    Err(Error::HeightExceeded)
                );
            }
        }
        assert_eq!(CODEC.board_from_u64(u64::MAX), Err(Error::InvalidId));

        type Invalid = BoardCodec<Game<10, 2>, 4, 2>;
        let error = Invalid::try_new().expect_err("invalid fixture must be rejected");
        assert_eq!(error, Error::InvalidConfiguration);
        assert!(!format!("{error:?}").is_empty());
    }

    #[test]
    fn mutations_match_independent_cell_and_engine_oracle() {
        let mut s = 0x58a1_2095;
        for _ in 0..2048 {
            let mut rows = CODEC
                .rows_from_u64(random(&mut s) % CODEC.len())
                .expect("valid fixture in mutations_match_independent_cell_and_engine_oracle");
            let bit = random(&mut s) as usize % 40;
            rows[bit / 10] ^= 1 << (bit % 10);
            assert_eq!(
                CODEC.id_rows_u64(&rows).is_ok(),
                admitted::<StandardTetris, 4, 2>(&rows),
                "{rows:?}"
            );
        }
    }

    #[test]
    fn known_impossible_family_is_intrinsically_absent() {
        for a in 0usize..10 {
            for b in 0usize..10 {
                if a.abs_diff(b) < 2 {
                    continue;
                }
                for low in 0..3 {
                    let board =
                        TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::from_limbs(std::array::from_fn(|x| {
                            if x == a {
                                15
                            } else if x == b {
                                8 | (1 << low)
                            } else {
                                0
                            }
                        }));
                    assert_eq!(CODEC.id_u64(&board), Err(Error::ImpossibleBoard));
                }
            }
        }
    }

    #[test]
    fn seeded_legal_play_survives_the_filter() {
        let mut seed = 0x8013_2261;
        for _ in 0..512 {
            let mut board =
                TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::new();
            for _ in 0..32 {
                let mut choices = Vec::new();
                for p in TetrisPiecePlacement::<10>::ALL_PLACEMENTS.iter().copied() {
                    let mut next = board;
                    next.apply_piece_placement(p);
                    if !next.is_lost() && next.height() <= 4 && next.total_holes() <= 2 {
                        choices.push(next);
                    }
                }
                if choices.is_empty() {
                    break;
                }
                board = choices[random(&mut seed) as usize % choices.len()];
                let id = CODEC
                    .id_u64(&board)
                    .expect("valid fixture in seeded_legal_play_survives_the_filter");
                assert_eq!(CODEC.board_from_u64(id), Ok(board));
            }
        }
    }

    #[test]
    #[ignore = "bounded release-only traversal ablation"]
    fn profile_parity_traversal() -> Result<(), String> {
        use std::{hint::black_box, time::Instant};

        type D = GameDimensions<StandardTetris, 4, 2>;
        let levels: Vec<_> = <D as crate::decision::Source>::NODES
            [..<D as crate::decision::Source>::LEN]
            .iter()
            .map(|n| n.var)
            .collect();
        let Nodes::Medium(nodes) = Standard::TABLES.nodes else {
            return Err("expected medium nodes".into());
        };
        let mut seed = 0xb04d_c0de;
        let mut codes = Vec::new();
        for _ in 0..256 {
            let board = CODEC
                .board_from_u64(random(&mut seed) % CODEC.len())
                .expect("valid fixture in profile_parity_traversal");
            codes.push(
                board
                    .as_limbs()
                    .to_owned()
                    .iter()
                    .fold(0u64, |code, &col| (code << 4) | u64::from(col)),
            );
        }
        for &code in &codes {
            let a = rank_nodes(
                nodes,
                &levels,
                Standard::TABLES.root,
                CODEC.len(),
                40,
                |level| code & (1 << (39 - level)) != 0,
            );
            let b = rank_nodes(
                nodes,
                &[],
                Standard::TABLES.root,
                CODEC.len(),
                40,
                |level| code & (1 << (39 - level)) != 0,
            );
            assert_eq!(a, b);
        }
        for (trial, generic) in [false, true, true, false, true, false, false, true]
            .into_iter()
            .enumerate()
        {
            let selected = black_box(if generic { levels.as_slice() } else { &[] });
            let start = Instant::now();
            let mut checksum = 0u64;
            for _ in 0..4096 {
                for &code in &codes {
                    let code = black_box(code);
                    let rank = rank_nodes(
                        black_box(nodes),
                        selected,
                        black_box(Standard::TABLES.root),
                        black_box(CODEC.len()),
                        40,
                        |level| code & (1 << (39 - level)) != 0,
                    )
                    .expect("valid fixture in profile_parity_traversal");
                    checksum = checksum.wrapping_add(black_box(rank));
                }
            }
            println!(
                "{{\"trial\":{trial},\"generic\":{generic},\"ns_per_call\":{},\"checksum\":{checksum}}}",
                start.elapsed().as_secs_f64() * 1e9 / (4096.0 * 256.0)
            );
        }
        Ok(())
    }
}
