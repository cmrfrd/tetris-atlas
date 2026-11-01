use crate::{IdConfig, private, rules::occupancy_modulus};
use std::{fmt, marker::PhantomData};
use tetris_game::{TetrisBoard, TetrisGameConfig};

/// At the history cap, at most one top-row run may lack support underneath;
/// that unsupported run can have at most three cells. Floor support is handled
/// by the caller, which skips this rule for a one-row cap.
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
            Self::InvalidConfiguration => "codec height must fit the configured playable board",
            Self::HeightExceeded => "board exceeds the codec height cap",
            Self::InvalidRowMask => "row has occupied cells outside C::COLS",
            Self::EmptyInternalRow => "board has an empty row below an occupied row",
            Self::FullRow => "board has an uncleared full row",
            Self::OccupancyResidue => "occupied cells violate the tetromino/line-clear residue",
            Self::TooManyHoles => "board exceeds the hole cap",
            Self::UnsupportedTopRun => "top row violates the capped-history support restriction",
            Self::ImpossibleBoard => "board has no possible last move under the codec height cap",
            Self::InvalidId => "ID is outside the codec's dense range",
        })
    }
}
impl std::error::Error for Error {}

/// Type-level dimensions sharing a compiled rule diagram.
pub struct Dimensions<const COLS: usize, const ROWS: usize, const MAX_HOLES: usize>;
pub type GameDimensions<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> =
    Dimensions<{ C::COLS }, ROWS, MAX_HOLES>;
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
    fn zero(self) -> u32;
    fn one(self) -> u32;
    fn weight(self) -> u64;
}
macro_rules! access {
    ($($ty:ty),+) => { $(impl Decision for $ty {
        #[inline(always)] fn zero(self)->u32 { u32::from(self.zero) }
        #[inline(always)] fn one(self)->u32 { u32::from(self.one) }
        #[inline(always)] fn weight(self)->u64 { u64::from(self.weight) }
    })+ };
}
access!(SmallNode, MediumNode, WideNode);
#[derive(Clone, Copy)]
pub(crate) enum Nodes {
    Small(&'static [SmallNode]),
    Medium(&'static [MediumNode]),
    Wide(&'static [WideNode]),
}
impl Nodes {
    fn payload_bytes(self) -> usize {
        match self {
            Self::Small(n) => std::mem::size_of_val(n),
            Self::Medium(n) => std::mem::size_of_val(n),
            Self::Wide(n) => std::mem::size_of_val(n),
        }
    }
}
#[doc(hidden)]
pub struct Tables {
    pub(crate) holes: usize,
    pub(crate) root: u32,
    pub(crate) count: u64,
    pub(crate) nodes: Nodes,
    pub(crate) levels: &'static [u16],
}

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
    fn rank(&self, bit: impl FnMut(usize) -> bool, vars: usize) -> Option<u64> {
        match self.nodes {
            Nodes::Small(n) => rank_nodes(n, self.levels, self.root, self.count, vars, bit),
            Nodes::Medium(n) => rank_nodes(n, self.levels, self.root, self.count, vars, bit),
            Nodes::Wide(n) => rank_nodes(n, self.levels, self.root, self.count, vars, bit),
        }
    }
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
/// Dense IDs ordered lexicographically by column, top cell first.
/// Rules and completion counts are compiled from COLS, ROWS and MAX_HOLES.
/// The represented carrier excludes proved impossible patterns, but is not the exact reachable set.
pub type BoardCodec<C: TetrisGameConfig, const ROWS: usize, const MAX_HOLES: usize> =
    BoardCodecStorage<C, { C::COLS }, ROWS, MAX_HOLES>;
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct BoardCodecStorage<
    C: TetrisGameConfig,
    const COLS: usize,
    const ROWS: usize,
    const MAX_HOLES: usize,
> {
    tables: &'static Tables,
    config: PhantomData<C>,
}
impl<C: TetrisGameConfig, const COLS: usize, const ROWS: usize, const MAX_HOLES: usize>
    BoardCodecStorage<C, COLS, ROWS, MAX_HOLES>
where
    Dimensions<COLS, ROWS, MAX_HOLES>: CompiledConfig,
{
    const VALID_CONFIGURATION: bool =
        COLS == C::COLS && ROWS > 0 && ROWS <= C::ROWS && C::ROWS < 32;
    pub const BOARD_COUNT: u64 = {
        assert!(
            Self::VALID_CONFIGURATION,
            "codec height must fit the configured playable board"
        );
        <Dimensions<COLS, ROWS, MAX_HOLES> as CompiledConfig>::BOARD_COUNT
    };
    pub const ID_BITS: u32 = 64 - (Self::BOARD_COUNT - 1).leading_zeros();
    pub const COLS: usize = C::COLS;
    pub const OCCUPANCY_MODULUS: usize = occupancy_modulus(C::COLS);
    pub const fn new() -> Self {
        assert!(
            Self::VALID_CONFIGURATION,
            "codec height must fit the configured playable board"
        );
        Self {
            tables: <Dimensions<COLS, ROWS, MAX_HOLES> as CompiledConfig>::TABLES,
            config: PhantomData,
        }
    }
    pub const fn try_new() -> Result<Self, Error> {
        if !Self::VALID_CONFIGURATION {
            return Err(Error::InvalidConfiguration);
        }
        Ok(Self::new())
    }
    pub const fn required_bytes(bits_per_entry: u32) -> u128 {
        (Self::BOARD_COUNT as u128 * bits_per_entry as u128).div_ceil(8)
    }
    pub const fn len(&self) -> u64 {
        Self::BOARD_COUNT
    }
    pub const fn is_empty(&self) -> bool {
        Self::BOARD_COUNT == 0
    }
    pub const fn id_bits(&self) -> u32 {
        Self::ID_BITS
    }
    pub const fn effective_max_holes(&self) -> usize {
        self.tables.holes
    }
    pub fn table_payload_bytes(&self) -> usize {
        self.tables.nodes.payload_bytes() + std::mem::size_of_val(self.tables.levels)
    }
    fn rejection(&self, rows: &[u16; ROWS]) -> Error {
        self.validate(rows).err().unwrap_or(Error::ImpossibleBoard)
    }
    pub fn id_u64(&self, board: &TetrisBoard<C>) -> Result<u64, Error>
    where
        [(); C::COLS]:,
    {
        let cols = board.as_limbs();
        if cols.iter().any(|&col| col >> ROWS != 0) {
            return Err(Error::HeightExceeded);
        }
        let result = if COLS * ROWS <= 64 {
            let code = cols
                .iter()
                .fold(0u64, |code, &col| (code << ROWS) | u64::from(col));
            self.tables.rank(
                |level| code & (1 << (COLS * ROWS - 1 - level)) != 0,
                COLS * ROWS,
            )
        } else {
            self.tables.rank(
                |level| cols[level / ROWS] & (1 << (ROWS - 1 - level % ROWS)) != 0,
                COLS * ROWS,
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
    pub fn id_rows_u64(&self, rows: &[u16; ROWS]) -> Result<u64, Error> {
        let full = (1u16 << COLS) - 1;
        if rows.iter().any(|&row| row & !full != 0) {
            return Err(Error::InvalidRowMask);
        }
        self.tables
            .rank(
                |level| rows[ROWS - 1 - level % ROWS] & (1 << (level / ROWS)) != 0,
                COLS * ROWS,
            )
            .ok_or_else(|| self.rejection(rows))
    }
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
        if occupied as usize % Self::OCCUPANCY_MODULUS != 0 {
            return Err(Error::OccupancyResidue);
        }
        let mut seen = 0u16;
        let mut holes = 0;
        for &row in rows[..height].iter().rev() {
            holes += (seen & !row).count_ones() as usize;
            seen |= row;
        }
        if holes > self.tables.holes {
            return Err(Error::TooManyHoles);
        }
        if height == ROWS
            && height >= 2
            && !top_supported(rows[height - 1], rows[height - 2], C::COLS)
        {
            return Err(Error::UnsupportedTopRun);
        }
        Ok(height)
    }

    pub fn board_from_u64(&self, id: u64) -> Result<TetrisBoard<C>, Error>
    where
        [(); C::COLS]:,
    {
        if COLS * ROWS <= 64 {
            let mut code = 0u64;
            self.tables.select(id, COLS * ROWS, |_, one| {
                code = (code << 1) | u64::from(one)
            })?;
            Ok(TetrisBoard::<C>::from_limbs(std::array::from_fn(|x| {
                ((code >> ((COLS - 1 - x) * ROWS)) & ((1u64 << ROWS) - 1)) as u32
            })))
        } else {
            let mut cols = [0u32; C::COLS];
            self.tables.select(id, COLS * ROWS, |level, one| {
                cols[level / ROWS] |= u32::from(one) << (ROWS - 1 - level % ROWS)
            })?;
            Ok(TetrisBoard::<C>::from_limbs(cols))
        }
    }
    pub fn rows_from_u64(&self, id: u64) -> Result<[u16; ROWS], Error> {
        let mut rows = [0; ROWS];
        self.tables.select(id, COLS * ROWS, |level, one| {
            rows[ROWS - 1 - level % ROWS] |= u16::from(one) << (level / ROWS)
        })?;
        Ok(rows)
    }
}
/// Native integer entry points. Selection depends only on the dimension count,
/// so even formatting a rejected constructor does not evaluate guarded constants.
impl<C: TetrisGameConfig, const COLS: usize, const ROWS: usize, const MAX_HOLES: usize>
    BoardCodecStorage<C, COLS, ROWS, MAX_HOLES>
where
    Dimensions<COLS, ROWS, MAX_HOLES>: IdConfig,
{
    /// Storage width of the selected integer, independent of packed `ID_BITS`.
    pub const ID_BYTES: usize =
        std::mem::size_of::<crate::id::DimensionId<COLS, ROWS, MAX_HOLES>>();

    /// Encode using this configuration's smallest supported integer type.
    pub fn id(
        &self,
        board: &TetrisBoard<C>,
    ) -> Result<crate::id::DimensionId<COLS, ROWS, MAX_HOLES>, Error>
    where
        [(); C::COLS]:,
    {
        Self::narrow_id(self.id_u64(board)?)
    }

    /// Encode bottom-first rows using the selected integer type.
    pub fn id_rows(
        &self,
        rows: &[u16; ROWS],
    ) -> Result<crate::id::DimensionId<COLS, ROWS, MAX_HOLES>, Error> {
        Self::narrow_id(self.id_rows_u64(rows)?)
    }

    fn narrow_id(id: u64) -> Result<crate::id::DimensionId<COLS, ROWS, MAX_HOLES>, Error> {
        crate::id::DimensionId::<COLS, ROWS, MAX_HOLES>::try_from(id).map_err(|_| Error::InvalidId)
    }

    /// Recover the exact native board from this configuration's integer type.
    pub fn board_from_id(
        &self,
        id: crate::id::DimensionId<COLS, ROWS, MAX_HOLES>,
    ) -> Result<TetrisBoard<C>, Error>
    where
        [(); C::COLS]:,
    {
        self.board_from_u64(id.into())
    }

    /// Recover bottom-first rows from this configuration's integer type.
    pub fn rows_from_id(
        &self,
        id: crate::id::DimensionId<COLS, ROWS, MAX_HOLES>,
    ) -> Result<[u16; ROWS], Error> {
        self.rows_from_u64(id.into())
    }
}

impl<C: TetrisGameConfig, const COLS: usize, const ROWS: usize, const MAX_HOLES: usize> Default
    for BoardCodecStorage<C, COLS, ROWS, MAX_HOLES>
where
    Dimensions<COLS, ROWS, MAX_HOLES>: CompiledConfig,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C: TetrisGameConfig, const COLS: usize, const ROWS: usize, const MAX_HOLES: usize> fmt::Debug
    for BoardCodecStorage<C, COLS, ROWS, MAX_HOLES>
where
    Dimensions<COLS, ROWS, MAX_HOLES>: CompiledConfig,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Result::unwrap_err and Result's Debug instantiate this even when
        // try_new rejected C::ROWS. Do not evaluate the guarded BOARD_COUNT.
        let dimension_count = <Dimensions<COLS, ROWS, MAX_HOLES> as CompiledConfig>::BOARD_COUNT;
        f.debug_struct("BoardCodec")
            .field("columns", &C::COLS)
            .field("rows", &ROWS)
            .field("max_holes", &MAX_HOLES)
            .field("board_count", &dimension_count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Game, admitted};
    use tetris_game::{StandardTetris, TetrisPiecePlacement};
    type Standard = BoardCodec<StandardTetris, 4, 2>;
    const CODEC: Standard = Standard::new();
    fn random(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    fn exhaustive<C: TetrisGameConfig, const R: usize, const H: usize>()
    where
        GameDimensions<C, R, H>: IdConfig,
        [(); C::COLS]:,
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
                let b = codec.board_from_u64(id).unwrap();
                assert_eq!(codec.id_u64(&b), Ok(id));
            }
        }
        assert!(seen.into_iter().all(|x| x));
        assert_eq!(codec.rows_from_u64(codec.len()), Err(Error::InvalidId));
    }
    #[test]
    fn exhaustive_small_carriers_match_engine_rule_with_dense_coverage() {
        exhaustive::<Game<2, 20>, 3, 1>();
        exhaustive::<Game<3, 20>, 3, 2>();
        exhaustive::<Game<4, 20>, 2, 1>();
        exhaustive::<Game<5, 20>, 2, 1>();
        exhaustive::<Game<6, 20>, 2, 1>();
        exhaustive::<StandardTetris, 1, 0>();
    }
    fn sampled<C: TetrisGameConfig, const R: usize, const H: usize>()
    where
        GameDimensions<C, R, H>: IdConfig,
        [(); C::COLS]:,
    {
        let codec = BoardCodec::<C, R, H>::new();
        let mut seed = 0xb04d_c0de;
        for id in [0, 1, codec.len() / 2, codec.len() - 1]
            .into_iter()
            .chain((0..512).map(|_| random(&mut seed) % codec.len()))
        {
            let rows = codec.rows_from_u64(id).unwrap();
            assert!(admitted::<C, R, H>(&rows));
            assert_eq!(codec.id_rows_u64(&rows), Ok(id));
            let native = codec.board_from_u64(id).unwrap();
            assert_eq!(codec.id_u64(&native), Ok(id));
            let typed = codec.id(&native).unwrap();
            assert_eq!(codec.board_from_id(typed), Ok(native));
            let reflected = std::array::from_fn(|y| rows[y].reverse_bits() >> (16 - C::COLS));
            let mirror = codec.id_rows_u64(&reflected).unwrap();
            assert_eq!(codec.rows_from_u64(mirror), Ok(reflected));
            assert_eq!(mirror == id, reflected == rows);
        }
    }
    #[test]
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
        assert_eq!(
            std::mem::size_of::<Standard>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(Standard::required_bytes(64), 3_139_450_768);
        assert_eq!(CODEC.id_rows_u64(&[0; 4]), Ok(0));
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
                    CODEC.id_u64(&TetrisBoard::<StandardTetris>::from_limbs(cols)),
                    Err(Error::HeightExceeded)
                );
            }
        }
        assert_eq!(CODEC.board_from_u64(u64::MAX), Err(Error::InvalidId));
        type Invalid = BoardCodec<Game<10, 2>, 4, 2>;
        let error = Invalid::try_new().unwrap_err();
        assert_eq!(error, Error::InvalidConfiguration);
        assert!(!format!("{error:?}").is_empty());
    }
    #[test]
    fn mutations_match_independent_cell_and_engine_oracle() {
        let mut s = 0x58a1_2095;
        for _ in 0..2048 {
            let mut rows = CODEC.rows_from_u64(random(&mut s) % CODEC.len()).unwrap();
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
                        TetrisBoard::<StandardTetris>::from_limbs(std::array::from_fn(|x| {
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
            let mut board = TetrisBoard::<StandardTetris>::new();
            for _ in 0..32 {
                let mut choices = Vec::new();
                for p in TetrisPiecePlacement::ALL_PLACEMENTS {
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
                let id = CODEC.id_u64(&board).unwrap();
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
        let Nodes::Medium(nodes) = CODEC.tables.nodes else {
            return Err("expected medium nodes".into());
        };
        let mut seed = 0xb04d_c0de;
        let mut codes = Vec::new();
        for _ in 0..256 {
            let board = CODEC
                .board_from_u64(random(&mut seed) % CODEC.len())
                .unwrap();
            codes.push(
                board
                    .as_limbs()
                    .iter()
                    .fold(0u64, |code, &col| (code << 4) | u64::from(col)),
            );
        }
        for &code in &codes {
            let a = rank_nodes(
                nodes,
                &levels,
                CODEC.tables.root,
                CODEC.len(),
                40,
                |level| code & (1 << (39 - level)) != 0,
            );
            let b = rank_nodes(nodes, &[], CODEC.tables.root, CODEC.len(), 40, |level| {
                code & (1 << (39 - level)) != 0
            });
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
                        black_box(CODEC.tables.root),
                        black_box(CODEC.len()),
                        40,
                        |level| code & (1 << (39 - level)) != 0,
                    )
                    .unwrap();
                    checksum = checksum.wrapping_add(black_box(rank));
                }
            }
            println!(
                "{{\"trial\":{trial},\"generic\":{generic},\"ns_per_call\":{},\"checksum\":{checksum}}}",
                start.elapsed().as_nanos() as f64 / (4096.0 * 256.0)
            );
        }
        Ok(())
    }
}
