//! Exact, memory-focused membership for column-major Tetris boards.
//!
//! [`BoardSet`] stores each unique whole board in a compressed, append-only arena.
//! A compact hash directory locates candidates; complete column comparisons keep
//! membership exact under hash collisions. Every backing bit is significant.
//! Bag state is outside the key.
//!
//! Use [`BoardSet::insert`] directly for first-visit processing. Sorted
//! [`iteration`](BoardSet::iter) allocates temporary references on demand, favoring
//! occasional export over retaining an ordering index. [`BoardSet::clear`] keeps
//! allocations for reuse; dropping the set releases them.
//!
//! Empty sets and capacity hints can be constructed in const contexts:
//!
//! ```
//! use tetris_board_set::BoardSet;
//!
//! const INITIAL: BoardSet<10> = BoardSet::with_capacity(1024);
//!
//! let mut seen = INITIAL;
//! if seen.insert([0; 10]) {
//!     assert!(seen.contains(&[0; 10]));
//! }
//! ```
#![warn(missing_docs)]

mod arena;
mod lookup;

use arena::Arena;

use tetris_game::constants::BackingColType as Limb;
use tetris_game::{StandardTetris, TetrisGameConfig};

/// An exact board set with immutable compressed records and direct hash lookup.
///
/// Zero-valued columns omit payload bytes; other columns use byte or half-byte
/// widths. Hash fingerprints locate candidates, and complete decoded columns
/// verify them.
/// Sorted iteration builds temporary compact references on demand; membership
/// retains no ordering index.
/// Bag state is outside the key. Every backing bit remains significant.
///
/// ```
/// use tetris_board_set::BoardSet;
///
/// let mut seen = BoardSet::new();
/// assert!(seen.insert([0; 10]));
/// assert!(!seen.insert([0; 10]));
/// assert!(seen.contains(&[0; 10]));
/// assert_eq!(seen.iter().collect::<Vec<_>>(), vec![[0; 10]]);
/// ```
///
/// Membership operations require the shared unroll macro's supported widths,
/// currently `0..=256`; larger widths fail at compile time.
///
/// ```compile_fail
/// let mut boards = tetris_board_set::BoardSet::<257>::new();
/// boards.insert([0; 257]);
/// ```
pub struct BoardSet<const N: usize = { StandardTetris::COLS }> {
    /// Number of unique whole boards, including the zero-width board if present.
    len: usize,
    /// Single immutable record per distinct board.
    arena: Arena<N>,
    /// Fingerprint/offset directory; slots include vacancies.
    lookup: lookup::Lookup,
    /// Initial board-count hint applied when directory storage is first needed.
    requested: usize,
}

impl<const N: usize> BoardSet<N> {
    /// Construct an empty set without allocating.
    ///
    /// Usable in const and static initializers. Capacity bounds are the same as
    /// [`Self::with_capacity`].
    pub const fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Construct an empty set with a directory-capacity hint.
    ///
    /// This is const-compatible and allocates nothing. Directory slots and encoded
    /// records grow when unique boards arrive; the hint counts boards, not bytes.
    ///
    /// # Panics
    ///
    /// Panics if `N` exceeds `u32::MAX`. Membership operations additionally require
    /// the shared unroll macro's supported widths, currently `0..=256`.
    pub const fn with_capacity(capacity: usize) -> Self {
        assert!(N <= u32::MAX as usize);
        Self {
            len: 0,
            arena: Arena::new(),
            lookup: lookup::Lookup::new(),
            requested: capacity,
        }
    }

    /// Return an exact occupied slot or the first vacant slot in its probe chain.
    ///
    /// Fingerprint matches only filter candidates; decoded columns decide equality.
    fn locate(&self, key: &[Limb; N], hash: u64) -> Result<usize, usize> {
        if self.lookup.len() == 0 {
            return Err(0);
        }
        let slots = self.lookup.len();
        let mut slot = self.lookup.start(hash);
        let tag = lookup::tag(hash);
        loop {
            let Some((fingerprint, offset)) = self.lookup.entry(slot) else {
                return Err(slot);
            };
            if fingerprint == tag && self.arena.matches(offset, key) {
                return Ok(slot);
            }
            slot += 1;
            if slot == slots {
                slot = 0;
            }
        }
    }

    /// Grow and rebuild the directory when its occupancy threshold is reached.
    ///
    /// Returns whether bucket mapping changed, invalidating a previously vacant slot.
    fn prepare_lookup(&mut self) -> bool {
        let old_capacity = self.lookup.len();
        if old_capacity != 0 && self.len < old_capacity / 8 * 7 {
            return false;
        }
        let capacity = if old_capacity == 0 {
            lookup::capacity_for(self.requested)
        } else {
            lookup::growth_capacity(old_capacity, self.len + 1)
        };
        // Every immutable arena record is a current member: there are no
        // individual deletions, and clear resets both structures. Rebuild in
        // append order instead of following scattered old directory entries.
        // Clear before resize so each old/new slot is zeroed only once.
        self.lookup.clear();
        self.lookup.resize(capacity);
        let mut offset = 0;
        // Count records rather than bytes: N=0 has a zero-byte record.
        for _ in 0..self.len {
            let key = self.arena.key(offset);
            let next = offset + self.arena.record_len(offset);
            let hash = lookup::hash(&key);
            let slot = self.lookup.vacant(hash);
            self.lookup.publish(slot, hash, offset);
            offset = next;
        }
        debug_assert_eq!(offset, self.arena.used);
        true
    }

    /// Insert a whole board, returning `true` only when it was absent.
    ///
    /// Use `if seen.insert(board) { ... }` to process a board on its first visit.
    /// A preceding [`Self::contains`] repeats the membership probe on misses.
    /// Duplicate insertion leaves the arena and directory capacities unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_board_set::BoardSet;
    ///
    /// let mut seen = BoardSet::new();
    /// assert!(seen.insert([0; 10]));
    /// assert!(!seen.insert([0; 10]));
    /// assert_eq!(seen.len(), 1);
    /// ```
    pub fn insert(&mut self, key: [Limb; N]) -> bool {
        let hash = lookup::hash(&key);
        let vacant = match self.locate(&key, hash) {
            Ok(_) => return false,
            Err(slot) => slot,
        };
        // Reuse the miss position unless resizing changed the bucket mapping.
        let slot = if self.prepare_lookup() {
            self.lookup.vacant(hash)
        } else {
            vacant
        };
        let offset = self.arena.append(&key);
        self.lookup.publish(slot, hash, offset);
        self.len += 1;
        true
    }

    /// Test exact membership without modifying or allocating set storage.
    ///
    /// Every backing bit participates; fingerprints alone are never accepted.
    pub fn contains(&self, key: &[Limb; N]) -> bool {
        self.len != 0 && self.locate(key, lookup::hash(key)).is_ok()
    }

    /// Number of distinct boards.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no boards are present.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Remove every board while retaining arena and directory allocations.
    ///
    /// Subsequent insertion reuses that storage. Drop the set to release it.
    pub fn clear(&mut self) {
        if self.len == 0 {
            return;
        }
        self.lookup.clear();
        self.arena.used = 0;
        self.len = 0;
    }

    /// Collect and sort temporary references without retaining ordering state.
    ///
    /// `pack` and `unpack` must preserve every live arena offset.
    fn sorted_offsets<T: Copy>(
        &self,
        pack: impl Fn(usize) -> T,
        unpack: impl Fn(T) -> usize,
    ) -> std::vec::IntoIter<T> {
        let mut offsets = Vec::with_capacity(self.len);
        if self.is_empty() {
            return offsets.into_iter();
        }
        for slot in 0..self.lookup.len() {
            if let Some((_, offset)) = self.lookup.entry(slot) {
                offsets.push(pack(offset));
            }
        }
        debug_assert_eq!(offsets.len(), self.len);
        offsets
            .sort_unstable_by(|&a, &b| self.arena.compare(unpack(a), &self.arena.key(unpack(b))));
        offsets.into_iter()
    }

    /// Reconstruct owned boards in lexicographic order.
    ///
    /// Construction scans the lookup table and sorts one temporary reference per
    /// board: O(slots + len * log(len)) comparisons and 3/4/usize bytes per board.
    /// Each live iterator owns its scratch storage, released when it is dropped.
    /// Intended for occasional export/inspection rather than hot-loop iteration.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "The selected offset variant is guarded by the arena extent: below 2^24 or at most u32::MAX."
    )]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = [Limb; N]> + '_ {
        // Nonempty records start before the logical end; zero-width records use zero.
        // An arena ending at the packed limit therefore still fits.
        let offsets = if self.arena.used <= PACKED_OFFSET_LIMIT {
            Offsets::Packed(self.sorted_offsets(
                |offset| {
                    let b = (offset as u32).to_le_bytes();
                    [b[0], b[1], b[2]]
                },
                unpack24,
            ))
        } else if self.arena.used <= u32::MAX as usize {
            Offsets::Compact(self.sorted_offsets(|x| x as u32, |x| x as usize))
        } else {
            Offsets::Wide(self.sorted_offsets(|x| x, |x| x))
        };
        Iter {
            offsets,
            arena: &self.arena,
        }
    }
}

impl<const N: usize> Default for BoardSet<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Bytes retained per iterator reference in the fixed three-byte representation.
const PACKED_OFFSET_BYTES: usize = 3;

/// Bits available in a packed iterator reference.
const PACKED_OFFSET_BITS: usize = PACKED_OFFSET_BYTES * u8::BITS as usize;

/// Exclusive upper bound of an individual packed iterator offset.
const PACKED_OFFSET_LIMIT: usize = 1 << PACKED_OFFSET_BITS;

/// Temporary ordering references owned by an iterator, using the narrowest offsets.
///
/// Membership retains none of this ordering storage.
enum Offsets {
    Packed(std::vec::IntoIter<[u8; PACKED_OFFSET_BYTES]>),
    Compact(std::vec::IntoIter<u32>),
    Wide(std::vec::IntoIter<usize>),
}

/// Decode a little-endian 24-bit arena coordinate without sign extension.
const fn unpack24(bytes: [u8; PACKED_OFFSET_BYTES]) -> usize {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]) as usize
}

impl Iterator for Offsets {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Packed(v) => v.next().map(unpack24),
            Self::Compact(v) => v.next().map(|x| x as usize),
            Self::Wide(v) => v.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            Self::Packed(v) => v.len(),
            Self::Compact(v) => v.len(),
            Self::Wide(v) => v.len(),
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for Offsets {}

/// Reconstruct owned boards from sorted offsets while borrowing the arena.
struct Iter<'a, const N: usize> {
    offsets: Offsets,
    arena: &'a Arena<N>,
}

impl<const N: usize> Iterator for Iter<'_, N> {
    type Item = [Limb; N];

    fn next(&mut self) -> Option<Self::Item> {
        self.offsets.next().map(|offset| self.arena.key(offset))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<const N: usize> ExactSizeIterator for Iter<'_, N> {}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "Reference bit streams mask individual bits; test coordinates and width fields are bounded to 256, and seeded values intentionally use low bits."
)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use crate::arena::tests::reference_record;

    fn validate<const N: usize>(set: &BoardSet<N>) {
        let mut keys = BTreeSet::new();
        let mut offsets = BTreeSet::new();
        for slot in 0..set.lookup.len() {
            if let Some((tag, offset)) = set.lookup.entry(slot) {
                let key = set.arena.key(offset);
                assert!(keys.insert(key));
                assert!(offsets.insert(offset));
                assert_eq!(lookup::tag(lookup::hash(&key)), tag);
                assert_eq!(set.locate(&key, lookup::hash(&key)), Ok(slot));
            }
        }
        assert_eq!(keys.len(), set.len);
        assert!(set.iter().eq(keys.iter().copied()));
        let mut cursor = 0;
        for &offset in &offsets {
            assert_eq!(offset, cursor);
            // Count the independently encoded bytes, not the production decoder's length.
            cursor += reference_record(&set.arena.key(offset)).len();
        }
        assert_eq!(cursor, set.arena.used);
        let mut occupied = 0;
        for slot in 0..set.lookup.len() {
            if let Some((tag, offset)) = set.lookup.entry(slot) {
                occupied += 1;
                assert!(offsets.contains(&offset));
                let key = set.arena.key(offset);
                assert_eq!(lookup::tag(lookup::hash(&key)), tag);
                assert_eq!(set.locate(&key, lookup::hash(&key)), Ok(slot));
            }
        }
        assert_eq!(occupied, set.len);
    }

    fn oracle<const N: usize>() {
        let mut set = BoardSet::<N>::new();
        let mut oracle = BTreeSet::new();
        let mut seed = 0xDEAD_BEEF_CAFE_u64;
        for round in 0..3 {
            for i in 0..4000 {
                let key = std::array::from_fn(|col| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    match round {
                        0 => (seed as u32) & 15,
                        1 => seed as u32,
                        _ => {
                            if col < N / 2 {
                                0
                            } else {
                                i
                            }
                        }
                    }
                });
                assert_eq!(set.insert(key), oracle.insert(key));
                assert!(!set.insert(key));
                assert!(set.contains(&key));
                let mut miss = key;
                if N != 0 {
                    miss[i as usize % N] ^= 1 << (i % 32);
                }
                assert_eq!(set.contains(&miss), oracle.contains(&miss));
                if i % 257 == 0 {
                    validate(&set);
                }
            }
            validate(&set);
            assert_eq!(set.len(), oracle.len());
            let mut iter = set.iter();
            assert_eq!(iter.len(), oracle.len());
            for expected in &oracle {
                assert_eq!(iter.next(), Some(*expected));
            }
            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next(), None);
            drop(iter);
            set.clear();
            set.clear();
            oracle.clear();
            assert!(set.is_empty());
            assert!(!set.contains(&[0; N]));
        }
    }

    #[test]
    fn const_construction_keeps_storage_lazy_and_allows_runtime_insertion() {
        static EMPTY: BoardSet<10> = BoardSet::new();
        const INITIAL_LEN: usize = EMPTY.len();
        const INITIALLY_EMPTY: bool = EMPTY.is_empty();
        const RESERVED: BoardSet<10> = BoardSet::with_capacity(128);

        assert_eq!(INITIAL_LEN, 0);
        const { assert!(INITIALLY_EMPTY) };
        let mut set = RESERVED;
        assert_eq!(set.requested, 128);
        assert_eq!(set.lookup.len(), 0);
        assert_eq!(set.arena.bytes.capacity(), 0);
        assert!(set.insert([0; 10]));
        assert!(set.lookup.len() >= 128);
        assert!(!set.insert([0; 10]));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn initialized_capacity_survives_clear_and_poisoned_refills() {
        let mut set = BoardSet::<10>::new();
        let first = [1; 10];
        assert!(set.insert(first));
        let capacity = set.arena.bytes.capacity();
        assert_eq!(set.arena.bytes.len(), capacity);
        set.clear();
        assert_eq!(set.arena.bytes.len(), capacity);
        assert_eq!(set.arena.used, 0);
        set.arena.bytes.fill(0xA5);
        let next = [0x1FF; 10];
        assert!(set.insert(next));
        assert!(!set.contains(&first));
        assert!(set.contains(&next));
        assert_eq!(&set.arena.bytes[..set.arena.used], reference_record(&next));
        validate(&set);
        set.clear();
        set.arena.bytes.fill(0xFF);
        assert!(set.insert([0; 10]));
        assert_eq!(
            &set.arena.bytes[..set.arena.used],
            reference_record(&[0; 10])
        );
        validate(&set);
    }

    #[test]
    fn exact_oracles_and_reuse() {
        oracle::<0>();
        oracle::<1>();
        oracle::<2>();
        oracle::<10>();
        oracle::<11>();
    }

    #[test]
    fn order_independent_insertion_and_all_bits() {
        for reverse in [false, true] {
            let mut set = BoardSet::<2>::with_capacity(2000);
            let mut keys: Vec<_> = (0..2000).map(|i| [i, u32::MAX - i]).collect();
            if reverse {
                keys.reverse();
            }
            for &key in &keys {
                assert!(set.insert(key));
            }
            keys.sort_unstable();
            assert_eq!(set.iter().collect::<Vec<_>>(), keys);
            assert!(set.insert([u32::MAX; 2]));
            assert!(set.contains(&[u32::MAX; 2]));
            assert!(!set.contains(&[u32::MAX, 0]));
        }
    }

    #[test]
    fn large_growth_partial_iteration_and_smaller_refill_are_exact() {
        let mut set = BoardSet::<10>::new();
        let keys: Vec<_> = (0..20_000)
            .map(|i| {
                let mut key = [0x55; 10];
                key[8] = i / 31;
                key[9] = i;
                key
            })
            .collect();
        for &key in &keys {
            assert!(set.insert(key));
        }
        validate(&set);
        let mut iter = set.iter();
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(iter.len(), keys.len() - i);
            assert_eq!(iter.size_hint(), (keys.len() - i, Some(keys.len() - i)));
            assert_eq!(iter.next(), Some(*key));
        }
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        drop(iter);
        let mut partial = set.iter();
        assert_eq!(partial.nth(127), Some(keys[127]));
        assert_eq!(partial.len(), keys.len() - 128);
        drop(partial);
        set.clear();
        set.clear();
        for i in (0..100).rev() {
            assert!(set.insert([i; 10]));
        }
        validate(&set);
        assert_eq!(set.len(), 100);
        assert!(!set.contains(&keys[19_999]));
        assert!(set.iter().eq((0..100).map(|i| [i; 10])));
    }

    #[test]
    fn payload_is_append_only_through_new_minima_and_hash_growth() {
        let mut set = BoardSet::<10>::new();
        for i in (0..1025u32).rev() {
            let old = set.arena.bytes[..set.arena.used].to_vec();
            assert!(set.insert([i; 10]));
            assert_eq!(&set.arena.bytes[..old.len()], old);
            if i % 128 == 0 {
                validate(&set);
            }
        }
        assert!(set.iter().eq((0..1025u32).map(|i| [i; 10])));
        let capacity = set.arena.bytes.capacity();
        let slots = set.lookup.len();
        set.clear();
        set.clear();
        assert_eq!(set.arena.bytes.capacity(), capacity);
        assert_eq!(set.lookup.len(), slots);
        assert_eq!(set.iter().len(), 0);
        assert!(!set.contains(&[0; 10]));
        assert!(set.insert([0; 10]));
        assert_eq!(set.arena.bytes.capacity(), capacity);
        validate(&set);
    }

    #[test]
    fn matching_fingerprints_are_not_accepted_without_exact_comparison() {
        let mut colliders = Vec::new();
        let table = {
            let mut lookup = lookup::Lookup::new();
            lookup.resize(16);
            lookup
        };
        for i in 0..100_000u32 {
            let key = [i, i.rotate_left(13) ^ 0xA5A5_5A5A];
            let hash = lookup::hash(&key);
            if table.start(hash) == 0 && lookup::tag(hash) == 1 {
                colliders.push(key);
                if colliders.len() == 9 {
                    break;
                }
            }
        }
        assert_eq!(colliders.len(), 9);
        let mut set = BoardSet::<2>::with_capacity(1);
        for &key in &colliders[..8] {
            assert!(set.insert(key));
        }
        assert_eq!(set.lookup.len(), 16);
        assert!(!set.contains(&colliders[8]));
        for key in &colliders[..8] {
            assert!(set.contains(key));
            assert!(!set.insert(*key));
        }
        validate(&set);
    }

    #[test]
    fn arena_rebuild_preserves_colliding_wrapped_probe_chains() {
        let table = {
            let mut lookup = lookup::Lookup::new();
            lookup.resize(16);
            lookup
        };
        let mut colliders = Vec::new();
        for i in 0..100_000u32 {
            let key = [i, i.rotate_left(13) ^ 0xA5A5_5A5A];
            let hash = lookup::hash(&key);
            if table.start(hash) == 15 && lookup::tag(hash) == 1 {
                colliders.push(key);
                if colliders.len() == 129 {
                    break;
                }
            }
        }
        assert_eq!(colliders.len(), 129);
        let mut set = BoardSet::<2>::new();
        for (i, &key) in colliders[..128].iter().enumerate() {
            assert!(set.insert(key));
            assert_eq!(set.len(), i + 1);
            for previous in &colliders[..=i] {
                assert!(set.contains(previous));
            }
            assert!(!set.contains(&colliders[128]));
            validate(&set);
        }
    }

    #[test]
    fn dense_tables_keep_vacancies_through_growth_and_clear() {
        for hint in [0, 1, 15, 16, 17, 63, 64, 65, 511, 1025] {
            let mut set = BoardSet::<2>::with_capacity(hint);
            let mut oracle = BTreeSet::new();
            for i in 0..2048u32 {
                let key = [i.rotate_left(13), i ^ 0xA5A5_5A5A];
                assert_eq!(set.insert(key), oracle.insert(key));
                assert!(set.len() < set.lookup.len());
                assert!(set.contains(&key));
                if i % 257 == 0 {
                    validate(&set);
                }
            }
            validate(&set);
            let capacity = set.lookup.len();
            set.clear();
            for key in oracle {
                assert!(set.insert(key));
            }
            assert_eq!(set.lookup.len(), capacity);
            validate(&set);
        }
    }

    #[test]
    fn duplicate_at_growth_threshold_does_not_allocate_or_append() {
        let mut set = BoardSet::<2>::new();
        for i in 0..14 {
            assert!(set.insert([i, i * 17]));
        }
        let entries: Vec<_> = (0..set.lookup.len())
            .map(|slot| set.lookup.entry(slot))
            .collect();
        let bytes = set.arena.bytes.clone();
        let capacity = set.arena.bytes.capacity();
        assert!(!set.insert([13, 13 * 17]));
        assert_eq!(set.lookup.len(), 16);
        assert_eq!(set.arena.bytes, bytes);
        assert_eq!(set.arena.bytes.capacity(), capacity);
        assert!(
            entries
                .iter()
                .enumerate()
                .all(|(slot, &entry)| set.lookup.entry(slot) == entry)
        );
        assert!(set.insert([14, 14 * 17]));
        assert_eq!(set.lookup.len(), 32);
        validate(&set);
    }

    #[test]
    fn temporary_offset_widths_preserve_all_coordinate_bits() {
        let mut packed =
            Offsets::Packed(vec![[0, 0, 0], [255, 0, 0], [0, 1, 0], [255; 3]].into_iter());
        assert_eq!(packed.len(), 4);
        assert_eq!(packed.next(), Some(0));
        assert_eq!(packed.len(), 3);
        assert_eq!(packed.collect::<Vec<_>>(), [255, 256, (1 << 24) - 1]);
        let compact = Offsets::Compact(vec![1 << 24, u32::MAX].into_iter());
        assert_eq!(compact.collect::<Vec<_>>(), [1 << 24, u32::MAX as usize]);
        let wide = Offsets::Wide(vec![usize::MAX].into_iter());
        assert_eq!(wide.collect::<Vec<_>>(), [usize::MAX]);
    }

    #[test]
    fn independently_owned_sorted_iterators_preserve_partial_progress() {
        let mut set = BoardSet::<2>::new();
        let expected: Vec<_> = (0..1025u32).map(|i| [i, !i]).collect();
        for &key in expected.iter().rev() {
            assert!(set.insert(key));
        }
        let mut first = set.iter();
        assert_eq!(first.next(), Some(expected[0]));
        assert_eq!(first.len(), expected.len() - 1);
        let second = set.iter();
        assert!(second.eq(expected.iter().copied()));
        assert!(first.eq(expected[1..].iter().copied()));
        let mut exhausted = set.iter();
        assert_eq!(exhausted.nth(expected.len()), None);
        assert_eq!(exhausted.next(), None);
        assert_eq!(exhausted.len(), 0);
    }
    #[test]
    fn header_word_boundary_and_byte_crossings_preserve_every_column() {
        fn check<const N: usize>() {
            let mut set = BoardSet::<N>::new();
            let mut oracle = BTreeSet::new();
            for step in 0..513u32 {
                let key = std::array::from_fn(|i| match (i + step as usize) % 5 {
                    0 => 0,
                    1 => 255 ^ step,
                    2 => 256 ^ step,
                    3 => 65536 ^ step,
                    _ => u32::MAX ^ step,
                });
                assert_eq!(set.contains(&key), oracle.contains(&key));
                assert_eq!(set.insert(key), oracle.insert(key));
                assert!(set.contains(&key));
                if N > 0 {
                    let mut query = key;
                    query[step as usize % N] ^= 1 << (step % 32);
                    assert_eq!(set.contains(&query), oracle.contains(&query));
                }
            }
            validate(&set);
        }
        check::<0>();
        check::<1>();
        check::<21>();
        check::<22>();
        check::<33>();
        check::<127>();
        check::<128>();
        check::<129>();
        check::<200>();
    }
    #[test]
    fn exhaustive_small_universe_survives_reordering_and_clear() {
        let values = [0, 15, 16, 255, 256, u32::MAX];
        let universe: Vec<_> = values
            .into_iter()
            .flat_map(|a| {
                values
                    .into_iter()
                    .flat_map(move |b| values.map(|c| [a, b, c]))
            })
            .collect();
        for capacity in [0, 1, 16, 127] {
            let mut set = BoardSet::<3>::with_capacity(capacity);
            for round in 0..3 {
                set.clear();
                set.clear();
                let mut oracle = BTreeSet::new();
                for (index, &key) in universe.iter().rev().enumerate() {
                    if index % 3 != round {
                        assert_eq!(set.insert(key), oracle.insert(key));
                        assert!(!set.insert(key));
                    }
                }
                for key in &universe {
                    assert_eq!(set.contains(key), oracle.contains(key));
                    let mut query = *key;
                    query[round] ^= 1 << 31;
                    assert_eq!(set.contains(&query), oracle.contains(&query));
                }
                assert_eq!(set.len(), oracle.len());
                assert!(set.iter().eq(oracle.iter().copied()));
                validate(&set);
            }
        }
    }
}
