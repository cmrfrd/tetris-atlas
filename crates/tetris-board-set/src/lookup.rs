//! Compact fingerprint/offset directory for privately encoded arena records.
//!
//! Linear probing filters candidates with fingerprints. The caller always verifies
//! the complete encoded board before accepting membership.
use tetris_game::constants::BackingColType as Limb;

/// Each occupied slot carries the same five fingerprint bits in every representation.
const TAG_BITS: u32 = 5;

/// Low bits available for a byte offset in the common four-byte slot.
const COMPACT_OFFSET_BITS: u32 = u32::BITS - TAG_BITS;

/// Low bits available for a byte offset in the eight-byte slot.
const WIDE_OFFSET_BITS: u32 = u64::BITS - TAG_BITS;

/// Mask away the fingerprint when reading a four-byte slot.
const COMPACT_OFFSET_MASK: u32 = (1 << COMPACT_OFFSET_BITS) - 1;

/// Mask away the fingerprint when reading an eight-byte slot.
const WIDE_OFFSET_MASK: u64 = (1 << WIDE_OFFSET_BITS) - 1;

/// Hash every backing column for routing and fingerprint filtering.
pub(crate) fn hash<const N: usize>(key: &[Limb; N]) -> u64 {
    let mut even = 0x517C_C1B7_2722_0A95u64;
    let mut odd = 0xD6E8_FEB8_6659_FD93u64;
    let mut pairs = key.chunks_exact(2);
    for pair in pairs.by_ref() {
        even = (even ^ u64::from(pair[0]))
            .wrapping_mul(0x9E37_79B1_85EB_CA87)
            .rotate_left(7);
        odd = (odd ^ u64::from(pair[1]))
            .wrapping_mul(0x9E37_79B1_85EB_CA87)
            .rotate_left(7);
    }
    let mut hash = even ^ odd.rotate_left(29);
    for &col in pairs.remainder() {
        hash = (hash ^ u64::from(col)).wrapping_mul(0x9E37_79B1_85EB_CA87);
    }
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    hash ^ (hash >> 32)
}

/// Nonzero five-bit fingerprint, independent of the high-product bucket bits.
///
/// Zero marks a vacancy. Hash fingerprints zero and one intentionally share tag
/// one; the caller must still compare the complete record on every tag match.
pub(crate) fn tag(hash: u64) -> u8 {
    // Range reduction uses the high product bits for the starting bucket.
    // Keep the fingerprint in the low bits to avoid correlating both filters.
    ((hash & ((1 << TAG_BITS) - 1)) as u8).max(1)
}

/// Power-of-two slot bound with at least one eighth of the table vacant.
fn power_bound(needed: usize) -> usize {
    needed
        .max(1)
        .saturating_mul(8)
        .div_ceil(7)
        .next_power_of_two()
        .max(16)
}

/// Plan reserved directory capacity around 84% occupancy, rounded to slot groups.
pub(crate) fn capacity_for(needed: usize) -> usize {
    // Aim for 84% occupancy, within the unchanged 87.5% growth limit. Round
    // compact storage in 64-byte increments, capped by a dense power of two.
    // Compute ceil(25*n/21) without overflowing the intermediate product.
    let needed = needed.max(1);
    let extra = (needed / 21) * 4 + ((needed % 21) * 4).div_ceil(21);
    needed
        .saturating_add(extra)
        .next_multiple_of(16)
        .max(16)
        .min(power_bound(needed))
}

/// Grow by half, rounding to slot groups and capping at the next power bound.
pub(crate) fn growth_capacity(current: usize, needed: usize) -> usize {
    (current + current / 2)
        .next_multiple_of(16)
        .min(power_bound(needed))
}

/// Slot representations selected by the largest published arena offset.
enum Entries {
    Compact(Vec<u32>), // five nonzero fingerprint bits + 27 offset bits
    Wide(Vec<u64>),    // five fingerprint bits + 59 offset bits
    Full(Vec<(usize, u8)>),
}

/// Open-addressed slots storing fingerprints and stable logical arena offsets.
pub(crate) struct Lookup {
    /// Current slot width; promotion preserves every live fingerprint and offset.
    entries: Entries,
}

impl Lookup {
    /// Construct an empty directory without allocating.
    pub(crate) const fn new() -> Self {
        Self {
            entries: Entries::Compact(Vec::new()),
        }
    }

    /// Extend initialized slot storage without shrinking or changing its representation.
    pub(crate) fn resize(&mut self, slots: usize) {
        /// Grow initialized storage while preserving already published entries.
        fn grow<T: Copy>(entries: &mut Vec<T>, slots: usize, empty: T) {
            debug_assert!(slots >= entries.len());
            entries.reserve_exact(slots - entries.len());
            entries.resize(slots, empty);
        }
        match &mut self.entries {
            Entries::Compact(v) => grow(v, slots, 0),
            Entries::Wide(v) => grow(v, slots, 0),
            Entries::Full(v) => grow(v, slots, (0, 0)),
        }
    }

    /// Number of slots, including vacancies; this is not the member count.
    pub(crate) const fn len(&self) -> usize {
        match &self.entries {
            Entries::Compact(v) => v.len(),
            Entries::Wide(v) => v.len(),
            Entries::Full(v) => v.len(),
        }
    }

    /// Map a hash into the current slot range with multiply-high reduction.
    ///
    /// An empty directory returns zero and must not be probed.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "The high half of hash times slot count is strictly below the usize slot count."
    )]
    pub(crate) fn start(&self, hash: u64) -> usize {
        // Multiply-high maps hashes into arbitrary table lengths without a
        // runtime remainder. Empty tables map to zero and are never probed.
        ((u128::from(hash) * self.len() as u128) >> u64::BITS) as usize
    }

    /// Read an occupied slot's fingerprint and arena offset, or `None` for a vacancy.
    ///
    /// `slot` must be below [`Self::len`].
    pub(crate) fn entry(&self, slot: usize) -> Option<(u8, usize)> {
        match &self.entries {
            Entries::Compact(v) => {
                let e = v[slot];
                (e != 0).then_some((
                    (e >> COMPACT_OFFSET_BITS) as u8,
                    (e & COMPACT_OFFSET_MASK) as usize,
                ))
            }
            Entries::Wide(v) => {
                let e = v[slot];
                (e != 0).then_some((
                    (e >> WIDE_OFFSET_BITS) as u8,
                    (e & WIDE_OFFSET_MASK) as usize,
                ))
            }
            Entries::Full(v) => {
                let (offset, tag) = v[slot];
                (tag != 0).then_some((tag, offset))
            }
        }
    }

    /// Publish an offset and fingerprint, promoting slot width before truncation.
    ///
    /// `slot` must be a vacancy in the current directory. The arena must already
    /// contain the corresponding immutable board record.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "The compact variant is promoted before offsets exceed its 27-bit field."
    )]
    pub(crate) fn publish(&mut self, slot: usize, hash: u64, offset: usize) {
        let fits_compact = offset < 1 << COMPACT_OFFSET_BITS;
        let fits_wide = (offset as u64) < 1u64 << WIDE_OFFSET_BITS;
        let promote = match &self.entries {
            Entries::Compact(_) => !fits_compact,
            Entries::Wide(_) => !fits_wide,
            Entries::Full(_) => false,
        };
        if promote {
            let old = std::mem::replace(&mut self.entries, Entries::Compact(Vec::new()));
            self.entries = match old {
                Entries::Compact(v) if fits_wide => Entries::Wide(
                    v.into_iter()
                        .map(|e| {
                            if e == 0 {
                                0
                            } else {
                                (u64::from(e >> COMPACT_OFFSET_BITS) << WIDE_OFFSET_BITS)
                                    | u64::from(e & COMPACT_OFFSET_MASK)
                            }
                        })
                        .collect(),
                ),
                Entries::Compact(v) => Entries::Full(
                    v.into_iter()
                        .map(|e| {
                            (
                                (e & COMPACT_OFFSET_MASK) as usize,
                                (e >> COMPACT_OFFSET_BITS) as u8,
                            )
                        })
                        .collect(),
                ),
                Entries::Wide(v) => Entries::Full(
                    v.into_iter()
                        .map(|e| {
                            (
                                (e & WIDE_OFFSET_MASK) as usize,
                                (e >> WIDE_OFFSET_BITS) as u8,
                            )
                        })
                        .collect(),
                ),
                Entries::Full(v) => Entries::Full(v),
            };
        }
        let tag = tag(hash);
        match &mut self.entries {
            Entries::Compact(v) => {
                v[slot] = (u32::from(tag) << COMPACT_OFFSET_BITS) | offset as u32
            }
            Entries::Wide(v) => v[slot] = (u64::from(tag) << WIDE_OFFSET_BITS) | offset as u64,
            Entries::Full(v) => v[slot] = (offset, tag),
        }
    }

    /// Find the first vacancy in the hash's wrapped linear probe chain.
    ///
    /// The directory must be nonempty and contain at least one vacant slot.
    pub(crate) fn vacant(&self, hash: u64) -> usize {
        let slots = self.len();
        let mut slot = self.start(hash);
        while self.entry(slot).is_some() {
            slot += 1;
            if slot == slots {
                slot = 0;
            }
        }
        slot
    }

    /// Mark every slot vacant while retaining its allocation and representation.
    pub(crate) fn clear(&mut self) {
        match &mut self.entries {
            Entries::Compact(v) => v.fill(0),
            Entries::Wide(v) => v.fill(0),
            Entries::Full(v) => v.fill((0, 0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocate initialized slots for directory-focused tests.
    fn with_slots(slots: usize) -> Lookup {
        let mut lookup = Lookup::new();
        lookup.resize(slots);
        lookup
    }

    #[test]
    fn routing_hash_observes_every_bit_at_pair_and_quad_boundaries() {
        fn check<const N: usize>() {
            let mut key = [0; N];
            let mut hashes = std::collections::HashSet::new();
            assert!(hashes.insert(hash(&key)));
            for column in 0..N {
                for bit in 0..Limb::BITS {
                    key[column] = 1 << bit;
                    // Detect accidentally omitted columns, truncated upper
                    // halves and lost odd remainders independently of lookup.
                    assert!(
                        hashes.insert(hash(&key)),
                        "N={N}, column={column}, bit={bit}"
                    );
                }
                key[column] = 0;
            }
        }
        check::<0>();
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        check::<9>();
        check::<10>();
        check::<11>();
        check::<21>();
        check::<22>();
        check::<255>();
        check::<256>();
    }

    #[test]
    fn capacity_plans_keep_headroom_without_power_of_two_rounding_waste() {
        for needed in (0..10_000usize).chain((4..26).flat_map(|bits| {
            let edge = ((1usize << bits) / 8) * 7;
            [edge - 1, edge, edge + 1]
        })) {
            let slots = capacity_for(needed);
            assert!(slots >= 16 && slots.is_multiple_of(16));
            assert!(needed <= slots / 8 * 7);
            assert!(slots <= power_bound(needed));
            let old_target = needed
                .max(1)
                .saturating_mul(5)
                .div_ceil(4)
                .next_multiple_of(16)
                .max(16);
            assert!(slots <= old_target);
        }
        let mut slots = capacity_for(0);
        for _ in 0..40 {
            let needed = slots / 8 * 7 + 1;
            let next = growth_capacity(slots, needed);
            assert!(next > slots);
            assert!(needed <= next / 8 * 7);
            assert!(next <= power_bound(needed));
            assert!(next.is_multiple_of(16));
            slots = next;
        }
        assert!(capacity_for(1_000_000) < 1_400_000);
    }

    #[test]
    fn directory_growth_needs_at_most_two_rebuilds_per_doubling() {
        for bits in 4..27 {
            let capacity = 1usize << bits;
            let first = growth_capacity(capacity, capacity / 8 * 7 + 1);
            assert!(first > capacity && first <= capacity * 2);
            if first < capacity * 2 {
                let second = growth_capacity(first, first / 8 * 7 + 1);
                assert_eq!(second, capacity * 2);
            }
        }
    }

    #[test]
    fn arbitrary_lengths_map_in_range_and_probe_across_the_end() {
        for len in [0, 1, 16, 17, 63, 64, 65, 1031] {
            let lookup = with_slots(len);
            assert_eq!(lookup.start(0), 0);
            assert_eq!(lookup.start(u64::MAX), len.saturating_sub(1));
            for hash in [1, 1 << 32, 1 << 63, u64::MAX - 1] {
                assert!(len == 0 || lookup.start(hash) < len);
            }
        }
        let mut lookup = with_slots(17);
        for expected in [16, 0, 1] {
            let slot = lookup.vacant(u64::MAX);
            assert_eq!(slot, expected);
            lookup.publish(slot, u64::MAX, 100 + expected);
        }
        assert_eq!(lookup.vacant(u64::MAX), 2);
        lookup.clear();
        assert_eq!(lookup.vacant(u64::MAX), 16);
    }

    #[test]
    fn fingerprints_do_not_change_when_only_the_start_bucket_bits_change() {
        for low in 0..32u64 {
            assert_eq!(tag(low), tag(low | (1 << 63)));
        }
        assert_ne!(
            with_slots(1031).start(7),
            with_slots(1031).start(7 | (1 << 63))
        );
    }

    #[test]
    fn offset_promotions_preserve_every_fingerprint_and_coordinate() {
        let mut t = with_slots(8);
        t.publish(0, 0, 0);
        t.publish(1, u64::MAX, (1 << 27) - 1);
        assert_eq!(t.entry(0), Some((1, 0)));
        assert_eq!(t.entry(1), Some((31, (1 << 27) - 1)));
        t.publish(2, 7, 1 << 27);
        assert_eq!(t.entry(2), Some((7, 1 << 27)));
        assert_eq!(t.entry(0), Some((1, 0)));
        if usize::BITS > 59 {
            t.publish(3, 9, usize::MAX);
            assert_eq!(t.entry(3), Some((9, usize::MAX)));
            assert_eq!(t.entry(1), Some((31, (1 << 27) - 1)));
        }
        t.resize(17);
        assert_eq!(t.len(), 17);
        assert_eq!(t.entry(0), Some((1, 0)));
        assert_eq!(t.entry(1), Some((31, (1 << 27) - 1)));
        assert_eq!(t.entry(2), Some((7, 1 << 27)));
        assert!((4..17).all(|slot| t.entry(slot).is_none()));
        t.clear();
        assert!((0..17).all(|slot| t.entry(slot).is_none()));
    }
    #[test]
    fn every_tag_and_zero_offset_survive_direct_and_staged_promotion() {
        for staged in [false, true] {
            let mut lookup = with_slots(64);
            for fingerprint in 1..=31u8 {
                lookup.publish(usize::from(fingerprint), u64::from(fingerprint), 0);
            }
            if staged {
                lookup.publish(32, 0, 1 << 27);
            }
            lookup.publish(33, 0, usize::MAX);
            for fingerprint in 1..=31u8 {
                assert_eq!(
                    lookup.entry(usize::from(fingerprint)),
                    Some((fingerprint, 0))
                );
            }
            assert_eq!(lookup.entry(33), Some((1, usize::MAX)));
            assert!(lookup.entry(0).is_none());
            lookup.clear();
            assert!((0..64).all(|slot| lookup.entry(slot).is_none()));
            // Clearing retains the promoted representation but must not leave
            // stale offsets or treat the valid zero offset as a vacancy.
            lookup.publish(0, 0, 0);
            assert_eq!(lookup.entry(0), Some((1, 0)));
            assert_eq!(tag(0), tag(1));
        }
    }
}
