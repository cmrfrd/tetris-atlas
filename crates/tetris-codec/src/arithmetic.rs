#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "The one-hole domain has ten columns, R in 4..=6, and exponents at most ten. Heights and parity fit u8, column masks fit u32, and signed inclusion-exclusion expressions count nonnegative subsets, with assertions for the impossible-family totals."
)]
//! Algebraic ranking for bounded board families, without a decision graph.
//!
//! Ten-column codecs with caps 4..=6 and at most one hole use hole metadata followed
//! by lexicographic column heights. Shared const arrays hold powers, signed recurrence
//! counts, metadata offsets and column descriptions.
//!
//! No board catalog or excluded-ID list is retained.

use tetris_macros::inline_conditioned;

/// Powers for height-alphabet sizes up to seven and suffix lengths up to ten.
/// The largest entry, 7^10, fits in `u32`; the shared table occupies 352 bytes.
const POWERS: [[u32; 11]; 8] = {
    let mut powers = [[1; 11]; 8];
    let mut base = 0;
    while base < 8 {
        let mut n = 1;
        while n < 11 {
            powers[base][n] = powers[base][n - 1] * base as u32;
            n += 1;
        }
        base += 1;
    }

    powers
};

/// Packs height in bits 0..=2 and hole-row-plus-one in bits 3..=5; bit 7 rejects the column.
#[inline_conditioned]
const fn make_columns<const R: usize, const N: usize>() -> [u8; N] {
    assert!(R >= 4 && R <= 6, "column height must be 4..=6");
    assert!(N == 1 << R, "column table must cover every R-bit mask");

    let mut out = [0; N];
    let mut column = 0;
    while column < N {
        let bits = column as u32;
        let height = 32 - bits.leading_zeros();
        let holes = height - bits.count_ones();
        out[column] = if holes > 1 {
            0x80
        } else {
            let hole_code = if holes == 0 {
                0
            } else {
                (((1u32 << height) - 1) ^ bits).trailing_zeros() + 1
            };

            assert!(
                height < 8 && hole_code < 8,
                "column fields must fit three bits"
            );
            assert!(
                hole_code == 0 || hole_code < height,
                "hole must be below the top cell"
            );
            height as u8 | ((hole_code as u8) << 3)
        };
        column += 1;
    }

    out
}

const COLUMNS_4: [u8; 16] = make_columns::<4, 16>();
const COLUMNS_5: [u8; 32] = make_columns::<5, 32>();
const COLUMNS_6: [u8; 64] = make_columns::<6, 64>();

/// Returns the byte descriptions for the configured column height.
#[inline_conditioned(always)]
const fn columns<const R: usize>() -> &'static [u8] {
    match R {
        4 => &COLUMNS_4,
        5 => &COLUMNS_5,
        6 => &COLUMNS_6,
        _ => &[],
    }
}

#[derive(Clone, Copy)]
struct Prefix {
    parity: u8,
    min: usize,
    max: usize,
}

impl Prefix {
    /// Creates an empty column-prefix state.
    #[inline_conditioned]
    const fn empty() -> Self {
        Self {
            parity: 0,
            min: 32,
            max: 0,
        }
    }

    /// Extends the prefix with a solid column height.
    #[inline_conditioned]
    const fn add(self, h: usize) -> Self {
        Self {
            parity: self.parity ^ (h as u8 & 1),
            min: if h < self.min { h } else { self.min },
            max: if h > self.max { h } else { self.max },
        }
    }
}

/// Counts bounded height words with the requested occupancy parity.
#[inline_conditioned(always)]
const fn words(n: usize, lo: usize, hi: usize, parity: u8) -> i64 {
    debug_assert!(n <= 10, "suffix length exceeds the ten-column domain");
    debug_assert!(parity <= 1, "parity must be a bit");

    if lo > hi {
        return (n == 0 && parity == 0) as i64;
    }

    let size = hi - lo + 1;
    debug_assert!(
        size < POWERS.len(),
        "height alphabet exceeds the power table"
    );

    // Odd-sized alphabets and zero-length words differ by one between parities.
    let extra = ((size & 1) | (n == 0) as usize) as i64;
    let imbalance_power = extra - 2 * (extra & (lo & n & 1) as i64);
    (POWERS[size][n] as i64 + imbalance_power * (1 - 2 * (parity != 0) as i64)) / 2
}

/// Counts solid-column completions that satisfy the row and parity restrictions.
#[inline_conditioned(always)]
const fn solid_count<const R: usize>(p: Prefix, n: usize) -> u64 {
    let count = words(n, 0, R, p.parity)
        - if p.min == 0 {
            0
        } else {
            words(n, 1, R, p.parity)
        };
    debug_assert!(count >= 0, "solid completion count must be nonnegative");
    count as u64
}

/// Counts completions with one fixed hole and the required row coverage.
#[inline_conditioned(always)]
const fn one_count<const R: usize>(p: Prefix, n: usize, y: usize) -> u64 {
    let limit = if y == 0 { 1 } else { 0 };
    let need_min = p.min > limit;
    let need_max = p.max <= y;
    let mut count = words(n, 0, R, p.parity);

    if need_min {
        count -= words(n, limit + 1, R, p.parity);
    }
    if need_max {
        count -= words(n, 0, y, p.parity);
    }
    if need_min && need_max {
        count += words(n, limit + 1, y, p.parity);
    }

    assert!(count >= 0);
    count as u64
}

// Isolated cap columns and bounded lower columns form an inductively impossible
// family. Count independent cap markers with a second-order weighted recurrence.
#[derive(Clone, Copy)]
struct Bad {
    prev: bool,
    cap: bool,
    parity: u8,
    zero: bool,
    possible: bool,
}

impl Bad {
    /// Creates an empty column-prefix state.
    #[inline_conditioned]
    const fn empty() -> Self {
        Self {
            prev: false,
            cap: false,
            parity: 0,
            zero: false,
            possible: true,
        }
    }

    /// Marks the hole column in the impossible-family prefix.
    #[inline_conditioned]
    const fn hole(mut self) -> Self {
        if self.prev {
            self.possible = false;
        }
        self.prev = true;
        self
    }

    /// Extends the prefix with a solid column height.
    #[inline_conditioned]
    const fn add<const R: usize>(mut self, h: usize, max_low: usize) -> Self {
        if !self.possible {
            return self;
        }

        if h == R {
            if self.prev {
                self.possible = false;
            }
            self.prev = true;
            self.cap = true;
        } else if h <= max_low {
            self.prev = false;
            self.zero |= h == 0;
        } else {
            self.possible = false;
        }

        self.parity ^= h as u8 & 1;
        self
    }
}

/// Signed independent-marker counts for low weights -1..=5, cap weights -1/+1,
/// and suffix lengths 0..=10. All 154 entries fit i32; the table uses 616 bytes.
const INDEPENDENT: [[[i32; 11]; 2]; 7] = {
    let mut out = [[[0; 11]; 2]; 7];
    let mut low_index = 0;
    while low_index < 7 {
        let low = low_index as i64 - 1;
        let mut cap_index = 0;
        while cap_index < 2 {
            let cap = 2 * cap_index as i64 - 1;
            out[low_index][cap_index][0] = 1;
            out[low_index][cap_index][1] = (low + cap) as i32;
            let mut n = 2;
            while n <= 10 {
                let next = low * out[low_index][cap_index][n - 1] as i64
                    + low * cap * out[low_index][cap_index][n - 2] as i64;
                assert!(next >= i32::MIN as i64 && next <= i32::MAX as i64);
                out[low_index][cap_index][n] = next as i32;
                n += 1;
            }
            cap_index += 1;
        }
        low_index += 1;
    }

    out
};

/// Counts weighted column words with no adjacent cap markers.
#[inline_conditioned]
const fn independent(n: usize, low: i64, cap: i64) -> i64 {
    debug_assert!(n <= 10, "suffix length exceeds the ten-column domain");
    debug_assert!(
        low >= -1 && low <= 5,
        "lower-symbol weight is outside -1..=5"
    );
    debug_assert!(
        cap == -1 || cap == 0 || cap == 1,
        "cap weight must be -1, 0 or 1"
    );

    if cap == 0 {
        let power = POWERS[low.unsigned_abs() as usize][n] as i64;
        if low < 0 && n & 1 != 0 { -power } else { power }
    } else {
        INDEPENDENT[(low + 1) as usize][(cap > 0) as usize][n] as i64
    }
}

/// Counts marker words around the distinguished hole column.
#[inline_conditioned]
const fn weighted(position: usize, prev: bool, hole_col: usize, low: i64, cap: i64) -> i64 {
    debug_assert!(position <= 10, "suffix position exceeds ten columns");
    debug_assert!(hole_col < 10, "hole column exceeds the board width");

    let n = 10 - position;
    if cap == 0 {
        return if hole_col < position {
            independent(n, low, 0)
        } else if prev && hole_col == position {
            0
        } else {
            independent(n - 1, low, 0)
        };
    }

    if hole_col < position {
        if !prev || n == 0 {
            independent(n, low, cap)
        } else {
            low * independent(n - 1, low, cap)
        }
    } else {
        let left = hole_col - position;
        let right = 9 - hole_col;
        let a = if left == 0 {
            if prev { 0 } else { 1 }
        } else if !prev {
            low * independent(left - 1, low, cap)
        } else if left == 1 {
            low
        } else {
            low * low * independent(left - 2, low, cap)
        };

        let b = if right == 0 {
            1
        } else {
            low * independent(right - 1, low, cap)
        };
        a * b
    }
}

/// Counts suffixes belonging to the inductively impossible family.
#[inline_conditioned]
const fn bad_count<const R: usize>(
    height: usize,
    y: usize,
    hole_col: usize,
    position: usize,
    state: Bad,
) -> u64 {
    if height != R || y + 4 < R || y == 0 || !state.possible {
        return 0;
    }

    let mut values = [0i64; 2];
    let mut sign = 0;
    while sign < 2 {
        let cap = if sign == 0 || R.is_multiple_of(2) {
            1
        } else {
            -1
        };

        let mut zero_free = 0;
        while zero_free < 2 {
            let low = if sign == 0 {
                (y + 1 - zero_free) as i64
            } else {
                (if y.is_multiple_of(2) { 1 } else { 0 }) - zero_free as i64
            };

            let mut n = weighted(position, state.prev, hole_col, low, cap);
            if !state.cap {
                n -= weighted(position, state.prev, hole_col, low, 0);
            }

            if zero_free == 0 {
                values[sign] += n;
            } else if !state.zero {
                values[sign] -= n;
            }
            zero_free += 1;
        }
        sign += 1;
    }

    let n = values[0]
        + if ((R - 1) as u8 ^ state.parity) & 1 == 0 {
            values[1]
        } else {
            -values[1]
        };
    assert!(n >= 0 && n % 2 == 0);
    (n / 2) as u64
}

/// Counts valid boards with fixed hole height, row and column.
#[inline_conditioned]
const fn bucket<const R: usize>(height: usize, y: usize, hole_col: usize) -> u64 {
    let p = Prefix {
        parity: ((height - 1) & 1) as u8,
        ..Prefix::empty()
    };

    let total = one_count::<R>(p, 9, y);
    let excluded = bad_count::<R>(height, y, hole_col, 0, Bad::empty());
    assert!(
        excluded <= total,
        "excluded bucket count exceeds its carrier"
    );
    total - excluded
}

// Differentiating the independent-marker recurrence counts a distinguished
// cap column, summing all ten possible hole positions without a position loop.
/// Counts weighted words with one distinguished cap marker.
#[inline_conditioned]
const fn marked(n: usize, low: i64, cap: i64) -> i64 {
    if n == 0 {
        return 0;
    }

    let (mut a, mut b, mut da, mut db) = (1, low + cap, 0, 1);
    let mut i = 2;
    while i <= n {
        let next = low * b + low * cap * a;
        let derivative = low * db + low * a + low * cap * da;
        a = b;
        b = next;
        da = db;
        db = derivative;
        i += 1;
    }

    db
}

/// Counts valid boards for one hole height and row across every column.
#[inline_conditioned]
const fn group<const R: usize>(height: usize, y: usize) -> u64 {
    let p = Prefix {
        parity: ((height - 1) & 1) as u8,
        ..Prefix::empty()
    };

    let base = 10 * one_count::<R>(p, 9, y);
    if height != R || y + 4 < R || y == 0 {
        return base;
    }

    let mut value = [0i64; 2];
    let mut sign = 0;
    while sign < 2 {
        let cap = if sign == 0 || R.is_multiple_of(2) {
            1
        } else {
            -1
        };

        let mut positive = 0;
        while positive < 2 {
            let low = if sign == 0 {
                (y + 1 - positive) as i64
            } else {
                (if y.is_multiple_of(2) { 1 } else { 0 }) - positive as i64
            };

            let n = marked(10, low, cap) - 10 * low.pow(9);
            value[sign] += if positive == 0 { n } else { -n };
            positive += 1;
        }
        sign += 1;
    }

    let removed = value[0]
        + if (R - 1).is_multiple_of(2) {
            value[1]
        } else {
            -value[1]
        };
    assert!(removed >= 0 && removed % 2 == 0);

    let excluded = (removed / 2) as u64;
    assert!(excluded <= base, "excluded group count exceeds its carrier");
    base - excluded
}

/// Builds cumulative hole-bucket offsets from the solid-sector count to the final ID sentinel.
#[inline_conditioned]
const fn make_metadata<const R: usize, const N: usize>() -> [u32; N] {
    assert!(R >= 4 && R <= 6, "metadata height must be 4..=6");
    assert!(
        N == 1 + 10 * R * (R - 1) / 2,
        "metadata table length is inconsistent"
    );

    let mut out = [0; N];
    let mut offset = solid_count::<R>(Prefix::empty(), 10);
    assert!(offset <= u32::MAX as u64);
    out[0] = offset as u32;
    let mut index = 1;
    let mut height = 2;
    while height <= R {
        let mut y = 0;
        while y + 1 < height {
            let mut hole_col = 0;
            while hole_col < 10 {
                offset = offset
                    .checked_add(bucket::<R>(height, y, hole_col))
                    .expect("metadata count overflow");
                assert!(offset <= u32::MAX as u64);
                out[index] = offset as u32;
                assert!(
                    out[index - 1] <= out[index],
                    "metadata offsets must be nondecreasing"
                );
                index += 1;
                hole_col += 1;
            }
            y += 1;
        }
        height += 1;
    }

    assert!(index == N && offset == count::<R>());
    out
}

const METADATA_4: [u32; 61] = make_metadata::<4, 61>();
const METADATA_5: [u32; 101] = make_metadata::<5, 101>();
const METADATA_6: [u32; 151] = make_metadata::<6, 151>();

/// Returns this height cap's cumulative metadata offsets.
#[inline_conditioned(always)]
const fn metadata<const R: usize>() -> &'static [u32] {
    match R {
        4 => &METADATA_4,
        5 => &METADATA_5,
        6 => &METADATA_6,
        _ => &[],
    }
}

/// Counts IDs preceding the supplied hole metadata.
#[inline_conditioned]
const fn metadata_before<const R: usize>(height: usize, y: usize, hole_col: usize) -> u64 {
    debug_assert!(
        height >= 2 && height <= R,
        "hole-column height is outside the cap"
    );
    debug_assert!(y < height - 1, "hole must be below the top cell");
    debug_assert!(hole_col < 10, "hole column exceeds the board width");

    let group = (height - 1) * (height - 2) / 2 + y;
    metadata::<R>()[10 * group + hole_col] as u64
}

/// Counts all boards in the refined one-hole arithmetic language.
#[inline_conditioned]
const fn count<const R: usize>() -> u64 {
    assert!(R >= 4 && R <= 6);

    let mut total = solid_count::<R>(Prefix::empty(), 10);
    let mut h = 2;
    while h <= R {
        let mut y = 0;
        while y + 1 < h {
            total += group::<R>(h, y);
            y += 1;
        }
        h += 1;
    }

    total
}

/// Ranks solid column heights using the parity-adjusted radix representation.
#[inline_conditioned]
fn suffix_rank<const R: usize>(heights: &[usize; 10], start: usize, skip: usize) -> u64 {
    debug_assert!(
        start <= 10 && skip <= 10,
        "invalid suffix bounds or skip sentinel"
    );
    debug_assert!(
        heights.iter().all(|&height| height <= R),
        "column exceeds the cap"
    );

    let mut raw = 0;
    for (x, &h) in heights.iter().enumerate().skip(start) {
        if x != skip {
            raw = raw * (R + 1) as u64 + h as u64;
        }
    }
    raw / 2
}

/// Restores solid column heights from their parity-adjusted radix ID.
#[inline_conditioned]
fn suffix_select<const R: usize>(
    id: u64,
    cols: &mut [u32; 10],
    start: usize,
    skip: usize,
    parity: u8,
) {
    debug_assert!(
        start <= 10 && skip <= 10,
        "invalid suffix bounds or skip sentinel"
    );
    debug_assert!(parity <= 1, "parity must be a bit");

    let mut raw = 2 * id
        + if R.is_multiple_of(2) {
            u64::from(parity)
        } else {
            0
        };
    let mut actual = 0;
    let mut last = 10;
    for x in (start..10).rev() {
        if x == skip {
            continue;
        }

        if last == 10 {
            last = x;
        }

        let h = (raw % (R + 1) as u64) as usize;
        raw /= (R + 1) as u64;
        actual ^= h as u8 & 1;
        cols[x] = (1 << h) - 1;
    }

    assert_eq!(raw, 0);

    if actual != parity {
        assert!(!R.is_multiple_of(2) && last != 10);

        let h = (32 - cols[last].leading_zeros()) as usize;
        cols[last] = (1 << (h + 1)) - 1;
    }
}

/// Arithmetic kernel. Ten columns and at most one hole; caps four through six.
/// Its language matches the recursively refined one-hole carrier, with a new dense ID order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArithmeticOneHole<const R: usize>;

impl<const R: usize> ArithmeticOneHole<R> {
    /// Number of boards in the refined arithmetic carrier.
    pub const BOARD_COUNT: u64 = count::<R>();

    /// Creates an arithmetic codec for a height cap of four through six.
    ///
    /// # Panics
    /// Panics if the height cap is outside the supported range.
    #[inline_conditioned]
    pub const fn new() -> Self {
        assert!(R >= 4 && R <= 6);
        Self
    }

    /// Ranks a packed column-major board, returning `None` outside the refined language.
    #[inline_conditioned]
    pub fn rank(&self, code: u64) -> Option<u64> {
        if code >> (R * 10) != 0 {
            return None;
        }

        let mut heights = [0usize; 10];
        let mut hole = None;
        let mut x = 0;
        while x < 10 {
            let col = ((code >> ((9 - x) * R)) & ((1 << R) - 1)) as u32;
            let description = columns::<R>()[col as usize];
            if description & 0x80 != 0 {
                return None;
            }

            let h = usize::from(description & 7);
            heights[x] = h;
            let hole_code = description >> 3;
            if hole_code != 0 {
                if hole.is_some() {
                    return None;
                }
                hole = Some((h, usize::from(hole_code - 1), x));
            }
            x += 1;
        }

        if let Some((height, y, hole_col)) = hole {
            let mut validity = Prefix {
                parity: ((height - 1) & 1) as u8,
                ..Prefix::empty()
            };
            let mut family = Bad::empty();
            for (x, &column_height) in heights.iter().enumerate() {
                if x == hole_col {
                    family = family.hole();
                } else {
                    validity = validity.add(column_height);
                    family = family.add::<R>(column_height, y);
                }
            }

            let limit = usize::from(y == 0);
            if validity.parity != 0 || validity.min > limit || validity.max <= y {
                return None;
            }

            if height == R && y + 4 >= R && family.possible && family.cap && family.zero {
                return None;
            }

            let mut id = metadata_before::<R>(height, y, hole_col);
            let mut p = Prefix {
                parity: ((height - 1) & 1) as u8,
                ..Prefix::empty()
            };
            let mut bad = Bad::empty();
            for x in 0..10 {
                if p.min <= if y == 0 { 1 } else { 0 }
                    && p.max > y
                    && (!bad.possible || height != R || y + 4 < R || y == 0)
                {
                    return Some(id + suffix_rank::<R>(&heights, x, hole_col));
                }

                if x == hole_col {
                    bad = bad.hole();
                    continue;
                }

                let remaining = 9 - x - usize::from(hole_col > x);
                for h in 0..heights[x] {
                    let q = p.add(h);
                    let b = bad.add::<R>(h, y);
                    let total = one_count::<R>(q, remaining, y);
                    let excluded = bad_count::<R>(height, y, hole_col, x + 1, b);
                    debug_assert!(
                        excluded <= total,
                        "excluded suffix count exceeds its carrier"
                    );
                    id += total - excluded;
                }
                p = p.add(heights[x]);
                bad = bad.add::<R>(heights[x], y);
            }

            Some(id)
        } else {
            let mut raw = 0;
            let mut positive_prefix = 0;
            let mut first_zero = 10;
            let mut parity = 0;
            for (x, &h) in heights.iter().enumerate() {
                raw = raw * (R + 1) as u64 + h as u64;
                parity ^= h & 1;
                if first_zero == 10 {
                    if h == 0 {
                        first_zero = x;
                    } else {
                        positive_prefix = positive_prefix * R as u64 + h as u64 - 1;
                    }
                }
            }

            if first_zero == 10 || parity != 0 {
                return None;
            }

            let excluded = positive_prefix * (R as u64).pow((10 - first_zero) as u32);
            Some(raw / 2 - excluded.div_ceil(2))
        }
    }

    /// Restores the packed column-major board for a valid dense ID.
    #[inline_conditioned]
    pub fn select(&self, mut id: u64) -> Option<u64> {
        if id >= Self::BOARD_COUNT {
            return None;
        }

        let zeros = solid_count::<R>(Prefix::empty(), 10);
        let mut cols = [0u32; 10];
        if id < zeros {
            let mut p = Prefix::empty();
            for x in 0..10 {
                if p.min == 0 {
                    suffix_select::<R>(id, &mut cols, x, 10, p.parity);
                    id = 0;
                    break;
                }

                let mut found = false;
                for h in 0..=R {
                    let n = solid_count::<R>(p.add(h), 9 - x);
                    if id >= n {
                        id -= n;
                    } else {
                        cols[x] = (1 << h) - 1;
                        p = p.add(h);
                        found = true;
                        break;
                    }
                }

                assert!(found);
            }
        } else {
            let offsets = metadata::<R>();

            // The last equal offset skips empty buckets; BOARD_COUNT is the upper sentinel.
            let split = offsets.partition_point(|&offset| u64::from(offset) <= id);
            debug_assert!(
                split > 0 && split < offsets.len(),
                "ID must fall before the sentinel"
            );

            let bucket_index = split - 1;
            debug_assert!(
                u64::from(offsets[bucket_index]) <= id && id < u64::from(offsets[bucket_index + 1]),
                "selected metadata bucket must contain the ID"
            );
            id -= u64::from(offsets[bucket_index]);
            let hole_col = bucket_index % 10;
            let mut y = bucket_index / 10;
            let mut height = 2;
            while y >= height - 1 {
                y -= height - 1;
                height += 1;
            }
            cols[hole_col] = ((1 << height) - 1) ^ (1 << y);
            let mut p = Prefix {
                parity: ((height - 1) & 1) as u8,
                ..Prefix::empty()
            };

            let mut bad = Bad::empty();
            for x in 0..10 {
                if p.min <= if y == 0 { 1 } else { 0 }
                    && p.max > y
                    && (!bad.possible || height != R || y + 4 < R || y == 0)
                {
                    suffix_select::<R>(id, &mut cols, x, hole_col, p.parity);
                    id = 0;
                    break;
                }

                if x == hole_col {
                    cols[x] = ((1 << height) - 1) ^ (1 << y);
                    bad = bad.hole();
                    continue;
                }

                let remaining = 9 - x - usize::from(hole_col > x);
                let mut found = false;
                for h in 0..=R {
                    let q = p.add(h);
                    let b = bad.add::<R>(h, y);
                    let total = one_count::<R>(q, remaining, y);
                    let excluded = bad_count::<R>(height, y, hole_col, x + 1, b);
                    debug_assert!(
                        excluded <= total,
                        "excluded suffix count exceeds its carrier"
                    );

                    let n = total - excluded;
                    if id >= n {
                        id -= n;
                    } else {
                        cols[x] = (1 << h) - 1;
                        p = q;
                        bad = b;
                        found = true;
                        break;
                    }
                }

                assert!(found);
            }
        }

        assert_eq!(id, 0);
        Some(cols.iter().fold(0, |v, &c| (v << R) | u64::from(c)))
    }
}

/// Returns whether the dimensions support the arithmetic one-hole specialization.
#[inline_conditioned]
pub(crate) const fn applies(cols: usize, rows: usize, holes: usize) -> bool {
    cols == 10 && holes == 1 && rows >= 4 && rows <= 6
}

/// Returns the exact size of the supported arithmetic carrier.
#[inline_conditioned]
pub(crate) const fn board_count(rows: usize) -> u64 {
    match rows {
        4 => ArithmeticOneHole::<4>::BOARD_COUNT,
        5 => ArithmeticOneHole::<5>::BOARD_COUNT,
        6 => ArithmeticOneHole::<6>::BOARD_COUNT,
        _ => 0,
    }
}

/// Ranks a packed column-major board, returning `None` outside the refined language.
#[inline_conditioned]
pub(crate) fn rank(rows: usize, code: u64) -> Option<u64> {
    match rows {
        4 => ArithmeticOneHole::<4>::new().rank(code),
        5 => ArithmeticOneHole::<5>::new().rank(code),
        6 => ArithmeticOneHole::<6>::new().rank(code),
        _ => None,
    }
}

/// Restores the packed column-major board for a valid dense ID.
#[inline_conditioned]
pub(crate) fn select(rows: usize, id: u64) -> Option<u64> {
    match rows {
        4 => ArithmeticOneHole::<4>::new().select(id),
        5 => ArithmeticOneHole::<5>::new().select(id),
        6 => ArithmeticOneHole::<6>::new().select(id),
        _ => None,
    }
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
    use tetris_game::{StandardTetris, TetrisGameConfig};

    /// Rejects metadata storage with the wrong number of bucket boundaries.
    #[test]
    #[should_panic(expected = "metadata table length is inconsistent")]
    fn metadata_requires_the_exact_boundary_count() {
        let _ = make_metadata::<4, 60>();
    }

    /// Rejects a future column cap before its fields can overlap the rejection bit.
    #[test]
    #[should_panic(expected = "column height must be 4..=6")]
    fn column_descriptions_require_a_supported_height() {
        let _ = make_columns::<7, 128>();
    }

    /// Rejects cap weights that would otherwise alias a supported table entry.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cap weight must be -1, 0 or 1")]
    fn recurrence_rejects_unsupported_cap_weights() {
        let _ = independent(2, 1, 2);
    }

    /// Catches invalid metadata before its arithmetic index can select another bucket.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "hole column exceeds the board width")]
    fn metadata_rejects_an_out_of_range_column() {
        let _ = metadata_before::<4>(2, 0, 10);
    }

    /// Checks selection at both boundaries of every metadata bucket.
    fn metadata_boundaries<const R: usize>() {
        let codec = ArithmeticOneHole::<R>::new();
        for interval in metadata::<R>().windows(2) {
            if interval[0] == interval[1] {
                continue;
            }

            for id in [u64::from(interval[0]), u64::from(interval[1]) - 1] {
                let code = codec.select(id).expect("metadata boundary ID");
                assert_eq!(codec.rank(code), Some(id));
            }
        }
    }

    /// Covers every group/column boundary, including the final dense ID.
    #[test]
    fn metadata_bucket_boundaries_roundtrip() {
        metadata_boundaries::<4>();
        metadata_boundaries::<5>();
        metadata_boundaries::<6>();
    }

    /// Reconstructs every possible packed column using an independent cell scan.
    fn check_column_descriptions<const R: usize>() {
        for column in 0usize..1 << R {
            let mut height = 0;
            for y in 0..R {
                if column & (1 << y) != 0 {
                    height = y + 1;
                }
            }

            let mut missing = 0;
            let mut hole_code = 0;
            for y in 0..height {
                if column & (1 << y) == 0 {
                    missing += 1;
                    hole_code = y + 1;
                }
            }

            let description = columns::<R>()[column];
            assert_eq!(description & 0x80 != 0, missing > 1);

            if missing <= 1 {
                assert_eq!(usize::from(description & 7), height);
                assert_eq!(usize::from(description >> 3), hole_code);
            }
        }
    }

    /// Covers all 112 column patterns, including multi-hole rejection markers.
    #[test]
    fn column_descriptions_match_cell_scans() {
        check_column_descriptions::<4>();
        check_column_descriptions::<5>();
        check_column_descriptions::<6>();
    }

    /// Enumerates signed marker words independently of the recurrence and its table.
    #[test]
    fn marker_counts_match_exhaustive_weighted_words() {
        for position in 0..=10 {
            let n = 10 - position;
            for hole_col in 0..10 {
                for prev in [false, true] {
                    for low in -1i64..=5 {
                        for cap in [-1i64, 0, 1] {
                            let mut expected = 0;
                            for markers in 0u32..1 << n {
                                if markers & (markers << 1) != 0 || (prev && markers & 1 != 0) {
                                    continue;
                                }

                                let has_hole = hole_col >= position;
                                if has_hole && markers & (1 << (hole_col - position)) == 0 {
                                    continue;
                                }

                                let caps = markers.count_ones();
                                expected +=
                                    low.pow(n as u32 - caps) * cap.pow(caps - u32::from(has_hole));
                            }

                            assert_eq!(
                                weighted(position, prev, hole_col, low, cap),
                                expected,
                                "position={position} hole={hole_col} prev={prev} low={low} cap={cap}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Checks every supported height interval and suffix length with a parity DP.
    #[test]
    fn word_counts_match_independent_parity_counting() {
        for lo in 0..=7 {
            for hi in 0..=6 {
                let mut expected = [1i64, 0];
                for n in 0..=10 {
                    for parity in 0..=1 {
                        assert_eq!(
                            words(n, lo, hi, parity),
                            expected[parity as usize],
                            "n={n} lo={lo} hi={hi} parity={parity}"
                        );
                    }

                    let mut next = [0; 2];
                    for height in lo..=hi {
                        next[height & 1] += expected[0];
                        next[(height & 1) ^ 1] += expected[1];
                    }
                    expected = next;
                }
            }
        }
    }

    /// Keeps interval sizing correct at the integer boundary and for empty alphabets.
    #[test]
    fn word_counts_handle_extreme_and_empty_intervals() {
        for n in 0..=10 {
            for parity in 0..=1 {
                assert_eq!(
                    words(n, usize::MAX, usize::MAX, parity),
                    i64::from(usize::from(parity) == (n & 1))
                );
                assert_eq!(
                    words(n, usize::MAX, usize::MAX - 1, parity),
                    i64::from(n == 0 && parity == 0)
                );
            }
        }
    }

    /// Checks fixed-seed IDs against independent admission and exact inverses.
    fn check<const R: usize>() {
        let codec = ArithmeticOneHole::<R>::new();
        let mut state = 0xb04dc0deu64;
        for id in [
            0,
            1,
            ArithmeticOneHole::<R>::BOARD_COUNT / 2,
            ArithmeticOneHole::<R>::BOARD_COUNT - 1,
        ]
        .into_iter()
        .chain((0..64).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % ArithmeticOneHole::<R>::BOARD_COUNT
        })) {
            let code = codec.select(id).expect("valid fixture in check");
            assert_eq!(codec.rank(code), Some(id));

            let rows = std::array::from_fn(|y| {
                (0..10).fold(0u16, |v, x| {
                    v | ((((code >> ((9 - x) * R + y)) & 1) as u16) << x)
                })
            });

            assert!(
                crate::board::tests::admitted::<StandardTetris, R, 1>(&rows),
                "R={R} id={id} code={code:x}"
            );
        }

        assert_eq!(codec.select(ArithmeticOneHole::<R>::BOARD_COUNT), None);
        assert_eq!(codec.rank(1u64 << (R * 10)), None);
        assert_eq!(std::mem::size_of_val(&codec), 0);
    }

    #[test]
    fn arithmetic_counts_and_rank_selection_match_the_refined_language() {
        assert_eq!(ArithmeticOneHole::<4>::BOARD_COUNT, 58_149_719);
        assert_eq!(ArithmeticOneHole::<5>::BOARD_COUNT, 459_948_915);
        assert_eq!(ArithmeticOneHole::<6>::BOARD_COUNT, 2_551_894_523);
        check::<4>();
        check::<5>();
        check::<6>();
    }

    #[test]
    fn arithmetic_rejects_structurally_invalid_boards() {
        let c = ArithmeticOneHole::<4>::new();
        assert_eq!(c.rank(1), None); // odd occupied-cell count
        assert_eq!(c.rank((1 << 40) - 1), None); // full rows
        assert_eq!(c.rank((9u64 << 36) | 9), None); // more than one hole
    }

    /// Enumerates suffixes to validate arithmetic completion counts.
    fn prefix_counts<const R: usize>() {
        let mut seed = 0xb04dc0deu64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for sample in 0usize..32 {
            let j = random() as usize % 10;
            let height = if sample.is_multiple_of(2) {
                R
            } else {
                2 + random() as usize % (R - 1)
            };

            let y = random() as usize % (height - 1);
            let mut cols = [0u32; 10];
            cols[j] = ((1 << height) - 1) ^ (1 << y);
            let mut p = Prefix {
                parity: ((height - 1) & 1) as u8,
                ..Prefix::empty()
            };

            let mut bad = Bad::empty();
            for (x, col) in cols.iter_mut().enumerate().take(7) {
                if x == j {
                    bad = bad.hole();
                    continue;
                }

                let h = if sample.is_multiple_of(2) {
                    random() as usize % (y + 1)
                } else {
                    random() as usize % (R + 1)
                };
                *col = (1 << h) - 1;
                p = p.add(h);
                bad = bad.add::<R>(h, y);
            }

            let n = 3 - usize::from(j >= 7);
            let actual = one_count::<R>(p, n, y) - bad_count::<R>(height, y, j, 7, bad);
            let mut expected = 0;
            for digits in 0..(R + 1).pow(n as u32) {
                let mut digits = digits;
                for (x, col) in cols.iter_mut().enumerate().skip(7) {
                    if x == j {
                        continue;
                    }

                    let h = digits % (R + 1);
                    digits /= R + 1;
                    *col = (1 << h) - 1;
                }

                let rows = std::array::from_fn(|y| {
                    cols.iter()
                        .enumerate()
                        .fold(0u16, |v, (x, c)| v | (((c >> y) & 1) as u16) << x)
                });

                let mut cap_mask = 0u16;
                let mut family = height == R && y + 4 >= R;
                let mut other = false;
                let mut zero = false;
                for (x, &c) in cols.iter().enumerate() {
                    let h = (32 - c.leading_zeros()) as usize;
                    if h == R {
                        cap_mask |= 1 << x;
                        other |= x != j;
                    } else {
                        family &= h <= y;
                        zero |= h == 0;
                    }
                }
                family &= other && zero && cap_mask & (cap_mask << 1) == 0;
                expected += u64::from(
                    crate::board::tests::admitted::<StandardTetris, R, 1>(&rows) && !family,
                );
            }

            assert_eq!(
                actual, expected,
                "R={R} sample={sample} j={j} h={height} y={y}"
            );
        }
    }

    #[test]
    fn arithmetic_suffix_counts_match_independent_engine_enumeration() {
        prefix_counts::<4>();
        prefix_counts::<5>();
        prefix_counts::<6>();
    }

    /// Compares optimized metadata and radix formulas with direct counting.
    fn arithmetic_identities<const R: usize>() {
        let codec = ArithmeticOneHole::<R>::new();
        let mut offset = solid_count::<R>(Prefix::empty(), 10);
        for height in 2..=R {
            for y in 0..height - 1 {
                let mut total = 0;
                for j in 0..10 {
                    assert_eq!(metadata_before::<R>(height, y, j), offset);

                    let n = bucket::<R>(height, y, j);
                    total += n;
                    offset += n;
                }

                assert_eq!(total, group::<R>(height, y));
            }
        }

        assert_eq!(offset, ArithmeticOneHole::<R>::BOARD_COUNT);

        for n in 0..=4 {
            for wanted in 0..=1u8 {
                let mut next_id = 0;
                for raw in 0..((R + 1) as u64).pow(n) {
                    let mut digits = raw;
                    let mut sum = 0;
                    for _ in 0..n {
                        sum ^= (digits % (R + 1) as u64) as u8 & 1;
                        digits /= (R + 1) as u64;
                    }

                    if sum != wanted {
                        continue;
                    }

                    assert_eq!(raw / 2, next_id);

                    let mut cols = [0; 10];
                    suffix_select::<R>(next_id, &mut cols, 10 - n as usize, 10, wanted);
                    let heights = cols.map(|c| (32 - c.leading_zeros()) as usize);
                    let actual = heights
                        .iter()
                        .skip(10 - n as usize)
                        .fold(0, |v, &h| v * (R + 1) as u64 + h as u64);
                    assert_eq!(actual, raw);
                    next_id += 1;
                }
            }
        }

        let mut seed = 0xb04dc0deu64;
        for _ in 0..1024 {
            let heights: [usize; 10] = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as usize % (R + 1)
            });

            let mut prefix = Prefix::empty();
            let mut expected = 0;
            for (x, &height) in heights.iter().enumerate() {
                for h in 0..height {
                    expected += solid_count::<R>(prefix.add(h), 9 - x);
                }
                prefix = prefix.add(height);
            }

            let code = heights.iter().fold(0u64, |v, &h| (v << R) | ((1 << h) - 1));
            if solid_count::<R>(prefix, 0) == 1 {
                assert_eq!(codec.rank(code), Some(expected));
                assert_eq!(codec.select(expected), Some(code));
            } else {
                assert_eq!(codec.rank(code), None);
            }
        }
    }

    #[test]
    fn optimized_metadata_and_radix_formulas_match_direct_counts() {
        arithmetic_identities::<4>();
        arithmetic_identities::<5>();
        arithmetic_identities::<6>();
    }

    #[test]
    fn a_legal_last_move_does_not_make_the_closed_impossible_family_reachable() {
        use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};
        let columns = [11, 0, 15, 0, 3, 3, 0, 0, 1, 0];
        let rows = std::array::from_fn(|y| {
            columns
                .iter()
                .enumerate()
                .fold(0u16, |v, (x, c)| v | (((c >> y) & 1) as u16) << x)
        });

        assert!(crate::board::tests::admitted::<StandardTetris, 4, 1>(&rows));

        let codec = ArithmeticOneHole::<4>::new();
        let pack = |c: [u32; 10]| c.iter().fold(0u64, |v, &c| (v << 4) | u64::from(c));
        assert_eq!(codec.rank(pack(columns)), None);

        let before = [11, 0, 15, 0, 0, 0, 0, 0, 1, 0];
        assert_eq!(codec.rank(pack(before)), None);

        let action = *TetrisPiecePlacement::<10>::ALL_PLACEMENTS
            .iter()
            .find(|p| p.piece == TetrisPiece::O && usize::from(p.orientation.column.inner()) == 4)
            .expect("valid fixture in a_legal_last_move_does_not_make_the_closed_impossible_family_reachable");
        let mut replay =
            TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::from_limbs(before);
        assert_eq!(replay.apply_piece_placement(action).lines_cleared, 0);
        assert_eq!(replay.as_limbs().to_owned(), columns);
    }

    /// Checks the impossible-family geometry against local engine footprints.
    fn local_engine_geometry<const R: usize>() {
        use tetris_game::{TetrisBoard, TetrisPiecePlacement};
        let mut checked = 0;
        for action in TetrisPiecePlacement::<10>::ALL_PLACEMENTS.iter().copied() {
            if usize::from(action.orientation.column.inner()) != 3 {
                continue;
            }

            let width = usize::from(action.piece.width(action.orientation.rotation));
            let mut shape =
                TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::new();
            shape.apply_piece_placement(action);
            let height = shape.height() as usize;
            let mask = shape.as_limbs().to_owned();
            for y in R - 4..R - 1 {
                for raw in 0..(y + 3).pow(width as u32) {
                    let mut raw = raw;
                    let mut cols = [0u32; 10];
                    let mut hole = 0;
                    let mut holes = 0;
                    let mut caps = 0u16;
                    for (x, col) in cols.iter_mut().enumerate().skip(3).take(width) {
                        let digit = raw % (y + 3);
                        raw /= y + 3;
                        *col = if digit <= y {
                            (1 << digit) - 1
                        } else if digit == y + 1 {
                            (1 << R) - 1
                        } else {
                            hole = x;
                            holes += 1;
                            ((1 << R) - 1) ^ (1 << y)
                        };

                        if digit > y {
                            caps |= 1 << x;
                        }
                    }

                    if holes > 1 || caps & (caps << 1) != 0 {
                        continue;
                    }
                    cols[0] = if holes == 0 {
                        ((1 << R) - 1) ^ (1 << y)
                    } else {
                        (1 << R) - 1
                    };
                    cols[9] = (1 << R) - 1;
                    let target = TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::from_limbs(cols);
                    for dy in 0..=R - height {
                        let cells = mask.map(|c| c << dy);
                        if !(3..3 + width).all(|x| cols[x] & cells[x] == cells[x]) {
                            continue;
                        }

                        let before = std::array::from_fn(|x| cols[x] & !cells[x]);
                        let mut replay = TetrisBoard::<
                            { StandardTetris::COLS },
                            { StandardTetris::ROWS },
                        >::from_limbs(before);
                        let result = replay.apply_piece_placement(action);
                        if result.lines_cleared != 0 || replay != target {
                            continue;
                        }

                        assert_eq!(
                            before[hole], cols[hole],
                            "hole fragment removed R={R} y={y} action={action:?}"
                        );

                        for (x, &column) in before.iter().enumerate().skip(3).take(width) {
                            if x == hole {
                                continue;
                            }

                            let h = (32 - column.leading_zeros()) as usize;
                            assert!(
                                h <= y || h == R,
                                "intermediate column created R={R} y={y} action={action:?}"
                            );
                            assert_eq!(column, (1 << h) - 1);
                        }
                        checked += 1;
                    }
                }
            }
        }

        assert!(checked > 0);
    }

    #[test]
    fn every_local_engine_footprint_preserves_the_impossible_family_geometry() {
        local_engine_geometry::<4>();
        local_engine_geometry::<5>();
        local_engine_geometry::<6>();
    }
}
