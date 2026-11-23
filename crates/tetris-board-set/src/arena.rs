//! Self-contained board records and their compact width/payload codec.
//!
//! `used` ends the logical record stream. The initialized allocation also covers
//! decoder lookahead and spare capacity; those bytes are never part of a key.
//!
//! Encoding first chooses byte widths, tries shorter uniform/exception headers,
//! then accepts half-byte payloads only if the entire record shrinks. Ties keep
//! the earlier representation. Decoding, comparison and record walking must all
//! interpret that same choice, including the zero-byte record when N is zero.
//!
//! For N <= 21, Header caches three-bit width lanes in one word. Wider boards
//! read ordinary width fields directly or retain a sparse exception descriptor.
//! The top descriptor bit selects half-byte rather than byte payload units.

use std::{cmp::Ordering, marker::PhantomData};
use tetris_game::constants::BackingColType as Limb;

/// Header selectors. Literal widths 0..=4 occupy their own selector bits;
/// the remaining selectors reserve a prefix before the width fields.
mod format {
    /// One bit per column, representing widths 0 or 1.
    pub(super) const BINARY: u8 = 5;
    /// Two bits per column, representing widths 0..=3.
    pub(super) const TWO_BIT: u8 = 6;
    /// Two bits per column biased by one, representing widths 1..=4.
    pub(super) const NONZERO: u8 = 7;
    /// One common nonzero byte width, encoded in a single header byte.
    pub(super) const UNIFORM: u8 = 15;
    /// A common byte width with at most one exceptional column.
    pub(super) const ONE_EXCEPTION: u8 = 31;
    /// Byte width one with exactly two exceptional columns.
    pub(super) const TWO_EXCEPTIONS: u8 = 47;
}

/// Expand eight binary width fields into three-bit lanes at compile time.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Each extracted code is masked to one bit before expansion."
)]
const BINARY_WIDTHS: [u32; 256] = {
    let mut table = [0; 256];
    let mut code = 0;
    while code < 256 {
        let mut column = 0;
        while column < 8 {
            table[code] |= (((code >> column) & 1) as u32) << (column * 3);
            column += 1;
        }
        code += 1;
    }
    table
};

#[allow(
    clippy::cast_possible_truncation,
    reason = "Each extracted code is masked to two bits before expansion."
)]
/// Expand four two-bit width fields into three-bit lanes at compile time.
const TWO_BIT_WIDTHS: [u16; 256] = {
    let mut table = [0; 256];
    let mut code = 0;
    while code < 256 {
        let mut column = 0;
        while column < 4 {
            table[code] |= (((code >> (column * 2)) & 3) as u16) << (column * 3);
            column += 1;
        }
        code += 1;
    }
    table
};

/// Decode triangular pair ranks for boards through sixteen columns.
///
/// Larger boards use integer square root once per sparse-header decode.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Ranks below 128 produce column indices below 17, each fitting in eight bits."
)]
const PAIR_POSITIONS: [u16; 128] = {
    let mut table = [0; 128];
    let mut rank = 0;
    let mut second = 1;
    while rank < 128 {
        while (second + 1) * second / 2 <= rank {
            second += 1;
        }
        let first = rank - second * (second - 1) / 2;
        table[rank] = first as u16 | ((second as u16) << 8);
        rank += 1;
    }
    table
};

/// Cache half-byte payload units above the 21 three-bit width lanes.
///
/// Wider records use a low-bit sparse descriptor or direct metadata reads instead.
const NIBBLE_PAYLOAD: u64 = 1 << 63;

/// Decoded width metadata and the encoded header length for one record.
struct Header {
    /// Expanded width lanes or a sparse descriptor, plus the payload-unit flag.
    widths: u64,
    /// Bytes occupied by metadata before the first payload unit.
    bytes: usize,
}

impl Header {
    /// Whether payload widths use four-bit units instead of bytes.
    const fn nibbles(&self) -> bool {
        self.widths & NIBBLE_PAYLOAD != 0
    }
}

/// Immutable, self-contained board records followed by one initialized decoder tail.
///
/// Columns omit leading zero bytes, or half-bytes when that shortens a record.
/// Literal widths and compact/sparse headers describe payload lengths; a spare
/// metadata bit selects half-byte units. Logical offsets survive vector relocation.
/// No per-record padding, shared-prefix dependency or ordering index is retained.
pub(super) struct Arena<const N: usize> {
    /// Fully initialized allocation; `used` tracks logical records separately.
    ///
    /// Length equals capacity after growth, so appends can reuse initialized bytes
    /// without per-record zeroing. The final record always has its decoder tail.
    pub(super) bytes: Vec<u8>,
    /// Logical record bytes; bounded-decoder tail padding follows.
    pub(super) used: usize,
    /// Carry the column type and width without storing another board.
    width: PhantomData<[Limb; N]>,
}

impl<const N: usize> Arena<N> {
    /// Construct an empty arena without allocating or initializing a decoder tail.
    pub(super) const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            used: 0,
            width: PhantomData,
        }
    }

    /// Maximum encoded width-header length for this column count.
    const fn max_header_bytes() -> usize {
        (N * 3).div_ceil(8)
    }

    /// Maximum record length, also used as the shared initialized decoder tail.
    const fn max_record_bytes() -> usize {
        Self::max_header_bytes() + N * size_of::<Limb>()
    }

    /// Bits needed to encode one column index, including zero for empty boards.
    const fn index_bits() -> usize {
        (usize::BITS - N.saturating_sub(1).leading_zeros()) as usize
    }

    /// Bits needed for a triangular rank identifying two distinct columns.
    const fn pair_bits() -> usize {
        let pairs = N * N.saturating_sub(1) / 2;
        (usize::BITS - pairs.saturating_sub(1).leading_zeros()) as usize
    }

    /// Read the width format from its ordinary or escaped header prefix.
    const fn format(byte: u8) -> u8 {
        let mode = byte & 7;
        if mode == format::NONZERO && byte & 8 != 0 {
            byte & 63
        } else {
            mode
        }
    }

    /// Encoded header length for a format at this compile-time board width.
    const fn header_bytes(mode: u8) -> usize {
        match mode {
            format::BINARY => (N + 3).div_ceil(8),
            format::TWO_BIT => (N * 2 + 3).div_ceil(8),
            format::NONZERO => (N * 2 + 4).div_ceil(8),
            format::UNIFORM => 1,
            format::ONE_EXCEPTION => (12 + Self::index_bits()).div_ceil(8),
            format::TWO_EXCEPTIONS => (10 + Self::pair_bits()).div_ceil(8),
            _ => Self::max_header_bytes(),
        }
    }

    /// Bit position available for payload units; zero means no half-byte flag.
    ///
    /// A position on a byte boundary has no spare padding bit and is not usable.
    const fn payload_flag_bit(mode: u8) -> usize {
        match mode {
            format::BINARY => N + 3,
            format::TWO_BIT => N * 2 + 3,
            format::NONZERO => N * 2 + 4,
            format::UNIFORM | format::ONE_EXCEPTION | format::TWO_EXCEPTIONS => 0,
            _ => N * 3,
        }
    }

    /// Decode a sparse header into its default width and exceptional positions.
    ///
    /// The caller expands the descriptor to three-bit lanes for widths through 21;
    /// wider boards retain the descriptor for direct column lookup.
    fn sparse_info(mode: u8, raw: u32) -> u64 {
        match mode {
            format::UNIFORM => {
                let width = (raw >> 6) & 3;
                u64::from(if width == 0 { 4 } else { width })
            }
            format::ONE_EXCEPTION => {
                let default = (raw >> 6) & 7;
                let first = (raw >> 9) & ((1 << Self::index_bits()) - 1);
                let width = (raw >> (9 + Self::index_bits())) & 7;
                u64::from(default) | (u64::from(first) << 3) | (u64::from(width) << 11)
            }
            format::TWO_EXCEPTIONS => {
                let rank = ((raw >> 6) & ((1 << Self::pair_bits()) - 1)) as usize;
                let (first, second) = if N <= 16 {
                    let pair = PAIR_POSITIONS[rank];
                    (usize::from(pair & 255), usize::from(pair >> 8))
                } else {
                    let second = (rank * 8 + 1).isqrt().div_ceil(2);
                    (rank - second * (second - 1) / 2, second)
                };
                debug_assert!(first < second && second < N);
                let code0 = (raw >> (6 + Self::pair_bits())) & 3;
                let code1 = (raw >> (8 + Self::pair_bits())) & 3;
                let width0 = if code0 == 0 { 0 } else { code0 + 1 };
                let width1 = if code1 == 0 { 0 } else { code1 + 1 };
                1 | ((first as u64) << 3)
                    | (u64::from(width0) << 11)
                    | ((second as u64) << 14)
                    | (u64::from(width1) << 22)
            }
            _ => 0,
        }
    }

    /// Expand a sparse descriptor into width lanes when they fit in one word.
    const fn sparse_widths(info: u64, mode: u8) -> u64 {
        if N > 21 {
            return info;
        }
        let mut widths = (((1u64 << (N * 3)) - 1) / 7) * (info & 7);
        if mode != format::UNIFORM {
            let shift = ((info >> 3) & 255) * 3;
            widths = (widths & !(7 << shift)) | (((info >> 11) & 7) << shift);
        }
        if mode == format::TWO_EXCEPTIONS {
            let shift = ((info >> 14) & 255) * 3;
            widths = (widths & !(7 << shift)) | (((info >> 22) & 7) << shift);
        }
        widths
    }

    /// Pack width fields into metadata without touching overlapping payload storage.
    ///
    /// Widths must fit `BITS` after subtracting `BIAS`; `PREFIX` reserves format bits.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Write the low byte of a split width field; any high bits are written to the next byte."
    )]
    fn pack_header<const BITS: usize, const BIAS: u8, const PREFIX: usize>(
        widths: &[u8; N],
        mode: u8,
    ) -> [u8; N] {
        // ceil(3*N/8) <= N. Keep metadata independent of overlapping payload
        // stores; only the selected header prefix is copied into the record.
        let mut bytes = [0; N];
        if N != 0 {
            bytes[0] = mode;
        }
        tetris_utils::repeat_idx_unroll!(@bounded N, I, {
            let bit = I * BITS + PREFIX;
            let value = u16::from(widths[I] - BIAS) << (bit % 8);
            bytes[bit / 8] |= value as u8;
            if bit % 8 + BITS > 8 {
                bytes[bit / 8 + 1] |= (value >> 8) as u8;
            }
        });
        bytes
    }

    /// Append one canonical record and return its stable logical byte offset.
    ///
    /// Choose a shorter half-byte or sparse form only when the complete record shrinks.
    /// The initialized tail permits overlapping word stores without exposing padding
    /// as key data. Existing records are never rewritten.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "u32 bit lengths are at most 32; bounded unrolling limits columns to 255 and pair ranks to 32639."
    )]
    pub(super) fn append(&mut self, key: &[Limb; N]) -> usize {
        let offset = self.used;
        // Select byte widths first; compact headers and half-byte payloads must
        // strictly shorten the full record before replacing this representation.
        let bits = key.map(|value| (32 - value.leading_zeros()) as u8);
        let mut widths = bits.map(|bits| bits.div_ceil(8));
        let mut mode = if N == 0 {
            0
        } else if widths.iter().all(|&width| width <= 1) {
            format::BINARY
        } else if widths.iter().all(|&width| width <= 3) {
            format::TWO_BIT
        } else if widths.iter().all(|&width| width != 0) {
            format::NONZERO
        } else {
            0
        };
        let mut headers = Self::header_bytes(mode);
        let mut payload = widths
            .iter()
            .map(|&width| usize::from(width))
            .sum::<usize>();
        let mut sparse = 0u32;
        if N != 0 {
            let first = widths[0];
            if headers > 1 && first > 0 && widths.iter().all(|&width| width == first) {
                mode = format::UNIFORM;
                headers = 1;
                sparse = u32::from(format::UNIFORM) | (u32::from(first & 3) << 6);
            }
            if Self::header_bytes(format::ONE_EXCEPTION) < headers {
                let second = if N > 1 { widths[1] } else { first };
                let third = if N > 2 { widths[2] } else { first };
                let default = if first == second || first == third {
                    first
                } else {
                    second
                };
                let mut exception = None;
                let mut valid = true;
                for (column, &width) in widths.iter().enumerate() {
                    if width != default {
                        if exception.is_some() {
                            valid = false;
                            break;
                        }
                        exception = Some((column, width));
                    }
                }
                if valid {
                    let (column, width) = exception.unwrap_or((0, default));
                    mode = format::ONE_EXCEPTION;
                    headers = Self::header_bytes(mode);
                    sparse = u32::from(format::ONE_EXCEPTION)
                        | (u32::from(default) << 6)
                        | ((column as u32) << 9)
                        | (u32::from(width) << (9 + Self::index_bits()));
                }
            }
            if Self::header_bytes(format::TWO_EXCEPTIONS) < headers && N >= 2 {
                let mut positions = [0; 2];
                let mut count = 0;
                for (column, &width) in widths.iter().enumerate() {
                    if width != 1 {
                        if count == 2 {
                            count = 3;
                            break;
                        }
                        positions[count] = column;
                        count += 1;
                    }
                }
                if count == 2 {
                    let [first, second] = positions;
                    let rank = second * (second - 1) / 2 + first;
                    let code0 = widths[first].saturating_sub(1);
                    let code1 = widths[second].saturating_sub(1);
                    mode = format::TWO_EXCEPTIONS;
                    headers = Self::header_bytes(mode);
                    sparse = u32::from(format::TWO_EXCEPTIONS)
                        | ((rank as u32) << 6)
                        | (u32::from(code0) << (6 + Self::pair_bits()))
                        | (u32::from(code1) << (8 + Self::pair_bits()));
                }
            }
        }
        let mut nibbles = false;
        if N != 0 {
            let half_widths = bits.map(|bits| bits.div_ceil(4));
            let half_mode = if half_widths.iter().all(|&width| width <= 1)
                && !Self::payload_flag_bit(format::BINARY).is_multiple_of(8)
            {
                Some(format::BINARY)
            } else if half_widths.iter().all(|&width| width <= 3) {
                Some(format::TWO_BIT)
            } else if half_widths.iter().all(|&width| (1..=4).contains(&width))
                && !Self::payload_flag_bit(format::NONZERO).is_multiple_of(8)
            {
                Some(format::NONZERO)
            } else if half_widths[0] <= 4
                && half_widths.iter().all(|&width| width <= 7)
                && !Self::payload_flag_bit(0).is_multiple_of(8)
            {
                Some(0)
            } else {
                None
            };
            if let Some(half_mode) = half_mode {
                let half_headers = Self::header_bytes(half_mode);
                let half_payload = half_widths
                    .iter()
                    .map(|&width| usize::from(width))
                    .sum::<usize>()
                    .div_ceil(2);
                // Never trade extra bytes for the packed form. A unit flag
                // occupies existing metadata padding, without lengthening it.
                if half_headers + half_payload < headers + payload {
                    widths = half_widths;
                    mode = half_mode;
                    headers = half_headers;
                    payload = half_payload;
                    nibbles = true;
                }
            }
        }
        debug_assert!(headers <= Self::max_header_bytes());
        let mut header = match mode {
            format::UNIFORM | format::ONE_EXCEPTION | format::TWO_EXCEPTIONS => {
                let mut bytes = [0; N];
                bytes[..headers].copy_from_slice(&sparse.to_le_bytes()[..headers]);
                bytes
            }
            format::BINARY => Self::pack_header::<1, 0, 3>(&widths, mode),
            format::TWO_BIT => Self::pack_header::<2, 0, 3>(&widths, mode),
            format::NONZERO => Self::pack_header::<2, 1, 4>(&widths, mode),
            _ => Self::pack_header::<3, 0, 0>(&widths, 0),
        };
        if nibbles {
            let bit = Self::payload_flag_bit(mode);
            header[bit / 8] |= 1 << (bit % 8);
        }
        // Finish metadata selection before touching storage. Offsets and encoded
        // records stay independent of Vec relocation and unused capacity.
        let size = headers + payload;
        let needed = offset + size;
        let allocated = needed + Self::max_record_bytes();
        if self.bytes.capacity() < allocated {
            // Keep 1.5625% headroom in the global buffer. Reserve toward the
            // same absolute capacity despite retaining the initialized tail;
            // logical byte offsets stay valid if Vec reallocates.
            self.bytes.reserve_exact(
                (allocated + allocated / 64).next_multiple_of(64) - self.bytes.len(),
            );
            // Initialize the complete allocation once per growth. Keeping these
            // u8 elements alive makes later bounded views valid without zeroing
            // on every record or on clear/reuse.
            self.bytes.resize(self.bytes.capacity(), 0);
        }
        // Header/payload stores replace every logical record byte. Spare bytes
        // may be stale after clear or an overlapping store; their values are
        // immaterial, but they stay initialized for bounded decoder word reads.
        let record = &mut self.bytes[offset..allocated];
        record[..headers].copy_from_slice(&header[..headers]);
        if nibbles {
            let mut pos = headers * 2;
            tetris_utils::repeat_idx_unroll!(@bounded N, I, {
                // A word store overlaps only future packed values/tail bytes.
                // At an odd position, preserve the preceding low half-byte.
                let byte = pos / 2;
                let old = Limb::from_le_bytes(
                    record[byte..byte + 4]
                        .try_into()
                        .expect("bounded decoder word has the required fixed byte width"),
                );
                let keep = if pos.is_multiple_of(2) { 0 } else { 15 };
                let word = (old & keep) | (key[I] << ((pos % 2) * 4));
                record[byte..byte + 4].copy_from_slice(&word.to_le_bytes());
                pos += usize::from(widths[I]);
            });
            debug_assert_eq!(offset + pos.div_ceil(2), needed);
        } else {
            let mut pos = headers;
            tetris_utils::repeat_idx_unroll!(@bounded N, I, {
                let value = key[I];
                let width = usize::from(widths[I]);
                // Full-word stores may overlap only future payload or initialized
                // tail bytes. Subsequent columns overwrite that overlap; the final
                // tail is not part of the key. This avoids variable-length copies.
                record[pos..pos + size_of::<Limb>()].copy_from_slice(&value.to_le_bytes());
                pos += width;
            });
            debug_assert_eq!(offset + pos, needed);
        }
        self.used = needed;
        offset
    }

    /// Read metadata at a record boundary emitted by [`Self::append`].
    fn header(&self, offset: usize) -> Header {
        Self::decode_header(&self.bytes[offset..offset + Self::max_record_bytes()])
    }

    /// Decode metadata from an initialized maximum-record view.
    ///
    /// The view must start at a valid, privately encoded record boundary. The arena
    /// tail provides the bounded word loads even for the final short record.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Only the low byte selects a format and the low 32 bits contain every sparse header."
    )]
    fn decode_header(record: &[u8]) -> Header {
        if N == 0 {
            return Header {
                widths: 0,
                bytes: 0,
            };
        }
        // The shared initialized tail backs a maximum-record view even for a
        // short final key. N=1 fits a word; larger headers fit an eight-byte load.
        let raw = if N > 21 || Self::max_header_bytes() <= size_of::<u32>() {
            u64::from(u32::from_le_bytes(
                record[..4]
                    .try_into()
                    .expect("bounded decoder word has the required fixed byte width"),
            ))
        } else {
            u64::from_le_bytes(
                record[..8]
                    .try_into()
                    .expect("bounded decoder word has the required fixed byte width"),
            )
        };
        let mode = Self::format(raw as u8);
        let bytes = Self::header_bytes(mode);
        if matches!(
            mode,
            format::UNIFORM | format::ONE_EXCEPTION | format::TWO_EXCEPTIONS
        ) {
            return Header {
                widths: Self::sparse_widths(Self::sparse_info(mode, raw as u32), mode),
                bytes,
            };
        }
        let bit = Self::payload_flag_bit(mode);
        let nibbles = !bit.is_multiple_of(8)
            && if N <= 21 {
                raw & (1u64 << (bit % 64)) != 0
            } else {
                record[bit / 8] & (1 << (bit % 8)) != 0
            };
        let flag = if nibbles { NIBBLE_PAYLOAD } else { 0 };
        if N > 21 {
            return Header {
                widths: flag,
                bytes,
            };
        }
        let mut widths = 0;
        match mode {
            format::BINARY => {
                // Mask the final group: raw word reads can include following
                // payload bytes or poisoned initialized padding.
                let packed = (raw >> 3) & ((1u64 << N) - 1);
                tetris_utils::repeat_idx_unroll!(@bounded N, I, {
                    if I.is_multiple_of(8) {
                        widths |= u64::from(BINARY_WIDTHS[((packed >> (I % 64)) & 255) as usize]) << ((I * 3) % 64);
                    }
                });
            }
            format::TWO_BIT | format::NONZERO => {
                let packed =
                    (raw >> if mode == format::NONZERO { 4 } else { 3 }) & ((1u64 << (N * 2)) - 1);
                tetris_utils::repeat_idx_unroll!(@bounded N, I, {
                    if I.is_multiple_of(4) {
                        widths |= u64::from(
                            TWO_BIT_WIDTHS[((packed >> ((I * 2) % 64)) & 255) as usize],
                        ) << ((I * 3) % 64);
                    }
                });
                if mode == format::NONZERO {
                    // Each three-bit lane is <=3, so adding one cannot carry
                    // into the next lane. Mask excludes nonexistent columns.
                    widths += 0x1249_2492_4924_9249 & ((1u64 << (N * 3)) - 1);
                }
            }
            _ => widths = raw & ((1u64 << (N * 3)) - 1),
        }
        Header {
            widths: widths | flag,
            bytes,
        }
    }

    /// Read one column width in the payload units selected by `header`.
    ///
    /// `offset` must identify that header's record and `column` must be below `N`.
    fn column_width(&self, offset: usize, header: &Header, column: usize) -> usize {
        if N <= 21 {
            ((header.widths >> (column * 3)) & 7) as usize
        } else {
            let mode = Self::format(self.bytes[offset]);
            if mode == format::UNIFORM {
                return (header.widths & 7) as usize;
            }
            if mode == format::ONE_EXCEPTION || mode == format::TWO_EXCEPTIONS {
                if column == ((header.widths >> 3) & 255) as usize {
                    return ((header.widths >> 11) & 7) as usize;
                }
                if mode == format::TWO_EXCEPTIONS
                    && column == ((header.widths >> 14) & 255) as usize
                {
                    return ((header.widths >> 22) & 7) as usize;
                }
                return (header.widths & 7) as usize;
            }
            let (stride, prefix, bias) = match mode {
                format::BINARY => (1, 3, 0),
                format::TWO_BIT => (2, 3, 0),
                format::NONZERO => (2, 4, 1),
                _ => (3, 0, 0),
            };
            let bit = column * stride + prefix;
            let mut bits = u16::from(self.bytes[offset + bit / 8]);
            if bit % 8 + stride > 8 {
                bits |= u16::from(self.bytes[offset + bit / 8 + 1]) << 8;
            }
            usize::from((bits >> (bit % 8)) & ((1 << stride) - 1)) + bias
        }
    }

    /// Read a zero-extended byte payload and advance an absolute byte position.
    ///
    /// The private encoding guarantees a width through four and initialized bytes.
    fn value(&self, width: usize, pos: &mut usize) -> Limb {
        if width == 0 {
            return 0;
        }
        debug_assert!(width <= 4);
        // The mask makes a safe word load crossing a column/record boundary
        // equivalent to reading exactly `width` bytes. Only the arena tail
        // needs a short copy. All encodings are private and produced above.
        let word = if let Some(bytes) = self.bytes.get(*pos..*pos + 4) {
            Limb::from_le_bytes(
                bytes
                    .try_into()
                    .expect("bounded decoder word has the required fixed byte width"),
            )
        } else {
            let mut bytes = [0; 4];
            bytes[..width].copy_from_slice(&self.bytes[*pos..*pos + width]);
            Limb::from_le_bytes(bytes)
        };
        *pos += width;
        word & (u32::MAX >> ((4 - width) * 8))
    }

    /// Read a zero-extended half-byte payload and advance a nibble position.
    ///
    /// Widths are at most seven; the initialized record view must cover a word load
    /// at the byte containing the starting nibble.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Nibble widths are 0 through 7, so their masks fit the u32 column."
    )]
    fn nibble_value(bytes: &[u8], width: usize, pos: &mut usize) -> Limb {
        debug_assert!(width <= 7);
        let width = width.min(7);
        let byte = *pos / 2;
        let word = Limb::from_le_bytes(
            bytes[byte..byte + 4]
                .try_into()
                .expect("bounded decoder word has the required fixed byte width"),
        );
        let value = word >> ((*pos % 2) * 4);
        *pos += width;
        value & ((1u64 << (width * 4)) - 1) as Limb
    }

    /// Reconstruct every backing bit of the board at a valid record offset.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Decoded byte widths are 0 through 4, so the selected payload mask fits u32."
    )]
    pub(super) fn key(&self, offset: usize) -> [Limb; N] {
        // As in membership, the initialized decoder tail permits one bounded
        // record view. No per-column tail fallback or temporary short copy.
        let bytes = &self.bytes[offset..offset + Self::max_record_bytes()];
        let header = Self::decode_header(bytes);
        if header.nibbles() {
            let mut pos = header.bytes * 2;
            return std::array::from_fn(|column| {
                Self::nibble_value(bytes, self.column_width(offset, &header, column), &mut pos)
            });
        }
        let mut pos = header.bytes;
        std::array::from_fn(|column| {
            let width = self.column_width(offset, &header, column);
            debug_assert!(width <= size_of::<Limb>());
            let width = width.min(size_of::<Limb>());
            let word = Limb::from_le_bytes(
                bytes[pos..pos + size_of::<Limb>()]
                    .try_into()
                    .expect("bounded decoder word has the required fixed byte width"),
            );
            pos += width;
            word & ((1u64 << (width * u8::BITS as usize)) - 1) as Limb
        })
    }

    /// Logical encoded length, excluding shared decoder padding.
    ///
    /// An empty-width board has a zero-byte record; arena walks count members too.
    pub(super) fn record_len(&self, offset: usize) -> usize {
        if N == 0 {
            return 0;
        }
        let header = self.header(offset);
        let payload = if N <= 9 || (N <= 15 && !header.nibbles()) {
            // Pair three-bit widths into six-bit lanes (each at most eight).
            // Multiplication sums those lanes. Byte widths have 4*N <= 60;
            // nibble widths have 7*N <= 63. Both bounds prevent lane carries.
            // Masks/factors depend only on N and constant-fold.
            // Mask first: odd N must not include a nonexistent partner column.
            let widths = header.widths & ((1u64 << (N * 3)) - 1);
            let lanes = N.div_ceil(2);
            let factor = ((1u64 << (lanes * 6)) - 1) / 63;
            let mask = factor * 7;
            let paired = (widths & mask) + ((widths >> 3) & mask);
            ((paired.wrapping_mul(factor) >> (lanes.saturating_sub(1) * 6)) & 63) as usize
        } else {
            (0..N)
                .map(|column| self.column_width(offset, &header, column))
                .sum()
        };
        header.bytes
            + if header.nibbles() {
                payload.div_ceil(2)
            } else {
                payload
            }
    }

    /// Compare a complete key with one encoded record, stopping at a mismatch.
    ///
    /// Callers must verify the full board after a directory fingerprint match.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Decoded byte widths are 0 through 4, so the selected comparison mask fits u32."
    )]
    pub(super) fn matches(&self, offset: usize, key: &[Limb; N]) -> bool {
        // One full-record view is valid even for the shortest final record:
        // append keeps max_record_bytes() of padding after logical data.
        let bytes = &self.bytes[offset..offset + Self::max_record_bytes()];
        let header = Self::decode_header(bytes);
        if header.nibbles() {
            let mut pos = header.bytes * 2;
            tetris_utils::repeat_idx_unroll!(@bounded N, I, {
                if Self::nibble_value(bytes, self.column_width(offset, &header, I), &mut pos) != key[I] {
                    return false;
                }
            });
            return true;
        }
        let mut pos = header.bytes;
        tetris_utils::repeat_idx_unroll!(@bounded N, I, {
            let width = self.column_width(offset, &header, I);
            debug_assert!(width <= size_of::<Limb>());
            // Keep the width bound visible to LLVM for safe word-load proofs.
            let width = width.min(size_of::<Limb>());
            let word = Limb::from_le_bytes(
                bytes[pos..pos + size_of::<Limb>()]
                    .try_into()
                    .expect("bounded decoder word has the required fixed byte width"),
            );
            let mask = ((1u64 << (width * u8::BITS as usize)) - 1) as Limb;
            pos += width;
            if word & mask != key[I] {
                return false;
            }
        });
        debug_assert!(pos <= bytes.len());
        true
    }

    /// Compare a record with a key in numeric lexicographic column order.
    pub(super) fn compare(&self, offset: usize, key: &[Limb; N]) -> Ordering {
        let header = self.header(offset);
        if header.nibbles() {
            let bytes = &self.bytes[offset..offset + Self::max_record_bytes()];
            let mut pos = header.bytes * 2;
            for (column, &value) in key.iter().enumerate() {
                let decoded =
                    Self::nibble_value(bytes, self.column_width(offset, &header, column), &mut pos);
                let order = decoded.cmp(&value);
                if !order.is_eq() {
                    return order;
                }
            }
            return Ordering::Equal;
        }
        let mut pos = offset + header.bytes;
        for (i, column) in key.iter().enumerate() {
            let order = self
                .value(self.column_width(offset, &header, i), &mut pos)
                .cmp(column);
            if !order.is_eq() {
                return order;
            }
        }
        Ordering::Equal
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "Reference bit streams mask individual bits; coordinates and widths are bounded to 256, and seeded values intentionally use low bits."
)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn reference_record<const N: usize>(key: &[Limb; N]) -> Vec<u8> {
        fn encode(key: &[Limb], widths: &[usize], mode: u8, nibbles: bool) -> Vec<u8> {
            let (stride, bias) = match mode {
                5 => (1, 0),
                6 => (2, 0),
                7 => (2, 1),
                _ => (3, 0),
            };
            let mut bits = Vec::new();
            if mode != 0 {
                bits.extend((0..if mode == 7 { 4 } else { 3 }).map(|bit| (mode >> bit) & 1));
            }
            for &width in widths {
                bits.extend((0..stride).map(|bit| (((width - bias) >> bit) & 1) as u8));
            }
            if !bits.len().is_multiple_of(8) {
                bits.push(u8::from(nibbles));
            } else {
                assert!(!nibbles);
            }
            let mut bytes = Vec::new();
            for byte in bits.chunks(8) {
                bytes.push(byte.iter().rev().fold(0, |value, &bit| (value << 1) | bit));
            }
            if nibbles {
                let mut payload = Vec::new();
                for (value, &width) in key.iter().zip(widths) {
                    payload.extend(
                        value
                            .to_le_bytes()
                            .into_iter()
                            .flat_map(|byte| [byte & 15, byte >> 4])
                            .take(width),
                    );
                }
                for pair in payload.chunks(2) {
                    bytes.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
                }
            } else {
                for (value, &width) in key.iter().zip(widths) {
                    bytes.extend_from_slice(&value.to_le_bytes()[..width]);
                }
            }
            bytes
        }
        let widths: Vec<_> = key
            .iter()
            .map(|value| {
                value
                    .to_le_bytes()
                    .into_iter()
                    .rposition(|byte| byte != 0)
                    .map_or(0, |last| last + 1)
            })
            .collect();
        let max = widths.iter().copied().max().unwrap_or(0);
        let min = widths.iter().copied().min().unwrap_or(0);
        let mode = if N == 0 {
            0
        } else if max < 2 {
            5
        } else if max < 4 {
            6
        } else if min > 0 {
            7
        } else {
            0
        };
        let mut best = encode(key, &widths, mode, false);
        let payload: Vec<_> = key
            .iter()
            .zip(&widths)
            .flat_map(|(value, &width)| value.to_le_bytes().into_iter().take(width))
            .collect();
        let index_bits = if N <= 1 {
            0
        } else {
            (N - 1).ilog2() as usize + 1
        };
        let pairs = N * N.saturating_sub(1) / 2;
        let pair_bits = if pairs <= 1 {
            0
        } else {
            (pairs - 1).ilog2() as usize + 1
        };
        let mut consider = |word: u32, bytes: usize| {
            let mut candidate = word.to_le_bytes()[..bytes].to_vec();
            candidate.extend_from_slice(&payload);
            if candidate.len() < best.len() {
                best = candidate;
            }
        };
        if N > 0 {
            if min == max && min > 0 {
                consider(15 | (((min & 3) as u32) << 6), 1);
            }
            let mut counts = std::collections::BTreeMap::new();
            for &width in &widths {
                *counts.entry(width).or_insert(0usize) += 1;
            }
            if let Some((&default, &count)) = counts.iter().max_by_key(|entry| entry.1)
                && count >= N.saturating_sub(1)
            {
                let column = widths
                    .iter()
                    .position(|&width| width != default)
                    .unwrap_or(0);
                let word = 31
                    | ((default as u32) << 6)
                    | ((column as u32) << 9)
                    | ((widths[column] as u32) << (9 + index_bits));
                consider(word, (12 + index_bits).div_ceil(8));
            }
            let exceptions: Vec<_> = widths
                .iter()
                .enumerate()
                .filter(|&(_, width)| *width != 1)
                .collect();
            if exceptions.len() == 2 {
                let (first, &width0) = exceptions[0];
                let (second, &width1) = exceptions[1];
                let rank = second * (second - 1) / 2 + first;
                let code0 = if width0 == 0 { 0 } else { width0 - 1 };
                let code1 = if width1 == 0 { 0 } else { width1 - 1 };
                consider(
                    47 | ((rank as u32) << 6)
                        | ((code0 as u32) << (6 + pair_bits))
                        | ((code1 as u32) << (8 + pair_bits)),
                    (10 + pair_bits).div_ceil(8),
                );
            }
        }
        if N != 0 {
            let half: Vec<_> = key
                .iter()
                .map(|value| {
                    value
                        .to_le_bytes()
                        .into_iter()
                        .flat_map(|byte| [byte & 15, byte >> 4])
                        .enumerate()
                        .fold(
                            0,
                            |width, (index, nibble)| if nibble == 0 { width } else { index + 1 },
                        )
                })
                .collect();
            let max = half.iter().copied().max().unwrap_or(0);
            let min = half.iter().copied().min().unwrap_or(0);
            let mode = if max < 2 && !(N + 3).is_multiple_of(8) {
                Some(5)
            } else if max < 4 {
                Some(6)
            } else if min > 0 && max < 5 && !(N * 2 + 4).is_multiple_of(8) {
                Some(7)
            } else if half[0] < 5 && max < 8 && !(N * 3).is_multiple_of(8) {
                Some(0)
            } else {
                None
            };
            if let Some(mode) = mode {
                let record = encode(key, &half, mode, true);
                if record.len() < best.len() {
                    best = record;
                }
            }
        }
        best
    }

    #[test]
    fn empty_storage_and_layout_are_const_compatible() {
        const HEADER_BYTES: usize = Arena::<10>::header_bytes(5);
        const RECORD_BYTES: usize = Arena::<10>::max_record_bytes();
        const EMPTY: Arena<10> = Arena::new();
        assert_eq!(HEADER_BYTES, 2);
        assert_eq!(RECORD_BYTES, 44);
        assert_eq!(EMPTY.bytes.capacity(), 0);
    }

    #[test]
    fn sparse_headers_preserve_zero_exceptions_and_wide_coordinates() {
        fn check<const N: usize>() {
            let mut arena = Arena::<N>::new();
            for value in [1, 255, 256, 65536, u32::MAX] {
                for variant in 0..5 {
                    let mut key = [value; N];
                    if N > 0 {
                        match variant {
                            1 => key[N - 1] = u32::MAX,
                            2 => {
                                key[0] = 0;
                                key[N - 1] = u32::MAX;
                            }
                            3 => {
                                key[0] = 256;
                                if N > 1 {
                                    key[1] = 256;
                                }
                            }
                            4 => {
                                key.fill(0);
                                key[N - 1] = u32::MAX;
                            }
                            _ => {}
                        }
                    }
                    let offset = arena.append(&key);
                    let expected = reference_record(&key);
                    assert_eq!(&arena.bytes[offset..arena.used], expected);
                    arena.bytes[arena.used..].fill(0xA5);
                    assert_eq!(arena.record_len(offset), expected.len());
                    assert_eq!(arena.key(offset), key);
                    assert!(arena.matches(offset, &key));
                    assert_eq!(arena.compare(offset, &key), Ordering::Equal);
                }
            }
        }
        check::<0>();
        check::<1>();
        check::<2>();
        check::<3>();
        check::<5>();
        check::<10>();
        check::<16>();
        check::<17>();
        check::<21>();
        check::<22>();
        check::<64>();
        check::<256>();
    }

    #[test]
    fn nibble_payloads_preserve_odd_positions_wide_fields_and_record_lengths() {
        fn check<const N: usize>() {
            let mut arena = Arena::<N>::new();
            for round in 0..64 {
                let key = std::array::from_fn(|column| {
                    if column == 0 {
                        4095
                    } else {
                        [0, 1, 15, 16, 255, 4095, 65535, 0xFFF_FFFF][(column + round) % 8]
                    }
                });
                let offset = arena.append(&key);
                let expected = reference_record(&key);
                assert_eq!(&arena.bytes[offset..arena.used], expected);
                arena.bytes[arena.used..].fill(0xFF);
                assert_eq!(arena.record_len(offset), expected.len());
                assert_eq!(arena.key(offset), key);
                assert!(arena.matches(offset, &key));
                if N > 0 {
                    let mut miss = key;
                    miss[round % N] ^= 1 << (round % 32);
                    assert!(!arena.matches(offset, &miss));
                    assert_eq!(arena.compare(offset, &miss), key.cmp(&miss));
                }
            }
            for round in 0..3 {
                let key = std::array::from_fn(|column| match round {
                    0 => 15,
                    1 => 4095,
                    _ => {
                        if column.is_multiple_of(2) {
                            4095
                        } else {
                            65535
                        }
                    }
                });
                let offset = arena.append(&key);
                let expected = reference_record(&key);
                assert_eq!(&arena.bytes[offset..arena.used], expected);
                arena.bytes[arena.used..].fill(0xFF);
                assert_eq!(arena.record_len(offset), expected.len());
                assert_eq!(arena.key(offset), key);
                assert!(arena.matches(offset, &key));
            }
            if N > 1 {
                let mut key = [0xFFF_FFFF; N];
                key[0] = 4095;
                let offset = arena.append(&key);
                assert_eq!(&arena.bytes[offset..arena.used], reference_record(&key));
                assert_eq!(arena.record_len(offset), arena.used - offset);
                assert_eq!(arena.key(offset), key);
            }
        }
        check::<0>();
        check::<1>();
        check::<2>();
        check::<5>();
        check::<8>();
        check::<9>();
        check::<10>();
        check::<15>();
        check::<21>();
        check::<22>();
        check::<64>();
        check::<256>();
    }

    #[test]
    fn packed_metadata_preserves_every_binary_and_two_bit_code() {
        fn check<const N: usize>(arena: &mut Arena<N>, key: [Limb; N]) {
            let offset = arena.append(&key);
            assert_eq!(&arena.bytes[offset..arena.used], reference_record(&key));
            let len = arena.used - offset;
            arena.bytes[arena.used..].fill(0xFF);
            assert_eq!(arena.record_len(offset), len);
            assert_eq!(arena.key(offset), key);
            assert!(arena.matches(offset, &key));
        }
        let mut binary = Arena::<8>::new();
        let mut zero_based = Arena::<4>::new();
        let mut one_based = Arena::<4>::new();
        for code in 0..256usize {
            check(
                &mut binary,
                std::array::from_fn(|column| if code & (1 << column) == 0 { 0 } else { 255 }),
            );
            check(
                &mut zero_based,
                std::array::from_fn(|column| [0, 1, 256, 65536][(code >> (column * 2)) & 3]),
            );
            check(
                &mut one_based,
                std::array::from_fn(|column| [1, 256, 65536, u32::MAX][(code >> (column * 2)) & 3]),
            );
        }
    }

    #[test]
    fn header_lengths_cover_width_combinations_and_lane_boundaries() {
        fn check<const N: usize>(exhaustive: bool) {
            let mut arena = Arena::<N>::new();
            let patterns = if exhaustive { 5usize.pow(N as u32) } else { 64 };
            for pattern in 0..patterns + 5 {
                let mut digits = pattern;
                let key = std::array::from_fn(|column| {
                    let width = if pattern >= patterns {
                        pattern - patterns
                    } else if exhaustive {
                        let width = digits % 5;
                        digits /= 5;
                        width
                    } else {
                        (pattern + column) % 5
                    };
                    [0, 1, 256, 65536, 0x100_0000][width]
                });
                let offset = arena.append(&key);
                let expected = arena.used - offset;
                assert_eq!(&arena.bytes[offset..arena.used], reference_record(&key));
                // Poison only padding bits of the selected metadata format.
                // They must not become extra columns or affect record lengths.
                let raw = if N == 0 { 0 } else { arena.bytes[offset] };
                let mode = if raw & 15 == 15 { raw & 63 } else { raw & 7 };
                let mut bits = match mode {
                    5 => N + 3,
                    6 => N * 2 + 3,
                    7 => N * 2 + 4,
                    15 => 8,
                    31 => 12 + Arena::<N>::index_bits(),
                    47 => 10 + Arena::<N>::pair_bits(),
                    _ => N * 3,
                };
                if mode < 8 && !bits.is_multiple_of(8) {
                    bits += 1;
                }
                if !bits.is_multiple_of(8) {
                    let keep = (1u8 << (bits % 8)) - 1;
                    let last = arena.header(offset).bytes - 1;
                    arena.bytes[offset + last] |= !keep;
                }
                assert_eq!(arena.record_len(offset), expected);
                assert_eq!(arena.key(offset), key);
            }
        }
        check::<0>(true);
        check::<1>(true);
        check::<2>(true);
        check::<3>(true);
        check::<4>(true);
        check::<5>(true);
        check::<6>(true);
        check::<7>(false);
        check::<9>(false);
        check::<10>(false);
        check::<11>(false);
        check::<15>(false);
        check::<16>(false);
        check::<21>(false);
        check::<22>(false);
        check::<64>(false);
        check::<256>(false);
    }

    #[test]
    fn record_walk_tracks_logical_ends_and_zero_byte_records() {
        fn check<const N: usize>() {
            let mut arena = Arena::<N>::new();
            let mut records = Vec::new();
            for round in 0..64 {
                let key = std::array::from_fn(|column| {
                    let value = ((round + 1) as u32)
                        .wrapping_mul(0x9E37_79B9)
                        .rotate_left(column as u32);
                    value & [0, 255, 65535, 0xFF_FFFF, u32::MAX][(round + column) % 5]
                });
                let start = arena.append(&key);
                records.push((key, start, arena.used));
            }
            arena.bytes[arena.used..].fill(0xA5);
            let mut cursor = 0;
            // Count records, including zero-byte N=0 records. Logical ends
            // come from the writer, independently of the reader's arithmetic.
            for (expected, start, end) in records {
                assert_eq!(cursor, start);
                let key = arena.key(cursor);
                let next = cursor + arena.record_len(cursor);
                assert_eq!(key, expected);
                assert_eq!(next, end);
                assert!(arena.matches(cursor, &expected));
                cursor = next;
            }
            assert_eq!(cursor, arena.used);
        }
        check::<0>();
        check::<1>();
        check::<10>();
        check::<11>();
        check::<22>();
        check::<64>();
        check::<256>();
    }

    #[test]
    fn append_reuses_poisoned_tail_without_changing_canonical_records() {
        fn check<const N: usize>() {
            let mut arena = Arena::<N>::new();
            let mut expected = Vec::new();
            let mut seed = 0x9B05_688C_2B3E_6C1Fu64;
            for round in 0..64 {
                let key = std::array::from_fn(|column| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let mask = [0, 255, 65535, 0xFF_FFFF, u32::MAX][(round + column) % 5];
                    seed as u32 & mask
                });
                let offset = expected.len();
                expected.extend_from_slice(&reference_record(&key));
                assert_eq!(arena.append(&key), offset);
                assert_eq!(&arena.bytes[..arena.used], expected);
                assert_eq!(arena.bytes.len(), arena.bytes.capacity());
                assert!(arena.bytes.len() >= arena.used + Arena::<N>::max_record_bytes());
                arena.bytes[arena.used..].fill(0xFF);
                assert_eq!(arena.key(offset), key);
                assert!(arena.matches(offset, &key));
            }
        }
        check::<0>();
        check::<1>();
        check::<10>();
        check::<11>();
        check::<22>();
        check::<256>();
    }

    #[test]
    fn arena_codec_has_canonical_bytes_and_ignores_following_records() {
        let key = [
            0,
            1,
            255,
            256,
            65535,
            65536,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX,
            0,
        ];
        let mut arena = Arena::<10>::new();
        let offset = arena.append(&key);
        assert_eq!(
            arena.bytes[..arena.used],
            [
                72, 164, 141, 4, 1, 255, 0, 1, 255, 255, 0, 0, 1, 255, 255, 255, 0, 0, 0, 1, 255,
                255, 255, 255
            ]
        );
        assert_eq!(arena.key(offset), key);
        assert!(arena.matches(offset, &key));
        let before = arena.bytes[..arena.used].to_vec();
        arena.append(&[0; 10]);
        assert_eq!(&arena.bytes[..before.len()], before);
        assert_eq!(arena.key(offset), key);
        let mut empty = Arena::<10>::new();
        let offset = empty.append(&[0; 10]);
        assert_eq!(empty.bytes[..empty.used], [5, 0]);
        assert_eq!(empty.key(offset), [0; 10]);
        empty.bytes[empty.used..].fill(0xA5);
        assert_eq!(empty.key(offset), [0; 10]);
    }

    #[test]
    fn arena_comparison_matches_full_width_numeric_order() {
        let values = [
            0,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            65535,
            65536,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX,
        ];
        let mut arena = Arena::<2>::new();
        for value in values {
            let key = [value, !value];
            let offset = arena.append(&key);
            for query in values.map(|x| [x, !x]) {
                assert_eq!(arena.compare(offset, &query), key.cmp(&query));
                assert_eq!(arena.matches(offset, &query), key == query);
            }
        }
    }

    #[test]
    fn ten_column_word_reads_ignore_poisoned_padding_for_every_backing_bit() {
        let mut arena = Arena::<10>::new();
        let mut records = Vec::new();
        for column in 0..10 {
            for bit in 0..32 {
                let mut key = [0; 10];
                key[column] = 1u32 << bit;
                let offset = arena.append(&key);
                arena.bytes[arena.used..].fill(0xA5);
                assert!(arena.matches(offset, &key));
                let mut absent = key;
                absent[(column + 1) % 10] ^= 1;
                assert!(!arena.matches(offset, &absent));
                records.push((offset, key));
            }
        }
        for (offset, key) in records {
            assert!(arena.matches(offset, &key));
            assert_eq!(arena.key(offset), key);
        }
    }

    #[test]
    fn generic_unrolled_matching_uses_derived_padding_at_every_header_boundary() {
        fn check<const N: usize>() {
            let mut arena = Arena::<N>::new();
            for value in [
                0,
                1,
                255,
                256,
                65535,
                65536,
                0xFF_FFFF,
                0x100_0000,
                u32::MAX,
            ] {
                let key = std::array::from_fn(|i| if i % 3 == 0 { value } else { 0 });
                let offset = arena.append(&key);
                assert_eq!(arena.bytes.len(), arena.bytes.capacity());
                assert!(arena.bytes.len() >= arena.used + Arena::<N>::max_record_bytes());
                arena.bytes[arena.used..].fill(0xA5);
                assert!(arena.matches(offset, &key));
                assert_eq!(arena.key(offset), key);
                if N > 0 {
                    let mut absent = key;
                    absent[N - 1] ^= 1;
                    assert!(!arena.matches(offset, &absent));
                }
            }
        }
        check::<0>();
        check::<1>();
        check::<4>();
        check::<10>();
        check::<11>();
        check::<21>();
        check::<22>();
        check::<33>();
        check::<64>();
        check::<65>();
        check::<128>();
        check::<200>();
        check::<255>();
        check::<256>();
    }
    #[test]
    fn exhaustive_three_column_width_boundaries_are_canonical() {
        let values = [
            0,
            1,
            15,
            16,
            255,
            256,
            65535,
            65536,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX,
        ];
        let mut arena = Arena::<3>::new();
        for a in values {
            for b in values {
                for c in values {
                    let key = [a, b, c];
                    let offset = arena.append(&key);
                    let expected = reference_record(&key);
                    assert_eq!(&arena.bytes[offset..arena.used], expected);
                    assert_eq!(arena.record_len(offset), expected.len());
                    arena.bytes[arena.used..].fill(0xA5);
                    assert_eq!(arena.key(offset), key);
                    assert!(arena.matches(offset, &key));
                    for column in 0..3 {
                        let mut query = key;
                        query[column] ^= 1 << 31;
                        assert!(!arena.matches(offset, &query));
                        assert_eq!(arena.compare(offset, &query), key.cmp(&query));
                    }
                }
            }
        }
    }

    #[test]
    fn byte_header_selection_preserves_exception_boundaries() {
        let mut single = [0xFFu32; 22];
        single[0] = 0;
        let mut double = single;
        double[21] = u32::MAX;
        let mut triple = double;
        triple[7] = 256;
        for (key, mode) in [
            (single, format::ONE_EXCEPTION),
            (double, format::TWO_EXCEPTIONS),
            (triple, 0),
        ] {
            let mut arena = Arena::<22>::new();
            let offset = arena.append(&key);
            assert_eq!(Arena::<22>::format(arena.bytes[offset]), mode);
            assert!(!arena.header(offset).nibbles());
            assert_eq!(&arena.bytes[offset..arena.used], reference_record(&key));
            arena.bytes[arena.used..].fill(0xA5);
            assert_eq!(arena.key(offset), key);
            assert!(arena.matches(offset, &key));
        }
    }
}
