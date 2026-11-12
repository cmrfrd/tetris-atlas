//! Rule-derived, reduced decision diagram. All construction runs in const evaluation.
#![allow(
    long_running_const_eval,
    reason = "Bounded codec graph construction may exceed the const evaluator progress threshold."
)]
use crate::board::{MediumNode, Nodes, SmallNode, WideNode, occupancy_modulus};
use crate::{CompiledConfig, Dimensions, Tables, arithmetic, private};
use tetris_game::{Rotation, TetrisPiece};

#[derive(Clone, Copy)]
struct Raw {
    var: u16,
    zero: u32,
    one: u32,
}

impl Raw {
    const EMPTY: Self = Self {
        var: u16::MAX,
        zero: 0,
        one: 0,
    };
}

#[derive(Clone, Copy)]
pub(crate) struct Node {
    pub zero: u32,
    pub one: u32,
    pub zero_count: u64,
    pub var: u16,
}

impl Node {
    const EMPTY: Self = Self {
        zero: 0,
        one: 0,
        zero_count: 0,
        var: u16::MAX,
    };
}

/// Returns the bounded symbolic node capacity for these dimensions.
pub const fn capacity(c: usize, r: usize, h: usize) -> usize {
    assert!(r >= 2, "codec requires at least two rows");
    assert!(
        c >= 1 && c <= 12 && r <= 31,
        "codec dimensions out of range"
    );
    let h = if h < c * (r - 1) { h } else { c * (r - 1) };
    let width = if c > 6 { 1 << (c - 6) } else { 1 };
    let complexity = if h >= 4 { 2 } else { 1 };
    let n = (c * r * (h + 1) * 32 * width * complexity).next_power_of_two();
    let minimum = if c > 4 && r >= 3 { 32768 } else { 8192 };
    let n = if n < minimum { minimum } else { n };
    assert!(
        n <= 1_048_576,
        "codec exceeds the symbolic workspace budget"
    );
    n
}

/// Returns the size of the symbolic unique-node hash table.
pub const fn slots(c: usize, r: usize, h: usize) -> usize {
    capacity(c, r, h) * 2
}

/// Mixes a node key for indexing a power-of-two hash table.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Hash indexing intentionally retains the low bits of the target pointer width."
)]
const fn mix(mut x: u64) -> usize {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    (x ^ (x >> 31)) as usize
}

#[derive(Clone, Copy)]
struct Frame {
    key: u64,
    var: u16,
    low: u32,
    high_a: u32,
    high_b: u32,
    returning_high: bool,
}

impl Frame {
    const EMPTY: Self = Self {
        key: 0,
        var: 0,
        low: 0,
        high_a: 0,
        high_b: 0,
        returning_high: false,
    };
}

struct Builder<const N: usize, const M: usize> {
    nodes: [Raw; N],
    len: usize,
    unique: [u32; M],
    keys: [u64; M],
    values: [u32; M],
    stamps: [u32; M],
    epoch: u32,
    cached: usize,
    map: [u32; N],
    cols: usize,
    rows: usize,
    frames: [Frame; 373],
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "The bounded workspace holds at most 2^20 nodes and board variables are below 12 * 31; node indices fit u32 and variables fit u16."
)]
impl<const N: usize, const M: usize> Builder<N, M> {
    /// Creates an empty symbolic builder for the requested dimensions.
    const fn new(c: usize, r: usize) -> Self {
        Self {
            nodes: [Raw::EMPTY; N],
            len: 1,
            unique: [0; M],
            keys: [0; M],
            values: [0; M],
            stamps: [0; M],
            epoch: 1,
            cached: 0,
            map: [0; N],
            cols: c,
            rows: r,
            frames: [Frame::EMPTY; 373],
        }
    }

    /// Clears memoized Boolean operations without discarding live nodes.
    const fn clear_cache(&mut self) {
        self.epoch += 1;
        self.cached = 0;
    }

    /// Interns a decision node, reducing identical branches.
    const fn node(&mut self, var: u16, zero: u32, one: u32) -> u32 {
        if zero == one {
            return zero;
        }
        let sign = zero & 1;
        let zero = zero ^ sign;
        let one = one ^ sign;
        let mut at =
            mix(((zero as u64) << 32 | one as u64) ^ (var as u64).wrapping_mul(0x9e3779b97f4a7c15))
                & (M - 1);
        while self.unique[at] != 0 {
            let i = self.unique[at] as usize;
            let old = self.nodes[i];
            if old.var == var && old.zero == zero && old.one == one {
                return ((i as u32) * 2) ^ sign;
            }
            at = (at + 1) & (M - 1);
        }
        assert!(self.len < N, "codec rule diagram exceeds node budget");
        let i = self.len;
        self.len += 1;
        self.nodes[i] = Raw { var, zero, one };
        self.unique[at] = i as u32;
        ((i as u32) * 2) ^ sign
    }

    /// Computes the intersection of two symbolic board languages.
    const fn and(&mut self, mut a: u32, mut b: u32) -> u32 {
        let mut depth = 0;
        let mut returning = false;
        let mut result = 0;
        loop {
            if returning {
                if depth == 0 {
                    return result;
                }
                let at = depth - 1;
                let frame = self.frames[at];
                if !frame.returning_high {
                    self.frames[at].low = result;
                    self.frames[at].returning_high = true;
                    a = frame.high_a;
                    b = frame.high_b;
                    returning = false;
                } else {
                    result = self.node(frame.var, frame.low, result);
                    if self.cached >= M * 3 / 4 {
                        self.clear_cache();
                    }
                    let mut at = mix(frame.key) & (M - 1);
                    while self.stamps[at] == self.epoch && self.keys[at] != frame.key {
                        at = (at + 1) & (M - 1);
                    }
                    if self.stamps[at] != self.epoch {
                        self.cached += 1;
                    }
                    self.stamps[at] = self.epoch;
                    self.keys[at] = frame.key;
                    self.values[at] = result;
                    depth -= 1;
                }
                continue;
            }
            if a == 0 || b == 0 {
                result = 0;
                returning = true;
                continue;
            }
            if a == 1 {
                result = b;
                returning = true;
                continue;
            }
            if b == 1 || a == b {
                result = a;
                returning = true;
                continue;
            }
            if a == b ^ 1 {
                result = 0;
                returning = true;
                continue;
            }
            if a > b {
                let t = a;
                a = b;
                b = t;
            }
            let key = (a as u64) << 32 | b as u64;
            let mut at = mix(key) & (M - 1);
            while self.stamps[at] == self.epoch {
                if self.keys[at] == key {
                    break;
                }
                at = (at + 1) & (M - 1);
            }
            if self.stamps[at] == self.epoch {
                result = self.values[at];
                returning = true;
                continue;
            }
            let an = self.nodes[(a / 2) as usize];
            let bn = self.nodes[(b / 2) as usize];
            let var = if an.var < bn.var { an.var } else { bn.var };
            let (az, ao) = if an.var == var {
                (an.zero ^ (a & 1), an.one ^ (a & 1))
            } else {
                (a, a)
            };
            let (bz, bo) = if bn.var == var {
                (bn.zero ^ (b & 1), bn.one ^ (b & 1))
            } else {
                (b, b)
            };
            assert!(depth < self.frames.len());
            self.frames[depth] = Frame {
                key,
                var,
                low: 0,
                high_a: ao,
                high_b: bo,
                returning_high: false,
            };
            depth += 1;
            a = az;
            b = bz;
        }
    }

    /// Computes the union of two symbolic board languages.
    const fn or(&mut self, a: u32, b: u32) -> u32 {
        self.and(a ^ 1, b ^ 1) ^ 1
    }

    /// Returns the Boolean variable for one board cell.
    const fn bit(&mut self, x: usize, y: usize) -> u32 {
        self.node((x * self.rows + self.rows - 1 - y) as u16, 0, 1)
    }

    /// Builds the predicate that a row is entirely occupied or entirely empty.
    const fn row(&mut self, y: usize, all: bool) -> u32 {
        let mut f = if all { 1 } else { 0 };
        let mut x = 0;
        while x < self.cols {
            let b = self.bit(x, y);
            f = if all { self.and(f, b) } else { self.or(f, b) };
            x += 1;
        }
        f
    }

    /// Compacts nodes reachable from two roots and returns their relocated indices.
    const fn collect(&mut self, a: u32, b: u32) -> (u32, u32) {
        let old_len = self.len;
        let mut i = 0;
        while i < old_len {
            self.map[i] = 0;
            i += 1;
        }
        if a >= 2 {
            self.map[(a / 2) as usize] = 1;
        }
        if b >= 2 {
            self.map[(b / 2) as usize] = 1;
        }
        i = old_len;
        while i > 1 {
            i -= 1;
            if self.map[i] != 0 {
                let node = self.nodes[i];
                if node.zero >= 2 {
                    self.map[(node.zero / 2) as usize] = 1;
                }
                if node.one >= 2 {
                    self.map[(node.one / 2) as usize] = 1;
                }
            }
        }
        let mut len = 1;
        i = 1;
        while i < old_len {
            if self.map[i] != 0 {
                let n = self.nodes[i];
                self.map[i] = (len as u32) * 2;
                self.nodes[len] = Raw {
                    var: n.var,
                    zero: self.map[(n.zero / 2) as usize] ^ (n.zero & 1),
                    one: self.map[(n.one / 2) as usize] ^ (n.one & 1),
                };
                len += 1;
            }
            i += 1;
        }
        let a = self.map[(a / 2) as usize] ^ (a & 1);
        let b = self.map[(b / 2) as usize] ^ (b & 1);
        self.len = len;
        // Drop stale hash slots before rehashing relocated nodes.
        i = 0;
        while i < M {
            self.unique[i] = 0;
            i += 1;
        }
        i = 1;
        while i < len {
            let n = self.nodes[i];
            let mut at = mix(((n.zero as u64) << 32 | n.one as u64)
                ^ (n.var as u64).wrapping_mul(0x9e3779b97f4a7c15))
                & (M - 1);
            while self.unique[at] != 0 {
                at = (at + 1) & (M - 1);
            }
            self.unique[at] = i as u32;
            i += 1;
        }
        self.clear_cache();
        (a, b)
    }

    /// Builds the structural board language for the given hole budget.
    const fn carrier(&mut self, holes: usize) -> u32 {
        let modulus = occupancy_modulus(self.cols);
        let mut residues = [0; 4];
        residues[0] = 1;
        let mut v = self.cols * self.rows;
        while v > 0 {
            v -= 1;
            let old = residues;
            let mut q = 0;
            while q < modulus {
                residues[q] = self.node(v as u16, old[q], old[(q + modulus - 1) % modulus]);
                q += 1;
            }
        }
        let mut f = residues[0];
        // Threshold q means at least q holes. Update backwards so each hole is counted once.
        let mut thresholds = [0u32; 373];
        thresholds[0] = 1;
        if holes < self.cols * (self.rows - 1) {
            let mut x = 0;
            while x < self.cols {
                let mut y = 0;
                while y + 1 < self.rows {
                    let mut above = 0;
                    let mut z = y + 1;
                    while z < self.rows {
                        let bit = self.bit(x, z);
                        above = self.or(above, bit);
                        z += 1;
                    }
                    let bit = self.bit(x, y);
                    let hole = self.and(above, bit ^ 1);
                    let mut q = holes + 1;
                    while q > 0 {
                        let add = self.and(hole, thresholds[q - 1]);
                        thresholds[q] = self.or(thresholds[q], add);
                        q -= 1;
                    }
                    y += 1;
                }
                x += 1;
            }
            f = self.and(f, thresholds[holes + 1] ^ 1);
        }
        let mut y = 0;
        while y < self.rows {
            let full = self.row(y, true);
            f = self.and(f, full ^ 1);
            if y > 0 {
                let nonempty = self.row(y, false);
                let below = self.row(y - 1, false);
                let implication = self.or(nonempty ^ 1, below);
                f = self.and(f, implication);
            }
            y += 1;
        }
        // At width three a full L/J row above the cap can clear while leaving
        // a new unsupported cell beside an existing unsupported top run.
        // At width four the only such full row is an I, which clears entirely.
        if self.cols >= 4 {
            let mut unsupported = [0u32; 2];
            let mut long = 0;
            let mut a = 0;
            while a < self.cols {
                let mut end = a + 1;
                while end <= self.cols {
                    let mut term = 1;
                    let mut x = a;
                    let mut below = 0;
                    while x < end {
                        let top = self.bit(x, self.rows - 1);
                        term = self.and(term, top);
                        let b = self.bit(x, self.rows - 2);
                        below = self.or(below, b);
                        x += 1;
                    }
                    if a > 0 {
                        let b = self.bit(a - 1, self.rows - 1);
                        term = self.and(term, b ^ 1);
                    }
                    if end < self.cols {
                        let b = self.bit(end, self.rows - 1);
                        term = self.and(term, b ^ 1);
                    }
                    term = self.and(term, below ^ 1);
                    if end - a > 3 {
                        long = self.or(long, term);
                    } else {
                        let two = self.and(unsupported[0], term);
                        unsupported[1] = self.or(unsupported[1], two);
                        unsupported[0] = self.or(unsupported[0], term);
                    }
                    end += 1;
                }
                a += 1;
            }
            f = self.and(f, long ^ 1);
            f = self.and(f, unsupported[1] ^ 1);
        }
        f
    }

    /// Builds the predicate for a legal zero-clear predecessor with this footprint.
    const fn removable(&mut self, cells: &[u32; 12]) -> u32 {
        let mut term = 1;
        let mut support = 0;
        let mut floor = false;
        let mut occupied_rows = 0;
        let mut x = 0;
        while x < self.cols {
            let col = cells[x];
            occupied_rows |= col;
            if col != 0 {
                let low = col.trailing_zeros() as usize;
                floor |= low == 0;
                if low > 0 {
                    let b = self.bit(x, low - 1);
                    support = self.or(support, b);
                }
                let mut y = low;
                while y < self.rows {
                    let bit = self.bit(x, y);
                    term = self.and(term, if col & (1 << y) != 0 { bit } else { bit ^ 1 });
                    y += 1;
                }
            }
            x += 1;
        }
        if !floor {
            term = self.and(term, support);
        }
        // With separated top runs a single tetromino cannot remove the entire top row.
        // Thus the predecessor also reaches the cap and every remaining row must be nonempty.
        let mut y = 0;
        while y < self.rows {
            if occupied_rows & (1 << y) != 0 {
                let mut other = 0;
                x = 0;
                while x < self.cols {
                    if cells[x] & (1 << y) == 0 {
                        let b = self.bit(x, y);
                        other = self.or(other, b);
                    }
                    x += 1;
                }
                term = self.and(term, other);
            }
            y += 1;
        }
        term
    }

    /// Removes boards that violate the proved capped-history predecessor condition.
    const fn rules(&mut self, holes: usize) -> u32 {
        let mut carrier = self.carrier(holes);
        if self.cols <= 4 || self.rows > 28 {
            return self.collect(carrier, 0).0;
        }
        let mut prev = 0;
        let mut one = 0;
        let mut two = 0;
        let mut x = 0;
        while x < self.cols {
            let bit = self.bit(x, self.rows - 1);
            let start = self.and(bit, prev ^ 1);
            let add = self.and(one, start);
            two = self.or(two, add);
            one = self.or(one, start);
            prev = bit;
            x += 1;
        }
        let mut bad = self.and(carrier, two);
        let roots = self.collect(carrier, bad);
        carrier = roots.0;
        bad = roots.1;
        let pieces = TetrisPiece::all();
        let mut pass = 0;
        let mut since_collect = 0;
        // Vertical I first: it proves removability cheaply for many solid columns.
        while pass < 2 && bad != 0 {
            let mut p = 0;
            while p < pieces.len() && bad != 0 {
                let piece = pieces[p];
                let mut rotation = 0;
                while rotation < piece.num_rotations() && bad != 0 {
                    let rotation_value = Rotation::ALL[rotation as usize];
                    let (raw, width) = piece.shape_cols(rotation_value);
                    let width = width as usize;
                    let height = piece.height(rotation_value) as usize;
                    let vertical = width == 1 && height == 4;
                    if width <= self.cols && height <= self.rows && vertical == (pass == 0) {
                        let mut shift = 32;
                        let mut i = 0;
                        let mut cells = 0;
                        while i < width {
                            let trailing = raw[i].trailing_zeros();
                            if trailing < shift {
                                shift = trailing;
                            }
                            cells += raw[i].count_ones();
                            i += 1;
                        }
                        assert!(cells == 4, "codec needs four-cell tetrominoes");
                        let mut occupied = 0;
                        let mut col = 0;
                        while col < width {
                            occupied |= raw[col] >> shift;
                            col += 1;
                        }
                        assert!(
                            (u32::BITS - occupied.leading_zeros()) as usize == height,
                            "piece height must match engine masks"
                        );
                        let mut yy = 0;
                        while yy < height {
                            let mut row = 0u16;
                            i = 0;
                            while i < width {
                                row |= ((((raw[i] >> shift) >> yy) & 1) as u16) << i;
                                i += 1;
                            }
                            assert!(
                                (row & !(row << 1)).count_ones() == 1,
                                "codec needs contiguous piece rows"
                            );
                            yy += 1;
                        }
                        let mut dy = self.rows - height + 1;
                        while dy > 0 && bad != 0 {
                            dy -= 1;
                            let mut dx = 0;
                            while dx + width <= self.cols && bad != 0 {
                                let mut shape = [0; 12];
                                i = 0;
                                while i < width {
                                    shape[dx + i] = (raw[i] >> shift) << dy;
                                    i += 1;
                                }
                                let possible = self.removable(&shape);
                                bad = self.and(bad, possible ^ 1);
                                since_collect += 1;
                                if since_collect == 16 {
                                    let roots = self.collect(carrier, bad);
                                    carrier = roots.0;
                                    bad = roots.1;
                                    since_collect = 0;
                                }
                                dx += 1;
                            }
                        }
                    }
                    rotation += 1;
                }
                p += 1;
            }
            pass += 1;
        }
        let roots = self.collect(carrier, bad);
        let result = self.and(roots.0, roots.1 ^ 1);
        self.collect(result, 0).0
    }
}

pub struct Diagram<const N: usize> {
    pub nodes: [Node; N],
    pub len: usize,
    pub root: u32,
    pub count: u64,
}

/// Counts satisfying assignments, including skipped board variables.
#[allow(
    clippy::cast_possible_truncation,
    reason = "The preceding assertion checks that the shifted u128 count fits u64."
)]
const fn assignments<const N: usize>(
    diagram: &Diagram<N>,
    totals: &[u64; N],
    edge: u32,
    level: usize,
    vars: usize,
) -> u64 {
    if edge == 0 {
        return 0;
    }
    let (count, free) = if edge == 1 {
        (1, vars - level)
    } else {
        (
            totals[edge as usize],
            diagram.nodes[edge as usize].var as usize - level,
        )
    };
    assert!(free < 64, "codec count exceeds u64");
    let v = (count as u128) << free;
    assert!(v <= u64::MAX as u128, "codec count exceeds u64");
    v as u64
}

/// Constructs a reduced decision diagram and its completion counts at compile time.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Workspace indices fit u32; every u128 completion total is checked against u64::MAX before narrowing."
)]
pub const fn build<const N: usize, const M: usize, const OUT: usize>(
    c: usize,
    r: usize,
    h: usize,
) -> Diagram<OUT> {
    let holes = if h < c * (r - 1) { h } else { c * (r - 1) };
    let mut b = Builder::<N, M>::new(c, r);
    let root = b.rules(holes);
    let mut result = Diagram {
        nodes: [Node::EMPTY; OUT],
        len: 2,
        root: 0,
        count: 0,
    };
    let mut seen = [0u8; N];
    if root >= 2 {
        seen[(root / 2) as usize] |= 1 << (root & 1);
    }
    let mut at = b.len;
    while at > 1 {
        at -= 1;
        let node = b.nodes[at];
        let mut sign = 0;
        while sign < 2 {
            if seen[at] & (1 << sign) != 0 {
                let zero = node.zero ^ sign;
                let one = node.one ^ sign;
                if zero >= 2 {
                    seen[(zero / 2) as usize] |= 1 << (zero & 1);
                }
                if one >= 2 {
                    seen[(one / 2) as usize] |= 1 << (one & 1);
                }
            }
            sign += 1;
        }
    }
    let mut map = [0u32; OUT];
    map[1] = 1;
    let mut totals = [0u64; OUT];
    totals[1] = 1;
    let mut i = 1;
    while i < b.len {
        let mut sign = 0;
        while sign < 2 {
            if seen[i] & (1 << sign) != 0 {
                let n = b.nodes[i];
                let zero = map[(n.zero ^ sign) as usize];
                let one = map[(n.one ^ sign) as usize];
                let z = assignments(&result, &totals, zero, n.var as usize + 1, c * r);
                let o = assignments(&result, &totals, one, n.var as usize + 1, c * r);
                let count = z as u128 + o as u128;
                assert!(count <= u64::MAX as u128, "codec count exceeds u64");
                let at = result.len;
                assert!(at < OUT);
                result.len += 1;
                result.nodes[at] = Node {
                    zero,
                    one,
                    zero_count: z,
                    var: n.var,
                };
                totals[at] = count as u64;
                map[i * 2 + sign as usize] = at as u32;
            }
            sign += 1;
        }
        i += 1;
    }
    result.root = map[root as usize];
    result.count = assignments(&result, &totals, result.root, 0, c * r);
    assert!(result.count > 0);
    if occupancy_modulus(c) > 1 {
        assert!(result.root >= 2 && result.nodes[result.root as usize].var == 0);
        let mut at = 2;
        while at < result.len {
            let n = result.nodes[at];
            let children = [n.zero, n.one];
            let mut j = 0;
            while j < 2 {
                let child = children[j];
                if child == 1 {
                    assert!(n.var as usize + 1 == c * r);
                } else if child > 1 {
                    assert!(result.nodes[child as usize].var == n.var + 1);
                }
                j += 1;
            }
            at += 1;
        }
    }
    result
}

pub struct Configuration<const C: usize, const R: usize, const H: usize>;

impl<const C: usize, const R: usize, const H: usize> Configuration<C, R, H>
where
    [(); capacity(C, R, H)]:,
    [(); slots(C, R, H)]:,
    [(); capacity(C, R, H) * 2]:,
{
    // Generic-const normalization can visit GRAPH even when Source selects arithmetic.
    // Guard construction here too, so those configurations never run the graph builder.
    pub const GRAPH: Diagram<{ capacity(C, R, H) * 2 }> = if arithmetic::applies(C, R, H) {
        Diagram {
            nodes: [Node::EMPTY; capacity(C, R, H) * 2],
            len: 0,
            root: 0,
            count: arithmetic::board_count(R),
        }
    } else {
        build::<{ capacity(C, R, H) }, { slots(C, R, H) }, { capacity(C, R, H) * 2 }>(C, R, H)
    };
}

pub(crate) trait Source: private::Sealed {
    const HOLES: usize;
    const NODES: &'static [Node];
    const LEN: usize;
    const ROOT: u32;
    const COUNT: u64;
    const SMALL: usize;
    const MEDIUM: usize;
    const WIDE: usize;
    const LEVELS: usize;
}

impl<const C: usize, const R: usize, const H: usize> private::Sealed for Dimensions<C, R, H> {}

impl<const C: usize, const R: usize, const H: usize> Source for Dimensions<C, R, H>
where
    [(); capacity(C, R, H)]:,
    [(); slots(C, R, H)]:,
    [(); capacity(C, R, H) * 2]:,
{
    const HOLES: usize = if H < C * (R - 1) { H } else { C * (R - 1) };
    const NODES: &'static [Node] = if arithmetic::applies(C, R, H) {
        &[]
    } else {
        &Configuration::<C, R, H>::GRAPH.nodes
    };
    const LEN: usize = if arithmetic::applies(C, R, H) {
        0
    } else {
        Configuration::<C, R, H>::GRAPH.len
    };
    const ROOT: u32 = if arithmetic::applies(C, R, H) {
        0
    } else {
        Configuration::<C, R, H>::GRAPH.root
    };
    const COUNT: u64 = if arithmetic::applies(C, R, H) {
        arithmetic::board_count(R)
    } else {
        Configuration::<C, R, H>::GRAPH.count
    };
    const SMALL: usize = if storage_kind(Self::LEN, Self::COUNT) == 0 {
        Self::LEN
    } else {
        0
    };
    const MEDIUM: usize = if storage_kind(Self::LEN, Self::COUNT) == 1 {
        Self::LEN
    } else {
        0
    };
    const WIDE: usize = if storage_kind(Self::LEN, Self::COUNT) == 2 {
        Self::LEN
    } else {
        0
    };
    const LEVELS: usize = if occupancy_modulus(C) == 1 {
        Self::LEN
    } else {
        0
    };
}

/// Selects the narrowest lossless node representation.
const fn storage_kind(len: usize, count: u64) -> usize {
    if len <= u16::MAX as usize + 1 && count <= u16::MAX as u64 {
        0
    } else if len <= u16::MAX as usize + 1 && count <= u32::MAX as u64 {
        1
    } else {
        2
    }
}

/// Packs nodes whose children and completion counts fit in u16.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Each record is asserted to fit its u16 children and completion count before narrowing."
)]
const fn small<const N: usize>(nodes: &[Node]) -> [SmallNode; N] {
    let mut out = [SmallNode {
        zero: 0,
        one: 0,
        weight: 0,
    }; N];
    let mut i = 0;
    while i < N {
        let n = nodes[i];
        assert!(
            n.zero <= u16::MAX as u32
                && n.one <= u16::MAX as u32
                && n.zero_count <= u16::MAX as u64
        );
        out[i] = SmallNode {
            zero: n.zero as u16,
            one: n.one as u16,
            weight: n.zero_count as u16,
        };
        i += 1;
    }
    out
}

/// Packs u16 children with u32 completion counts.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Each record is asserted to fit its u16 children and u32 completion count before narrowing."
)]
const fn medium<const N: usize>(nodes: &[Node]) -> [MediumNode; N] {
    let mut out = [MediumNode {
        zero: 0,
        one: 0,
        weight: 0,
    }; N];
    let mut i = 0;
    while i < N {
        let n = nodes[i];
        assert!(
            n.zero <= u16::MAX as u32
                && n.one <= u16::MAX as u32
                && n.zero_count <= u32::MAX as u64
        );
        out[i] = MediumNode {
            zero: n.zero as u16,
            one: n.one as u16,
            weight: n.zero_count as u32,
        };
        i += 1;
    }
    out
}

/// Packs u32 children with u64 completion counts.
const fn wide<const N: usize>(nodes: &[Node]) -> [WideNode; N] {
    let mut out = [WideNode {
        zero: 0,
        one: 0,
        weight: 0,
    }; N];
    let mut i = 0;
    while i < N {
        let n = nodes[i];
        out[i] = WideNode {
            zero: n.zero,
            one: n.one,
            weight: n.zero_count,
        };
        i += 1;
    }
    out
}

/// Extracts decision levels for diagrams with skipped variables.
const fn levels<const N: usize>(nodes: &[Node]) -> [u16; N] {
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = nodes[i].var;
        i += 1;
    }
    out
}

impl<T: Source> CompiledConfig for T
where
    [(); T::SMALL]:,
    [(); T::MEDIUM]:,
    [(); T::WIDE]:,
    [(); T::LEVELS]:,
{
    const BOARD_COUNT: u64 = T::COUNT;
    const TABLES: &'static Tables = &Tables {
        holes: T::HOLES,
        root: T::ROOT,
        count: T::COUNT,
        nodes: if T::SMALL != 0 {
            Nodes::Small(&small::<{ T::SMALL }>(T::NODES))
        } else if T::MEDIUM != 0 {
            Nodes::Medium(&medium::<{ T::MEDIUM }>(T::NODES))
        } else {
            Nodes::Wide(&wide::<{ T::WIDE }>(T::NODES))
        },
        levels: &levels::<{ T::LEVELS }>(T::NODES),
    };
}

#[cfg(test)]
mod tests {
    use super::storage_kind;
    use tetris_game::{
        Column, Rotation, TetrisBoard, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement,
    };

    /// Checks normalized engine masks against placements, including symmetric rotations.
    #[test]
    fn engine_shapes_normalize_to_every_placement() {
        let normalized = |piece: TetrisPiece, rotation: Rotation| {
            let (columns, width) = piece.shape_cols(rotation);
            let shift = u32::BITS - u32::from(piece.height(rotation));
            (columns.map(|column| column >> shift), width)
        };
        for &placement in TetrisPiecePlacement::<10>::ALL_PLACEMENTS {
            let (columns, width) = normalized(placement.piece, placement.orientation.rotation);
            let mut board = TetrisBoard::<10, 20>::new();
            let result = board.apply_piece_placement(placement);
            assert_eq!(result.lines_cleared, 0);
            assert_eq!(board.count(), 4);
            let start = usize::from(placement.orientation.column.inner());
            let expected = std::array::from_fn(|x| {
                if (start..start + usize::from(width)).contains(&x) {
                    columns[x - start]
                } else {
                    0
                }
            });
            assert_eq!(board.as_limbs(), &expected);
        }
        // Also compare the symmetric rotations omitted from the canonical catalog.
        for piece in TetrisPiece::all() {
            for rotation in Rotation::ALL {
                let (columns, _) = normalized(piece, rotation);
                let mut board = TetrisBoard::<10, 20>::new();
                board.apply_piece_placement(TetrisPiecePlacement {
                    piece,
                    orientation: TetrisPieceOrientation {
                        rotation,
                        column: Column::new(0),
                    },
                });
                assert_eq!(&board.as_limbs()[..4], &columns);
            }
        }
    }

    #[test]
    fn packed_children_and_weights_are_selected_without_narrowing_loss() {
        assert_eq!(std::mem::size_of::<super::MediumNode>(), 8);
        assert_eq!(std::mem::align_of::<super::MediumNode>(), 8);
        assert_eq!(storage_kind(65536, 65535), 0);
        assert_eq!(storage_kind(65536, 65536), 1);
        assert_eq!(storage_kind(65536, u32::MAX as u64), 1);
        assert_eq!(storage_kind(65536, u32::MAX as u64 + 1), 2);
        assert_eq!(storage_kind(65537, 1), 2);
    }
}
