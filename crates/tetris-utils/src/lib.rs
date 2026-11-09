#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(const_index)]
#![doc(test(attr(feature(generic_const_exprs, const_trait_impl))))]

use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::hint::spin_loop;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut, Index};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use tetris_macros::inline_conditioned;

#[doc(hidden)]
pub const fn checked_assert(condition: bool, message: &str) -> usize {
    assert!(condition, "{}", message);
    0
}

/// Checks an assertion in a const expression, returning zero for an array-length bound.
///
/// ```
/// use tetris_utils::const_assert;
/// const _: [(); const_assert!(4 < 8, "four must be less than eight")] = [];
/// ```
///
/// A false assertion fails during const evaluation:
///
/// ```compile_fail,E0080
/// use tetris_utils::const_assert;
/// const _: [(); const_assert!(4 > 8, "four must be greater than eight")] = [];
/// ```
#[macro_export]
macro_rules! const_assert {
    ($condition:expr, $message:literal $(,)?) => {
        $crate::checked_assert($condition, $message)
    };
}

/// Converts an array of 8 bit values (0 or 1) into a single byte.
///
/// Each element in the array is treated as a single bit, with the first element
/// becoming the most significant bit (bit 7) and the last element becoming the
/// least significant bit (bit 0).
///
/// # Arguments
///
/// * `bits` - An array of 8 bytes, each containing either 0 or 1
///
/// # Returns
///
/// A byte formed by packing the 8 bits together
///
/// # Example
///
/// ```
/// use tetris_utils::bits_to_byte;
/// let bits = [1_u8, 0, 1, 0, 1, 0, 1, 0];
/// assert_eq!(bits_to_byte(&bits), 0b10101010);
/// ```
#[inline_conditioned(always)]
pub const fn bits_to_byte(bits: &[u8; u8::BITS as usize]) -> u8 {
    let x = u64::from_le_bytes(*bits) & 0x0101_0101_0101_0101;
    ((x.wrapping_mul(0x8040_2010_0804_0201)) >> 56) as u8
}

/// Expands a packed byte into 8 one-bit lanes (`0`/`1`), MSB-first.
///
/// The returned array is ordered such that:
/// - `bits[0]` is the most-significant bit (bit 7) of `byte`
/// - `bits[7]` is the least-significant bit (bit 0) of `byte`
///
/// This is the inverse of [`bits_to_byte`] (and matches the bit ordering it expects).
///
/// ## Example
///
/// ```
/// use tetris_utils::{bits_to_byte, byte_to_bits};
/// let byte = 0b1010_1010u8;
/// let bits = byte_to_bits(byte);
/// assert_eq!(bits, [1, 0, 1, 0, 1, 0, 1, 0]);
/// assert_eq!(bits_to_byte(&bits), byte);
/// ```
#[inline_conditioned(always)]
pub const fn byte_to_bits(byte: u8) -> [u8; u8::BITS as usize] {
    let mut x = (byte as u64) & 0xFF;
    x = (x | (x << 28)) & 0x0000_000F_0000_000F;
    x = (x | (x << 14)) & 0x0003_0003_0003_0003;
    x = (x | (x << 7)) & 0x0101_0101_0101_0101;
    x.to_be_bytes()
}

const _: () = {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "The loop visits only 0..256, so every value fits u8."
    )]
    const fn verify_all_u8() -> bool {
        let mut i: u16 = 0;
        while i < 256 {
            let b = i as u8;
            let bits = byte_to_bits(b);
            if bits_to_byte(&bits) != b {
                return false;
            }
            i += 1;
        }
        true
    }

    // Compile-time assertion: index out of bounds if verify_all_u8() returns false
    [()][(!verify_all_u8()) as usize];
};

/// Swaps chunks `i` and `j` in `x`.
///
/// Each chunk contains `BITS` bits. Chunks are numbered from the least significant
/// end of the value, starting at zero. All other bits are preserved.
///
/// `BITS` must be between 1 and 63; invalid widths are rejected at compile time.
/// Both chunks must fit entirely within `x`.
///
/// # Examples
///
/// ```rust
/// use tetris_utils::swap_bit_chunks;
///
/// // Four-bit chunks, from least to most significant: [1, 2, 3].
/// let mut x = 0x0000_0000_0000_0321_u64;
/// swap_bit_chunks::<4>(&mut x, 0, 2);
///
/// // The outer chunks are swapped and the middle chunk is preserved.
/// assert_eq!(x, 0x0000_0000_0000_0123);
/// ```
///
/// Zero-width chunks are rejected:
///
/// ```compile_fail,E0080
/// use tetris_utils::swap_bit_chunks;
/// let mut x = 0u64;
/// swap_bit_chunks::<0>(&mut x, 0, 0);
/// ```
///
/// The width must be smaller than the word:
///
/// ```compile_fail,E0080
/// use tetris_utils::swap_bit_chunks;
/// let mut x = 0u64;
/// swap_bit_chunks::<64>(&mut x, 0, 0);
/// ```
#[inline_conditioned(always)]
pub const fn swap_bit_chunks<const BITS: u32>(x: &mut u64, i: u32, j: u32)
where
    [(); (BITS > 0) as usize - 1]:,
    [(); (BITS < u64::BITS) as usize - 1]:,
{
    let i = i * BITS;
    let j = j * BITS;
    let different = ((*x >> i) ^ (*x >> j)) & ((1u64 << BITS) - 1);
    *x ^= (different << i) ^ (different << j);
}

/// Transposes an `ROWS × COLS` bit matrix stored in `[u8; N]` in-place.
///
/// The buffer holds a tightly packed bit grid: bit at linear index
/// `row * COLS + col` represents cell `(row, col)`. After transposition the
/// matrix is `COLS × ROWS`, packed into the same `N` bytes.
///
/// Generic Hacker's Delight transpose padded to `32 × 32` on a `[u32; 32]`
/// lane array. Works for any `(ROWS, COLS)` with `max(ROWS, COLS) ≤ 32` —
/// covering all Tetris-board shapes (10×20, 10×32, 32×10) plus 8×8, 4×4,
/// 16×16, etc. Larger matrices fall through to a bit-scatter loop.
/// Compile-time check: fails when `N * 8 < ROWS * COLS` or when
/// `ROWS`/`COLS`/`N` aren't in the supported `repeat_idx_unroll!` arities
/// ({0..16, 18, 20, 25, 32, 40, 64}).
///
/// # Example
///
/// ```
/// use tetris_utils::transpose_bits;
///
/// let mut m = [0xFFu8, 0, 0, 0, 0, 0, 0, 0];
/// transpose_bits::<8, 8, 8>(&mut m);
/// for i in 0..8 { assert_eq!(m[i], 0x01); }
///
/// let mut m = [0x0Fu8, 0x00];
/// transpose_bits::<4, 4, 2>(&mut m);
/// assert_eq!(m, [0x11, 0x11]);
/// ```
#[inline_conditioned(always)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "The fast path requires ROWS and COLS <= 32; shift offsets are reduced modulo 8 or 32, and byte extraction intentionally discards higher lanes."
)]
pub const fn transpose_bits<const ROWS: usize, const COLS: usize, const N: usize>(m: &mut [u8; N]) {
    assert!(
        N * 8 >= ROWS * COLS,
        "byte array too small for ROWS * COLS bits"
    );

    // Generic lane-u32 path: pad to 32×32, HD-transpose, spill via a u32
    // scratch. The two `const { ... }` blocks each evaluate to a bool at
    // compile time; the `&&` between them is a regular runtime expression
    // over two compile-time constants, so LLVM folds the whole dispatch and
    // only the matching branch survives monomorphization. (A single
    // `const { ROWS <= 32 && COLS <= 32 }` block is rejected — `&&` is
    // short-circuiting "control flow" inside a generic const expression.)
    if const { ROWS <= 32 } && const { COLS <= 32 } {
        let orig = *m;
        let mut a = [0u32; 32];

        // Load ROWS rows: row R occupies bits [R*COLS, R*COLS+COLS-1],
        // spanning up to 4 source bytes from byte `R*COLS/8` with shift
        // `(R*COLS) % 8`. Reads 4 bytes unconditionally; out-of-bounds bytes
        // contribute 0 (compile-time masked).
        //
        // `R` comes from the macro as a `const` item, but `COLS` is an outer
        // generic param — so anything derived from both must be a `let`
        // (Rust forbids inner `const` items from referencing outer generics).
        // After monomorphization every value here is a compile-time literal
        // and LLVM folds it away.
        crate::repeat_idx_unroll!(ROWS, R, {
            let bs: usize = (R * COLS) / 8;
            let sh: u32 = ((R * COLS) % 8) as u32;
            let row_mask: u32 = if COLS >= 32 {
                u32::MAX
            } else {
                1u32.wrapping_shl(COLS as u32).wrapping_sub(1)
            };
            let combined = orig[bs % N] as u32
                | ((if bs + 1 < N { orig[(bs + 1) % N] } else { 0 } as u32) << 8)
                | ((if bs + 2 < N { orig[(bs + 2) % N] } else { 0 } as u32) << 16)
                | ((if bs + 3 < N { orig[(bs + 3) % N] } else { 0 } as u32) << 24);
            a[R % 32] = (combined >> sh) & row_mask;
        });

        // Inlined LSB-is-col-0 Hacker's Delight transpose on the [u32; 32]
        // lane array. The classic `t = (a[k] ^ (a[k+j] >> j)) & m` is
        // mirrored to swap `a[k]`'s high half with `a[k+j]`'s low half,
        // matching our LSB convention.
        let (mut j, mut mask): (usize, u32) = (16, 0x0000_FFFF);
        while j > 0 {
            let mut k: usize = 0;
            while k < 32 {
                let t = ((a[k] >> j) ^ a[k + j]) & mask;
                a[k + j] ^= t;
                a[k] ^= t << j;
                k = ((k | j) + 1) & !j;
            }
            j >>= 1;
            mask ^= mask << j;
        }

        // Pack COLS transposed rows (each ROWS bits) into a u32 scratch, then
        // spill to bytes. The scratch keeps writes word-wide; spilling is
        // pure byte writes (no R-M-W).
        let mut out_u32 = [0u32; 32];
        crate::repeat_idx_unroll!(COLS, D, {
            let bit: usize = D * ROWS;
            let w: usize = bit / 32;
            let sh: u32 = (bit % 32) as u32;
            let row_mask: u32 = if ROWS >= 32 {
                u32::MAX
            } else {
                1u32.wrapping_shl(ROWS as u32).wrapping_sub(1)
            };
            let spans: bool = sh + (ROWS as u32) > 32;
            let v = a[D % 32] & row_mask;
            out_u32[w % 32] |= v << (sh & 31);
            if spans {
                out_u32[(w + 1) % 32] |= v >> ((32 - sh) & 31);
            }
        });
        crate::repeat_idx_unroll!(N, B, {
            m[B % N] = (out_u32[(B / 4) % 32] >> (((B % 4) * 8) as u32 & 31)) as u8;
        });
        return;
    }

    // Fallback: bit-scatter loop for max(ROWS, COLS) > 32.
    let orig = *m;
    let mut b = 0;
    while b < N {
        m[b] = 0;
        b += 1;
    }
    let (mut src, mut row) = (0usize, 0usize);
    while row < ROWS {
        let (mut dst, mut col) = (row, 0usize);
        while col < COLS {
            let bit = (orig[src >> 3] >> (src & 7)) & 1;
            m[dst >> 3] |= bit << (dst & 7);
            src += 1;
            dst += ROWS;
            col += 1;
        }
        row += 1;
    }
}
pub struct BitMask<const N: usize>(u64);

impl<const N: usize> BitMask<N> {
    /// Creates a mask with a logical width of `N` bits, preserving the raw value.
    /// `N` must be in `1..=64`; bits above that width remain available through `inner()`.
    ///
    /// ```
    /// use tetris_utils::BitMask;
    /// const MASK: BitMask<4> = BitMask::new_from_u64(0xff);
    /// assert_eq!(MASK.inner(), 0xff);
    /// assert_eq!(MASK.as_slice(), [1; 4]);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_utils::BitMask;
    /// let _ = BitMask::<0>::new_from_u64(0);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_utils::BitMask;
    /// let _ = BitMask::<65>::new_from_u64(0);
    /// ```
    #[inline_conditioned(always)]
    pub const fn new_from_u64(x: u64) -> Self {
        let _ = const { const_assert!(N > 0, "bit mask must contain at least one bit") };
        let _ = const { const_assert!(N <= u64::BITS as usize, "bit mask must fit in u64") };
        Self(x)
    }

    #[inline_conditioned(always)]
    pub const fn inner(&self) -> u64 {
        self.0
    }

    #[inline_conditioned(always)]
    pub const fn as_slice(&self) -> [u8; N] {
        let mut out = [0u8; N];
        crate::repeat_idx_unroll!(N, I, {
            out[I] = ((self.0 >> I) & 1) as u8;
        });
        out
    }

    /// Zero-extend the low `N` bits into a wider `BitMask<M>` (`M >= N`).
    ///
    /// ```compile_fail,E0080
    /// use tetris_utils::BitMask;
    /// let _ = BitMask::<4>::new_from_u64(0).widen::<65>();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_utils::BitMask;
    /// let _ = BitMask::<8>::new_from_u64(0).widen::<4>();
    /// ```
    #[inline_conditioned(always)]
    pub const fn widen<const M: usize>(&self) -> BitMask<M>
    where
        [(); M - N]:,
    {
        BitMask::new_from_u64(self.0 & (u64::MAX >> (64 - N)))
    }

    /// Truncate to the low `M` bits, producing a `BitMask<M>` (`M <= N`).
    ///
    /// ```compile_fail,E0080
    /// use tetris_utils::BitMask;
    /// let _ = BitMask::<4>::new_from_u64(0).narrow::<0>();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_utils::BitMask;
    /// let _ = BitMask::<4>::new_from_u64(0).narrow::<8>();
    /// ```
    #[inline_conditioned(always)]
    pub const fn narrow<const M: usize>(&self) -> BitMask<M>
    where
        [(); N - M]:,
    {
        BitMask::new_from_u64(self.0 & (u64::MAX >> (64 - M)))
    }
}

/// A spin lock for shared-mutual exclusion.
///
/// Used internally by [`VecPool`] for low-contention synchronization.
/// Only one guard can access the value at a time. Dropping the guard releases
/// the lock, including during unwinding; the lock does not implement poisoning.
/// Locking again on the same thread while holding a guard will spin indefinitely.
///
/// Sharing the lock requires a value that can be transferred between threads:
///
/// ```compile_fail,E0277
/// use std::rc::Rc;
/// use tetris_utils::SpinLock;
/// fn require_sync<T: Sync>() {}
/// require_sync::<SpinLock<Rc<usize>>>();
/// ```
#[derive(Debug)]
pub struct SpinLock<T: Debug> {
    lock: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Debug> Sync for SpinLock<T> where T: Send {}

/// Exclusive access to a [`SpinLock`]'s value; releases the lock when dropped.
///
/// Sharing a guard also requires the protected value to be `Sync`:
///
/// ```compile_fail,E0277
/// use std::cell::Cell;
/// use tetris_utils::SpinLockGuard;
/// fn require_sync<T: Sync>() {}
/// require_sync::<SpinLockGuard<'static, Cell<usize>>>();
/// ```
#[must_use = "the lock is released immediately if its guard is discarded"]
pub struct SpinLockGuard<'a, T> {
    lock: &'a AtomicBool,
    data: &'a mut T,
}

impl<'a, T> Drop for SpinLockGuard<'a, T> {
    fn drop(&mut self) {
        // Publish the protected writes before another thread acquires the flag.
        // This cannot panic, so unwinding also releases the lock without poisoning.
        self.lock.store(false, AtomicOrdering::Release);
    }
}

impl<T: Debug> SpinLock<T> {
    pub const fn new(data: T) -> Self {
        SpinLock {
            lock: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Spins until exclusive access is acquired. This lock is not reentrant.
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        while self.lock.swap(true, AtomicOrdering::Acquire) {
            spin_loop();
        }
        let data = unsafe { &mut *self.data.get() };
        SpinLockGuard {
            lock: &self.lock,
            data,
        }
    }

    /// Attempts to acquire exclusive access once, returning `None` if already held.
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        if self
            .lock
            .compare_exchange(
                false,
                true,
                AtomicOrdering::Acquire,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
        {
            let data = unsafe { &mut *self.data.get() };
            Some(SpinLockGuard {
                lock: &self.lock,
                data,
            })
        } else {
            None
        }
    }
}

impl<'a, T> Deref for SpinLockGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> DerefMut for SpinLockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

/// Fast fixed-capacity binary MIN-heap.
///
/// - Stack-allocated, zero heap allocations
/// - Compile-time loop unrolling with early exit
/// - Optimized for beam search (keep best K items)
pub struct FixedBinMinHeap<T: Copy + Ord, const SIZE: usize> {
    // Storage invariant at API boundaries and before user comparisons:
    // len <= SIZE, and every slot in 0..len holds an initialized T.
    // Slots in len..SIZE may be uninitialized. Bounds do not depend on heap ordering.
    data: [MaybeUninit<T>; SIZE],
    len: usize,
}

impl<T: Copy + Ord, const SIZE: usize> Default for FixedBinMinHeap<T, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Ord, const SIZE: usize> FixedBinMinHeap<T, SIZE> {
    // Calculate the minimum number of steps needed to sift up/down the heap
    // Successful const evaluation requires SIZE <= usize::MAX / 2; larger
    // capacities overflow `level *= 2`. This also bounds child-index arithmetic.
    pub const STEPS: usize = {
        let mut cap = 1usize;
        let mut level = 2usize;
        let mut steps = 0usize;
        while cap < SIZE {
            cap += level;
            level *= 2;
            steps += 1;
        }
        steps
    };

    #[inline_conditioned(always)]
    pub const fn new() -> Self {
        Self {
            len: 0,
            data: [MaybeUninit::uninit(); SIZE],
        }
    }

    #[inline_conditioned(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline_conditioned(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline_conditioned(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline_conditioned(always)]
    pub fn peek_min(&self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        Some(unsafe { self.data[0].assume_init() })
    }

    #[inline_conditioned(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Return a mutable slice of the initialized elements [0..len)
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    /// Push into heap. Unrolled sift-up with early exit.
    ///
    /// # Panics
    ///
    /// Panics if the heap is full.
    #[inline_conditioned(always)]
    pub fn push(&mut self, v: T) {
        assert!(self.len < SIZE, "FixedBinMinHeap::push called when full");
        let i = self.len;
        self.len = i + 1;
        unsafe {
            self.data.get_unchecked_mut(i).write(v);
            self.sift_up_min_unrolled(i);
        }
    }

    /// Pop minimum (root).
    #[inline_conditioned(always)]
    pub fn pop_min(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        unsafe {
            let out = self.data.get_unchecked(0).assume_init_read();
            self.len -= 1;
            if self.len != 0 {
                let last = self.data.get_unchecked(self.len).assume_init_read();
                self.data.get_unchecked_mut(0).write(last);
                self.sift_down_min_unrolled(0);
            }
            Some(out)
        }
    }

    /// Replace root and sift down (fast path).
    ///
    /// # Panics
    ///
    /// Panics if the heap is empty, including when its capacity is zero.
    #[inline_conditioned(always)]
    pub fn replace_min(&mut self, v: T) {
        assert!(
            self.len != 0,
            "FixedBinMinHeap::replace_min called when empty"
        );
        unsafe {
            self.data.get_unchecked_mut(0).write(v);
            self.sift_down_min_unrolled(0);
        }
    }

    /// Beam-search helper (same semantics as your old code):
    /// root is "worst kept" (smallest by Ord) and better items compare as larger.
    ///
    /// For plain T where "higher is better", this works as-is:
    /// - if full and v <= root => reject
    /// - else replace root
    ///
    /// Returns whether the candidate was inserted. A zero-capacity heap always rejects it.
    #[inline_conditioned(always)]
    pub fn push_if_better_min_heap(&mut self, v: T) -> bool {
        if SIZE == 0 {
            return false;
        }
        if self.len < SIZE {
            self.push(v);
            return true;
        }
        // Full: compare against root.
        unsafe {
            let root = self.data.get_unchecked(0).assume_init_ref();
            if v <= *root {
                return false;
            }
        }
        self.replace_min(v);
        true
    }

    // ----------------- bounded sift-up/down with early exit -----------------

    /// # Safety
    ///
    /// The storage invariant must hold and start must be below self.len.
    #[inline_conditioned(always)]
    unsafe fn sift_up_min_unrolled(&mut self, start: usize) {
        let ptr = self.data.as_mut_ptr().cast::<T>();
        let mut i = start;

        /// # Safety
        ///
        /// ptr must provide exclusive access to initialized T slots through *i,
        /// and the index variable must be separate from that storage.
        #[inline(always)]
        unsafe fn step_up<T: Copy + Ord>(ptr: *mut T, i: &mut usize) -> bool {
            let child = *i;
            if child == 0 {
                return false; // At root, can't go up
            }

            // Binary heap: parent = (child - 1) / 2
            let p = (child - 1) >> 1;
            let vc = unsafe { *ptr.add(child) };
            let vp = unsafe { *ptr.add(p) };

            // Branchless select: swap with parent or self.
            let do_swap = (vc < vp) as usize;
            let mask = 0usize.wrapping_sub(do_swap);
            let swap_idx = child ^ ((child ^ p) & mask);
            unsafe {
                ptr::swap(ptr.add(child), ptr.add(swap_idx));
            }
            *i = swap_idx;
            do_swap != 0
        }

        let mut step = 0usize;
        while step < Self::STEPS {
            if !unsafe { step_up::<T>(ptr, &mut i) } {
                return; // Early exit when no swap happened
            }
            step += 1;
        }
    }

    /// # Safety
    ///
    /// The storage invariant must hold and start must be below self.len.
    #[inline(always)]
    unsafe fn sift_down_min_unrolled(&mut self, start: usize) {
        let ptr = self.data.as_mut_ptr().cast::<T>();
        let len = self.len;
        let mut i = start;

        /// # Safety
        ///
        /// ptr must provide exclusive access to len initialized T slots, *i < len,
        /// and len <= usize::MAX / 2. The index variable must be separate from storage.
        #[inline(always)]
        unsafe fn step_down<T: Copy + Ord>(ptr: *mut T, len: usize, i: &mut usize) -> bool {
            let p = *i;
            // Binary heap: left child = 2*p + 1, right child = 2*p + 2
            let left = (p << 1) + 1;
            let right = left + 1;

            // No children - we're at a leaf
            if left >= len {
                return false;
            }

            // Find minimum child: either left-only or min(left, right)
            let vp = unsafe { *ptr.add(p) };

            let (min_val, min_idx) = if right >= len {
                (unsafe { *ptr.add(left) }, left)
            } else {
                // Both children exist - find minimum
                let vl = unsafe { *ptr.add(left) };
                let vr = unsafe { *ptr.add(right) };
                if vr < vl { (vr, right) } else { (vl, left) }
            };

            // Swap if child < parent
            if min_val < vp {
                unsafe {
                    ptr::swap(ptr.add(p), ptr.add(min_idx));
                }
                *i = min_idx;
                true
            } else {
                false // Heap property satisfied
            }
        }

        let mut step = 0usize;
        while step < Self::STEPS {
            if !unsafe { step_down::<T>(ptr, len, &mut i) } {
                return; // Early exit when no swap happened
            }
            step += 1;
        }
    }
}

/// HeaplessVec

#[derive(Debug, Clone, Copy)]
pub struct HeaplessVec<T: Copy, const N: usize> {
    // Storage invariant at API boundaries and before callbacks: len <= N,
    // and every slot in 0..len holds an initialized T. Remaining slots may be
    // uninitialized. T: Copy lets compaction retain valid copies in old slots.
    data: [MaybeUninit<T>; N],
    len: usize,
}

impl<T: Copy, const N: usize> Default for HeaplessVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> HeaplessVec<T, N> {
    const ELEM: MaybeUninit<T> = MaybeUninit::uninit();
    const INIT: [MaybeUninit<T>; N] = [Self::ELEM; N]; // important for optimization of `new`

    /// Creates a new empty `HeaplessVec` with fixed capacity `N`.
    ///
    /// # Returns
    ///
    /// An empty vector with no allocated elements
    pub const fn new() -> Self {
        Self {
            data: Self::INIT,
            len: 0,
        }
    }

    /// Returns a mutable raw pointer to the vector's buffer.
    ///
    /// # Pointer validity
    ///
    /// The pointer is aligned for `T` and points to `N` inline storage slots.
    /// Only `0..self.len()` is guaranteed initialized; `self.len()..N` may be uninitialized.
    /// Moving or dropping this vector invalidates the pointer. Dereferencing it must
    /// respect Rust's borrowing rules and preserve the initialized prefix. Writing spare
    /// slots does not change the vector's length.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    /// Returns the maximum number of elements the vector can hold.
    ///
    /// # Returns
    ///
    /// The capacity `N` of this vector
    #[inline_conditioned(always)]
    pub const fn capacity() -> usize {
        N
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Returns
    ///
    /// `true` if the vector is empty, `false` otherwise
    #[inline_conditioned(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the vector has reached its maximum capacity.
    ///
    /// # Returns
    ///
    /// `true` if the vector is full, `false` otherwise
    #[inline_conditioned(always)]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Returns the number of elements currently in the vector.
    ///
    /// # Returns
    ///
    /// The number of initialized elements
    #[inline_conditioned(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Appends an element to the end of the vector.
    ///
    /// # Arguments
    ///
    /// * `item` - The element to append
    ///
    /// # Panics
    ///
    /// Panics if the vector is full. Use [`Self::try_push`] to handle a full vector.
    #[inline_conditioned(always)]
    pub fn push(&mut self, item: T) {
        self.data[self.len].write(item);
        self.len += 1;
    }

    /// Attempts to push an element to the end of the vector.
    ///
    /// Returns `true` if the element was pushed, `false` if the vector is full.
    pub fn try_push(&mut self, item: T) -> bool {
        if self.len == N {
            return false;
        }
        unsafe {
            *self.data.get_unchecked_mut(self.len) = MaybeUninit::new(item);
        }
        self.len += 1;
        true
    }

    /// Returns a reference to the element at the given index, or `None` if out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve
    ///
    /// # Returns
    ///
    /// `Some(&T)` if the index is valid, `None` otherwise
    #[inline_conditioned(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        Some(unsafe { self.data[index].assume_init_ref() })
    }

    /// Returns a mutable reference to the element at the given index, or `None` if out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve
    ///
    /// # Returns
    ///
    /// `Some(&mut T)` if the index is valid, `None` otherwise
    #[inline_conditioned(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        Some(unsafe { self.data[index].assume_init_mut() })
    }

    /// Clears the vector, removing all elements.
    ///
    /// This operation does not deallocate memory; it simply resets the length to zero.
    #[inline_conditioned(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Elements for which the predicate returns `false` are removed. The order of
    /// retained elements is preserved.
    ///
    /// # Arguments
    ///
    /// * `f` - A predicate function that returns `true` for elements to keep
    #[inline_conditioned(always)]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let mut write_idx = 0;

        for read_idx in 0..self.len {
            unsafe {
                let item = self.data[read_idx].assume_init_read();
                if f(&item) {
                    if write_idx != read_idx {
                        self.data[write_idx].write(item);
                    }
                    write_idx += 1;
                }
            }
        }

        self.len = write_idx;
    }

    /// Removes multiple elements at the specified indices by swapping them with elements from the end.
    ///
    /// # Arguments
    ///
    /// * `indices` - Must be sorted in ascending order (duplicates allowed)
    ///
    /// # Panics
    ///
    /// Panics if indices are not sorted in ascending order.
    #[inline_conditioned(always)]
    pub fn swap_remove_indices(&mut self, indices: &[usize]) -> usize {
        let mut removed_count = 0;
        for i in (0..indices.len()).rev() {
            if i < indices.len() - 1 {
                assert!(
                    indices[i] <= indices[i + 1],
                    "indices must be sorted in ascending order"
                );
                if indices[i] == indices[i + 1] {
                    continue;
                }
            }
            if indices[i] < self.len {
                unsafe {
                    let _ = self.data[indices[i]].assume_init_read();
                    self.len -= 1;
                    if indices[i] < self.len {
                        self.data[indices[i]] =
                            MaybeUninit::new(self.data[self.len].assume_init_read());
                    }
                }
                removed_count += 1;
            }
        }
        removed_count
    }

    /// Returns `true` if any element satisfies the predicate.
    ///
    /// # Arguments
    ///
    /// * `f` - A predicate function to test each element
    ///
    /// # Returns
    ///
    /// `true` if at least one element satisfies the predicate, `false` otherwise
    pub fn any<F>(&self, f: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        for i in 0..self.len {
            let item = unsafe { self.data[i].assume_init_read() };
            if f(&item) {
                return true;
            }
        }
        false
    }

    /// Returns `true` if all elements satisfy the predicate.
    ///
    /// # Arguments
    ///
    /// * `f` - A predicate function to test each element
    ///
    /// # Returns
    ///
    /// `true` if all elements satisfy the predicate, `false` otherwise
    pub fn all<F>(&self, f: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        for i in 0..self.len {
            let item = unsafe { self.data[i].assume_init_read() };
            if !f(&item) {
                return false;
            }
        }
        true
    }

    /// Returns the first element that satisfies the predicate.
    ///
    /// # Arguments
    ///
    /// * `f` - A predicate function to test each element
    ///
    /// # Returns
    ///
    /// `Some(T)` containing the first matching element, or `None` if no match is found
    pub fn find<F>(&self, f: F) -> Option<T>
    where
        F: Fn(&T) -> bool,
    {
        for i in 0..self.len {
            let item = unsafe { self.data[i].assume_init_read() };
            if f(&item) {
                return Some(item);
            }
        }
        None
    }

    /// Applies a function to a shared reference to each initialized element.
    ///
    /// # Arguments
    ///
    /// * `f` - A function to apply to each element
    pub fn apply<F>(&self, f: F)
    where
        F: Fn(&T),
    {
        for i in 0..self.len {
            unsafe {
                f(self.data[i].assume_init_ref());
            }
        }
    }

    /// Applies a function to each element in the vector, allowing mutation.
    ///
    /// # Arguments
    ///
    /// * `f` - A function to apply to each element
    pub fn apply_mut<F>(&mut self, f: F)
    where
        F: Fn(&mut T),
    {
        for i in 0..self.len {
            unsafe {
                f(self.data[i].assume_init_mut());
            }
        }
    }

    /// Creates a new `HeaplessVec` by applying a function to each element.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that transforms elements from type `T` to type `U`
    ///
    /// # Returns
    ///
    /// A new `HeaplessVec` containing the transformed elements
    pub fn map<F, U: Copy + Sized + Debug>(&self, f: F) -> HeaplessVec<U, N>
    where
        F: Fn(&T) -> U,
    {
        let mut result = HeaplessVec::new();
        for i in 0..self.len {
            result.push(f(&unsafe { self.data[i].assume_init_read() }));
        }
        result
    }

    /// Fills this vector with elements from another vector, transferring from the end.
    ///
    /// Elements are transferred from the end of `other` to this vector until either
    /// this vector is full or `other` is empty. The transferred elements are removed
    /// from `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - The source vector to transfer elements from
    ///
    /// # Returns
    ///
    /// The number of elements transferred
    pub fn fill_from<const M: usize>(&mut self, other: &mut HeaplessVec<T, M>) -> usize {
        if self.len >= N {
            return 0;
        }

        // Calculate how many elements we can transfer
        let remaining_space = N - self.len;
        let transfer_count = std::cmp::min(other.len, remaining_space);

        unsafe {
            // Transfer elements from the END of other to self
            std::ptr::copy_nonoverlapping(
                other.data.as_ptr().add(other.len - transfer_count),
                self.data.as_mut_ptr().add(self.len),
                transfer_count,
            );
        }

        // Update lengths
        self.len += transfer_count;
        other.len -= transfer_count;

        transfer_count
    }

    /// Fills this vector with elements from a slice, taking from the end.
    ///
    /// Elements are copied from the end of the slice until this vector is full.
    ///
    /// # Arguments
    ///
    /// * `slice` - The source slice to copy elements from
    ///
    /// # Returns
    ///
    /// The number of elements copied
    #[inline_conditioned(always)]
    pub fn fill_from_slice(&mut self, slice: &[T]) -> usize {
        if self.len >= N {
            return 0;
        }

        let remaining_space = N - self.len;
        let transfer_count = std::cmp::min(slice.len(), remaining_space);

        unsafe {
            // Transfer elements from the END of slice to self
            std::ptr::copy_nonoverlapping(
                slice.as_ptr().add(slice.len() - transfer_count),
                self.data.as_mut_ptr().add(self.len) as *mut T,
                transfer_count,
            );
        }

        self.len += transfer_count;
        transfer_count
    }

    /// Fills this vector with elements from a fixed-size slice, taking from the end.
    ///
    /// Elements are copied from the end of the array until this vector is full.
    ///
    /// # Arguments
    ///
    /// * `arr` - The source array to copy elements from
    ///
    /// # Returns
    ///
    /// The new length of this vector
    #[inline_conditioned(always)]
    pub fn fill_from_array(&mut self, arr: &[T; N]) -> usize {
        let remaining_space = N - self.len;
        if remaining_space == 0 {
            return self.len;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                arr.as_ptr().add(self.len),
                self.as_mut_ptr().add(self.len),
                remaining_space,
            );
        }

        self.len = N;
        self.len
    }

    /// Replaces this vector's contents with all elements from a `Vec`.
    ///
    /// # Arguments
    ///
    /// * `vec` - The source vector to copy elements from
    ///
    /// # Panics
    ///
    /// Panics if the source vector has more elements than this vector's capacity.
    /// The destination is unchanged if this check fails.
    pub fn fill_from_vec(&mut self, vec: &Vec<T>) {
        assert!(
            vec.len() <= N,
            "HeaplessVec::fill_from_vec source exceeds capacity"
        );
        for (destination, &value) in self.data.iter_mut().zip(vec) {
            destination.write(value);
        }
        self.len = vec.len();
    }

    /// Converts this `HeaplessVec` to a standard `Vec`.
    ///
    /// # Returns
    ///
    /// A new `Vec<T>` containing copies of all elements
    pub fn to_vec(&self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len);
        if self.len > 0 {
            unsafe {
                // Copy all initialized elements at once using bulk memory copy
                std::ptr::copy_nonoverlapping(
                    self.data.as_ptr() as *const T,
                    vec.as_mut_ptr(),
                    self.len,
                );
                vec.set_len(self.len);
            }
        }
        vec
    }

    /// Returns a reference to the backing buffer as an array **only if** the vector is full.
    ///
    /// This avoids reading uninitialized elements (indices `self.len..N`).
    pub fn as_array(&self) -> Option<&[T; N]> {
        if self.len != N {
            return None;
        }
        // `len == N` implies all elements `0..N` were initialized.
        // We reuse the safe slice view and then convert it to an array reference.
        self.to_slice().try_into().ok()
    }

    /// Copies the contents into an array **only if** the vector is full.
    ///
    /// Returns `None` when `self.len != N`.
    pub fn to_array(&self) -> Option<[T; N]> {
        self.as_array().copied()
    }

    /// Returns a slice containing all elements in the vector.
    ///
    /// # Returns
    ///
    /// A slice view of the vector's elements
    pub fn to_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Returns a mutable slice containing all initialized elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    /// Returns an iterator over the elements of the vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding immutable references to each element
    pub fn into_iter(&self) -> std::slice::Iter<'_, T> {
        self.to_slice().iter()
    }

    /// Returns a mutable iterator over the elements of the vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to each element
    pub fn into_iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T: Copy, const N: usize> Index<usize> for HeaplessVec<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.to_slice()[index]
    }
}

/// Vec Pool
pub struct VecPool<T: Copy + Sized + Debug, const N: usize, const M: usize> {
    pool: [SpinLock<HeaplessVec<T, N>>; M],
}

impl<T: Copy + Sized + Debug, const N: usize, const M: usize> Default for VecPool<T, N, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Sized + Debug, const N: usize, const M: usize> VecPool<T, N, M> {
    /// Creates a new pool of `M` vectors, each with capacity `N`.
    ///
    /// # Returns
    ///
    /// A new `VecPool` with all vectors initialized to empty state
    pub const fn new() -> Self {
        Self {
            pool: [const { SpinLock::new(HeaplessVec::new()) }; M],
        }
    }

    /// Acquires a lock on one of the vectors in the pool.
    ///
    /// This method tries each vector in the pool in sequence until it finds one
    /// that is available. It will spin until a vector becomes available.
    ///
    /// # Returns
    ///
    /// A guard providing exclusive access to one of the pool's vectors
    pub fn get_lock(&self) -> SpinLockGuard<'_, HeaplessVec<T, N>> {
        loop {
            for lock in &self.pool {
                if let Some(guard) = lock.try_lock() {
                    return guard;
                }
            }
            spin_loop();
        }
    }
}

#[macro_export]
macro_rules! rep1_at {
    ($start:expr, $i:ident, $b:block) => {{
        const $i: usize = $start;
        $b
    }};
}

#[macro_export]
macro_rules! rep2_at {
    ($start:expr, $i:ident, $b:block) => {{
        $crate::rep1_at!($start + 0, $i, $b);
        $crate::rep1_at!($start + 1, $i, $b);
    }};
}

#[macro_export]
macro_rules! rep4_at {
    ($start:expr, $i:ident, $b:block) => {{
        $crate::rep2_at!(($start + 0), $i, $b);
        $crate::rep2_at!(($start + 2), $i, $b);
    }};
}

#[macro_export]
macro_rules! rep8_at {
    ($start:expr, $i:ident, $b:block) => {{
        $crate::rep4_at!(($start + 0), $i, $b);
        $crate::rep4_at!(($start + 4), $i, $b);
    }};
}

#[macro_export]
macro_rules! rep16_at {
    ($start:expr, $i:ident, $b:block) => {{
        $crate::rep8_at!(($start + 0), $i, $b);
        $crate::rep8_at!(($start + 8), $i, $b);
    }};
}

#[macro_export]
macro_rules! rep32_at {
    ($start:expr, $i:ident, $b:block) => {{
        $crate::rep16_at!(($start + 0), $i, $b);
        $crate::rep16_at!(($start + 16), $i, $b);
    }};
}

#[macro_export]
macro_rules! repeat_exact_idx {
    (0,  $i:ident, $b:block) => {};
    (1,  $i:ident, $b:block) => {
        $crate::rep1_at!(0, $i, $b)
    };
    (2,  $i:ident, $b:block) => {
        $crate::rep2_at!(0, $i, $b)
    };
    (3,  $i:ident, $b:block) => {{
        $crate::rep2_at!(0, $i, $b);
        $crate::rep1_at!(2, $i, $b);
    }};
    (4,  $i:ident, $b:block) => {
        $crate::rep4_at!(0, $i, $b)
    };
    (5,  $i:ident, $b:block) => {{
        $crate::rep4_at!(0, $i, $b);
        $crate::rep1_at!(4, $i, $b);
    }};
    (6,  $i:ident, $b:block) => {{
        $crate::rep4_at!(0, $i, $b);
        $crate::rep2_at!(4, $i, $b);
    }};
    (7,  $i:ident, $b:block) => {{
        $crate::rep4_at!(0, $i, $b);
        $crate::rep2_at!(4, $i, $b);
        $crate::rep1_at!(6, $i, $b);
    }};
    (8,  $i:ident, $b:block) => {
        $crate::rep8_at!(0, $i, $b)
    };
    (9,  $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep1_at!(8, $i, $b);
    }};
    (10, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep2_at!(8, $i, $b);
    }};
    (11, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep2_at!(8, $i, $b);
        $crate::rep1_at!(10, $i, $b);
    }};
    (12, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep4_at!(8, $i, $b);
    }};
    (13, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep4_at!(8, $i, $b);
        $crate::rep1_at!(12, $i, $b);
    }};
    (14, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep4_at!(8, $i, $b);
        $crate::rep2_at!(12, $i, $b);
    }};
    (15, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep4_at!(8, $i, $b);
        $crate::rep2_at!(12, $i, $b);
        $crate::rep1_at!(14, $i, $b);
    }};
    (16, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
    }};
    (17, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep1_at!(16, $i, $b);
    }};
    (18, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep2_at!(16, $i, $b);
    }};
    (19, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep2_at!(16, $i, $b);
        $crate::rep1_at!(18, $i, $b);
    }};
    (20, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep4_at!(16, $i, $b);
    }};
    (21, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep4_at!(16, $i, $b);
        $crate::rep1_at!(20, $i, $b);
    }};
    (22, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep4_at!(16, $i, $b);
        $crate::rep2_at!(20, $i, $b);
    }};
    (23, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep4_at!(16, $i, $b);
        $crate::rep2_at!(20, $i, $b);
        $crate::rep1_at!(22, $i, $b);
    }};
    (24, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
    }};
    (25, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep1_at!(24, $i, $b);
    }};
    (26, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep2_at!(24, $i, $b);
    }};
    (27, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep2_at!(24, $i, $b);
        $crate::rep1_at!(26, $i, $b);
    }};
    (28, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
    }};
    (29, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
        $crate::rep1_at!(28, $i, $b);
    }};
    (30, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
        $crate::rep2_at!(28, $i, $b);
    }};
    (31, $i:ident, $b:block) => {{
        $crate::rep16_at!(0, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
        $crate::rep2_at!(28, $i, $b);
        $crate::rep1_at!(30, $i, $b);
    }};
    (32, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
    }};
    (33, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep1_at!(32, $i, $b);
    }};
    (34, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep2_at!(32, $i, $b);
    }};
    (35, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep2_at!(32, $i, $b);
        $crate::rep1_at!(34, $i, $b);
    }};
    (36, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep4_at!(32, $i, $b);
    }};
    (37, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep4_at!(32, $i, $b);
        $crate::rep1_at!(36, $i, $b);
    }};
    (38, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep4_at!(32, $i, $b);
        $crate::rep2_at!(36, $i, $b);
    }};
    (39, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep4_at!(32, $i, $b);
        $crate::rep2_at!(36, $i, $b);
        $crate::rep1_at!(38, $i, $b);
    }};
    (40, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
    }};
    (41, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep1_at!(40, $i, $b);
    }};
    (42, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep2_at!(40, $i, $b);
    }};
    (43, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep2_at!(40, $i, $b);
        $crate::rep1_at!(42, $i, $b);
    }};
    (44, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep4_at!(40, $i, $b);
    }};
    (45, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep4_at!(40, $i, $b);
        $crate::rep1_at!(44, $i, $b);
    }};
    (46, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep4_at!(40, $i, $b);
        $crate::rep2_at!(44, $i, $b);
    }};
    (47, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep8_at!(32, $i, $b);
        $crate::rep4_at!(40, $i, $b);
        $crate::rep2_at!(44, $i, $b);
        $crate::rep1_at!(46, $i, $b);
    }};
    (48, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
    }};
    (49, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep1_at!(48, $i, $b);
    }};
    (50, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep2_at!(48, $i, $b);
    }};
    (51, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep2_at!(48, $i, $b);
        $crate::rep1_at!(50, $i, $b);
    }};
    (52, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep4_at!(48, $i, $b);
    }};
    (53, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep4_at!(48, $i, $b);
        $crate::rep1_at!(52, $i, $b);
    }};
    (54, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep4_at!(48, $i, $b);
        $crate::rep2_at!(52, $i, $b);
    }};
    (55, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep4_at!(48, $i, $b);
        $crate::rep2_at!(52, $i, $b);
        $crate::rep1_at!(54, $i, $b);
    }};
    (56, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
    }};
    (57, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep1_at!(56, $i, $b);
    }};
    (58, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep2_at!(56, $i, $b);
    }};
    (59, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep2_at!(56, $i, $b);
        $crate::rep1_at!(58, $i, $b);
    }};
    (60, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep4_at!(56, $i, $b);
    }};
    (61, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep4_at!(56, $i, $b);
        $crate::rep1_at!(60, $i, $b);
    }};
    (62, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep4_at!(56, $i, $b);
        $crate::rep2_at!(60, $i, $b);
    }};
    (63, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep4_at!(56, $i, $b);
        $crate::rep2_at!(60, $i, $b);
        $crate::rep1_at!(62, $i, $b);
    }};
    (64, $i:ident, $b:block) => {{
        $crate::rep32_at!(0, $i, $b);
        $crate::rep16_at!(32, $i, $b);
        $crate::rep8_at!(48, $i, $b);
        $crate::rep4_at!(56, $i, $b);
        $crate::rep2_at!(60, $i, $b);
        $crate::rep2_at!(62, $i, $b);
    }};
}

/// Unrolls a compile-time count of up to 64 iterations with a constant index.
#[macro_export]
macro_rules! repeat_idx_unroll {
    ($N:expr, $i:ident, $b:block) => {{
        assert!($N <= 64, "unroll count exceeds 64");
        $crate::repeat_exact_idx!(64, $i, {
            if const { $i < $N } $b
        });
    }};
}

/// Returns the bit index of the `k`-th set bit in `x` (0-indexed).
///
/// The ranking is from least-significant bit to most-significant bit:
/// `k = 0` selects the lowest set bit, `k = 1` the next one, and so on.
///
/// This implementation is branchless and performs a binary partition over
/// 32/16/8/4/2/1-bit chunks, using `count_ones()` to decide whether the target
/// set bit is in the lower chunk of the current window.
///
/// # Preconditions
///
/// * `k < x.count_ones()`
///
/// Violating the precondition yields an unspecified index.
#[inline(always)]
fn select_kth_u64(mut x: u64, mut k: u32) -> u32 {
    let mut i = 0u32;

    let c = (x & 0xFFFF_FFFF).count_ones();
    let g = ((k.wrapping_sub(c) >> 31) ^ 1) & 1;
    i += g * 32;
    k -= g * c;
    x >>= g * 32;
    let c = (x & 0x0000_FFFF).count_ones();
    let g = ((k.wrapping_sub(c) >> 31) ^ 1) & 1;
    i += g * 16;
    k -= g * c;
    x >>= g * 16;
    let c = (x & 0x0000_00FF).count_ones();
    let g = ((k.wrapping_sub(c) >> 31) ^ 1) & 1;
    i += g * 8;
    k -= g * c;
    x >>= g * 8;
    let c = (x & 0x0000_000F).count_ones();
    let g = ((k.wrapping_sub(c) >> 31) ^ 1) & 1;
    i += g * 4;
    k -= g * c;
    x >>= g * 4;
    let c = (x & 0x0000_0003).count_ones();
    let g = ((k.wrapping_sub(c) >> 31) ^ 1) & 1;
    i += g * 2;
    k -= g * c;
    x >>= g * 2;
    let c = (x & 0x0000_0001).count_ones();
    let g = ((k.wrapping_sub(c) >> 31) ^ 1) & 1;
    i + g
}

#[inline(always)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "A nonzero u64 mask has at most 64 set bits; the remainder is below that count."
)]
pub fn choose_set_bit_u64<R: rand::RngExt + ?Sized>(mask: u64, rng: &mut R) -> Option<u32> {
    (mask != 0).then(|| {
        let n = mask.count_ones(); // 1..=64, fits in u32
        let k = (rng.next_u64() % (n as u64)) as u32; // bias acceptable
        select_kth_u64(mask, k)
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixture casts use explicitly bounded board dimensions and case counts, or deliberately extract test bits."
)]
mod tests {
    use rand::RngExt;

    use super::*;

    mod heapless_fill_from_vec {
        use super::*;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        #[test]
        fn replaces_contents_and_shrinks_or_clears_the_initialized_prefix() {
            let mut values = HeaplessVec::<i32, 4>::new();
            values.push(99);
            for source in [vec![1, 2, 3, 4], vec![5, 6], vec![], vec![7]] {
                values.fill_from_vec(&source);
                assert_eq!(values.len(), source.len());
                assert_eq!(values.to_slice(), source.as_slice());
                assert_eq!(values.to_vec(), source);
            }
            values.push(8);
            assert_eq!(values.to_slice(), &[7, 8]);
        }

        #[test]
        fn oversized_input_preserves_contents_and_allows_subsequent_use() {
            let mut values = HeaplessVec::<i32, 3>::new();
            values.fill_from_vec(&vec![1, 2]);
            let source = vec![10, 20, 30, 40];
            let result = catch_unwind(AssertUnwindSafe(|| values.fill_from_vec(&source)));
            assert!(result.is_err());
            assert_eq!(values.len(), 2);
            assert_eq!(values.to_slice(), &[1, 2]);
            values.push(3);
            assert_eq!(values.to_array(), Some([1, 2, 3]));
            values.fill_from_vec(&vec![4]);
            assert_eq!(values.to_slice(), &[4]);
        }

        #[test]
        fn zero_capacity_accepts_empty_and_rejects_nonempty_input() {
            let mut values = HeaplessVec::<u32, 0>::new();
            values.fill_from_vec(&Vec::new());
            assert_eq!(values.to_array(), Some([]));
            let source = vec![1];
            assert!(catch_unwind(AssertUnwindSafe(|| values.fill_from_vec(&source))).is_err());
            assert!(values.is_empty());
            assert_eq!(values.to_array(), Some([]));
        }

        #[test]
        fn zero_sized_values_still_obey_logical_capacity() {
            let mut values = HeaplessVec::<(), 2>::new();
            values.fill_from_vec(&vec![(); 2]);
            assert_eq!(values.len(), 2);
            let source = vec![(); 3];
            assert!(catch_unwind(AssertUnwindSafe(|| values.fill_from_vec(&source))).is_err());
            assert_eq!(values.to_array(), Some([(), ()]));
            values.fill_from_vec(&vec![()]);
            assert_eq!(values.len(), 1);
            values.fill_from_vec(&Vec::new());
            assert!(values.is_empty());
        }

        #[test]
        fn copies_borrowed_values_without_extending_their_lifetimes() {
            let first = String::from("first");
            let second = String::from("second");
            let source = vec![first.as_str(), second.as_str()];
            let mut values = HeaplessVec::<&str, 2>::new();
            values.fill_from_vec(&source);
            drop(source);
            assert_eq!(values.to_array(), Some([first.as_str(), second.as_str()]));
        }
    }

    mod fixed_bin_min_heap {
        use super::*;
        use rand::{SeedableRng, rngs::SmallRng};
        use std::panic::{AssertUnwindSafe, catch_unwind};

        fn assert_matches<const N: usize>(heap: &FixedBinMinHeap<i32, N>, expected: &[i32]) {
            assert_eq!(heap.len(), expected.len());
            assert_eq!(heap.is_empty(), expected.is_empty());
            assert_eq!(heap.peek_min(), expected.first().copied());
            let mut actual = heap.as_slice().to_vec();
            actual.sort_unstable();
            assert_eq!(actual, expected);
            for (index, &value) in heap.as_slice().iter().enumerate().skip(1) {
                assert!(heap.as_slice()[(index - 1) / 2] <= value);
            }
        }

        #[test]
        fn zero_capacity_rejects_candidates_and_preserves_empty_state() {
            let mut heap = FixedBinMinHeap::<i32, 0>::new();
            for value in [i32::MIN, -1, 0, 1, i32::MAX] {
                assert!(!heap.push_if_better_min_heap(value));
                assert_matches(&heap, &[]);
                assert!(heap.as_mut_slice().is_empty());
                assert_eq!(heap.pop_min(), None);
                heap.clear();
            }
            assert!(catch_unwind(AssertUnwindSafe(|| heap.push(7))).is_err());
            assert_matches(&heap, &[]);
        }

        fn assert_empty_replacement_panics<const N: usize>(heap: &mut FixedBinMinHeap<i32, N>) {
            assert!(heap.is_empty());
            assert!(catch_unwind(AssertUnwindSafe(|| heap.replace_min(7))).is_err());
            assert_matches(heap, &[]);
        }

        fn check_empty_replacement<const N: usize>() {
            let mut heap = FixedBinMinHeap::<i32, N>::new();
            assert_empty_replacement_panics(&mut heap);
            if N != 0 {
                heap.push(3);
                heap.clear();
                assert_empty_replacement_panics(&mut heap);
                heap.push(5);
                assert_eq!(heap.pop_min(), Some(5));
                assert_empty_replacement_panics(&mut heap);
            }
        }

        #[test]
        fn replacement_rejects_fresh_cleared_and_drained_empty_heaps() {
            check_empty_replacement::<0>();
            check_empty_replacement::<1>();
            check_empty_replacement::<8>();
        }

        #[test]
        fn singleton_accepts_only_better_candidates_when_full() {
            let mut heap = FixedBinMinHeap::<i32, 1>::new();
            assert!(heap.push_if_better_min_heap(3));
            assert!(!heap.push_if_better_min_heap(2));
            assert!(!heap.push_if_better_min_heap(3));
            assert!(heap.push_if_better_min_heap(4));
            assert_matches(&heap, &[4]);
            heap.replace_min(-1);
            assert_matches(&heap, &[-1]);
        }

        fn check_sorted_reference<const N: usize>() {
            let mut rng = SmallRng::seed_from_u64(0x007e_715a_71a5);
            let mut inputs = vec![i32::MIN, i32::MAX, 0, 0, -1, 1, i32::MIN, i32::MAX];
            inputs.extend((0..128).map(|_| rng.random_range(-32..=32)));
            let mut heap = FixedBinMinHeap::<i32, N>::new();
            let mut expected = Vec::new();
            for (step, value) in inputs.into_iter().enumerate() {
                if step == 96 {
                    heap.clear();
                    expected.clear();
                }
                let accepted = N != 0 && (expected.len() < N || value > expected[0]);
                assert_eq!(heap.push_if_better_min_heap(value), accepted);
                expected.push(value);
                expected.sort_unstable();
                if expected.len() > N {
                    expected.remove(0);
                }
                assert_matches(&heap, &expected);

                if step.is_multiple_of(17) && !expected.is_empty() {
                    let replacement = value.saturating_add(1);
                    heap.replace_min(replacement);
                    expected.remove(0);
                    expected.push(replacement);
                    expected.sort_unstable();
                }
                if step.is_multiple_of(11) {
                    assert_eq!(heap.pop_min(), expected.first().copied());
                    if !expected.is_empty() {
                        expected.remove(0);
                    }
                }
                assert_matches(&heap, &expected);
            }
            for value in expected {
                assert_eq!(heap.pop_min(), Some(value));
            }
            assert_eq!(heap.pop_min(), None);
            assert!(heap.is_empty());
        }

        #[test]
        fn mixed_operations_match_a_sorted_reference() {
            check_sorted_reference::<0>();
            check_sorted_reference::<1>();
            check_sorted_reference::<2>();
            check_sorted_reference::<3>();
            check_sorted_reference::<7>();
            check_sorted_reference::<16>();
            check_sorted_reference::<64>();
        }

        #[test]
        fn zero_sized_values_respect_capacity() {
            let mut heap = FixedBinMinHeap::<(), 2>::new();
            assert!(heap.push_if_better_min_heap(()));
            assert!(heap.push_if_better_min_heap(()));
            assert!(!heap.push_if_better_min_heap(()));
            heap.replace_min(());
            assert_eq!(heap.len(), 2);
            assert_eq!(heap.pop_min(), Some(()));
            assert_eq!(heap.pop_min(), Some(()));
            assert_eq!(heap.pop_min(), None);
        }
    }

    mod spin_lock {
        use super::*;
        use std::cell::Cell;
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::sync::Barrier;
        use std::thread;

        #[test]
        fn try_lock_excludes_other_guards_and_releases_on_drop() {
            let lock = SpinLock::new(Cell::new(7usize));
            let mut guard = lock.try_lock().expect("unlocked");
            assert!(lock.try_lock().is_none());
            *guard = Cell::new(11);
            drop(guard);

            let guard = lock.try_lock().expect("released");
            assert_eq!(guard.get(), 11);
        }

        #[test]
        fn held_lock_is_unavailable_to_another_thread() {
            let lock = SpinLock::new(Cell::new(7usize));
            let guard = lock.lock();
            thread::scope(|scope| {
                scope
                    .spawn(|| assert!(lock.try_lock().is_none()))
                    .join()
                    .expect("contender");
            });
            guard.set(11);
            drop(guard);

            thread::scope(|scope| {
                scope
                    .spawn(|| assert_eq!(lock.try_lock().expect("released").get(), 11))
                    .join()
                    .expect("next owner");
            });
        }

        #[test]
        fn contending_owners_publish_updates_to_send_but_not_sync_values() {
            const WORKERS: usize = 4;
            const UPDATES: usize = 16;
            // Cell is Send but not Sync: the lock must supply synchronization.
            let lock = SpinLock::new(Cell::new((0usize, !0usize)));
            let start = Barrier::new(WORKERS);
            thread::scope(|scope| {
                for worker in 0..WORKERS {
                    let lock = &lock;
                    let start = &start;
                    scope.spawn(move || {
                        start.wait();
                        for _ in 0..UPDATES {
                            let guard = if worker.is_multiple_of(2) {
                                lock.lock()
                            } else {
                                loop {
                                    if let Some(guard) = lock.try_lock() {
                                        break guard;
                                    }
                                    thread::yield_now();
                                }
                            };
                            let (count, inverse) = guard.get();
                            assert_eq!(inverse, !count);
                            // Leave an inconsistent pair while holding the lock:
                            // another owner must not observe this intermediate state.
                            let next = count + 1;
                            guard.set((next, inverse));
                            thread::yield_now();
                            guard.set((next, !next));
                        }
                    });
                }
            });
            assert_eq!(lock.lock().get(), (WORKERS * UPDATES, !(WORKERS * UPDATES)));
        }

        #[test]
        #[allow(clippy::panic, reason = "Exercise guard cleanup during unwinding.")]
        fn unwinding_releases_the_lock_without_poisoning() {
            let lock = SpinLock::new(7usize);
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut guard = lock.lock();
                *guard = 11;
                panic!("intentional unwind while holding the guard");
            }));
            assert!(result.is_err());
            assert_eq!(*lock.try_lock().expect("released during unwind"), 11);
        }

        #[test]
        fn guard_can_be_transferred_and_released_on_another_thread() {
            let lock = SpinLock::new(7usize);
            let mut guard = lock.lock();
            thread::scope(|scope| {
                scope
                    .spawn(move || *guard = 11)
                    .join()
                    .expect("guard owner");
            });
            assert_eq!(*lock.try_lock().expect("released by other thread"), 11);
        }
    }

    #[test]
    fn pool_initializes_every_lock_and_retains_each_vector() {
        let pool: VecPool<u32, 2, 3> = const { VecPool::new() };
        for (index, lock) in pool.pool.iter().enumerate() {
            let mut values = lock.lock();
            assert!(values.is_empty());
            values.push(u32::try_from(index).expect("three pool entries fit in u32"));
        }
        for (index, lock) in pool.pool.iter().enumerate() {
            assert_eq!(
                lock.lock().to_slice(),
                &[u32::try_from(index).expect("small index")]
            );
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn indexing_spare_capacity_panics() {
        let mut values = HeaplessVec::<u32, 4>::new();
        values.push(7);
        let _value = values[1];
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn pushing_to_full_vector_panics() {
        let mut values = HeaplessVec::<u32, 1>::new();
        values.push(7);
        values.push(8);
    }

    #[test]
    #[should_panic(expected = "FixedBinMinHeap::push called when full")]
    fn pushing_to_full_heap_panics() {
        let mut heap = FixedBinMinHeap::<u32, 1>::new();
        heap.push(7);
        heap.push(8);
    }

    mod bit_chunks {
        use super::*;

        #[test]
        fn chunk_swaps_match_individual_bit_moves_for_every_supported_width() {
            fn check<const BITS: u32>()
            where
                [(); (BITS > 0) as usize - 1]:,
                [(); (BITS < u64::BITS) as usize - 1]:,
            {
                for original in [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210] {
                    for first in 0..u64::BITS / BITS {
                        for second in 0..u64::BITS / BITS {
                            let mut expected = 0u64;
                            for bit in 0..u64::BITS {
                                let source = if (first * BITS..(first + 1) * BITS).contains(&bit) {
                                    second * BITS + bit - first * BITS
                                } else if (second * BITS..(second + 1) * BITS).contains(&bit) {
                                    first * BITS + bit - second * BITS
                                } else {
                                    bit
                                };
                                expected |= ((original >> source) & 1) << bit;
                            }
                            let mut actual = original;
                            swap_bit_chunks::<BITS>(&mut actual, first, second);
                            assert_eq!(
                                actual, expected,
                                "bits={BITS}, first={first}, second={second}"
                            );
                        }
                    }
                }
            }
            // Every width needs its own const-generic instantiation.
            macro_rules! check_widths { ($($bits:literal),*) => { $(check::<$bits>();)* }; }
            check_widths!(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
            );
        }

        #[test]
        fn chunk_swaps_exchange_only_the_two_requested_slots() {
            const BITS: u32 = 4;
            const MASK: u64 = (1 << BITS) - 1;
            let values = [0u64, 1, 3, 7, 9, 12, 15];
            let packed = values.iter().enumerate().fold(0, |word, (index, value)| {
                word | (value << (index as u32 * BITS))
            });
            for first in 0..values.len() {
                for second in 0..values.len() {
                    let mut expected = values;
                    expected.swap(first, second);
                    let mut actual = packed;
                    swap_bit_chunks::<BITS>(&mut actual, first as u32, second as u32);
                    for (index, value) in expected.into_iter().enumerate() {
                        assert_eq!(
                            (actual >> (index as u32 * BITS)) & MASK,
                            value,
                            "first={first}, second={second}, slot={index}"
                        );
                    }
                }
            }
        }

        #[test]
        fn chunk_width_bounds_work_in_const_context() {
            const HIGH_BIT: u64 = {
                let mut x = 1;
                swap_bit_chunks::<1>(&mut x, 0, 63);
                x
            };
            const UNCHANGED: u64 = {
                let mut x = 0x8000_0000_0000_0321;
                swap_bit_chunks::<63>(&mut x, 0, 0);
                x
            };
            assert_eq!(HIGH_BIT, 0x8000_0000_0000_0000);
            assert_eq!(UNCHANGED, 0x8000_0000_0000_0321);
        }
    }

    mod unrolling {
        const fn indices<const N: usize>() -> [usize; N] {
            let mut result = [usize::MAX; N];
            let mut position = 0;
            crate::repeat_idx_unroll!(N, I, {
                const INDEX: usize = I;
                result[position] = INDEX;
                position += 1;
            });
            assert!(position == N, "iteration count must match N");
            result
        }

        #[test]
        fn every_count_preserves_constant_indices_and_order() {
            fn check<const N: usize>() {
                assert_eq!(indices::<N>().as_slice(), &(0..N).collect::<Vec<_>>());
            }
            macro_rules! check_counts {
                ($($count:literal),* $(,)?) => { $(check::<$count>();)* };
            }
            check_counts!(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            );
        }

        #[test]
        fn missing_count_can_be_evaluated_in_a_const_context() {
            const INDICES: [usize; 24] = indices::<24>();
            assert_eq!(INDICES.as_slice(), &(0..24).collect::<Vec<_>>());
        }

        #[test]
        #[should_panic(expected = "unroll count exceeds 64")]
        fn counts_above_capacity_are_rejected() {
            let _ = indices::<65>();
        }

        #[test]
        fn large_transpose_keeps_its_scatter_fallback() {
            let mut bytes = [0u8; 17];
            bytes[8] = 0b10; // Cell (row 32, column 1) is bit 65.
            let original = bytes;
            super::super::transpose_bits::<65, 2, 17>(&mut bytes);
            let mut expected = [0u8; 17];
            expected[12] = 0b10; // Transposed cell (row 1, column 32) is bit 97.
            assert_eq!(bytes, expected);
            super::super::transpose_bits::<2, 65, 17>(&mut bytes);
            assert_eq!(bytes, original);
        }
    }

    #[test]
    fn test_select_kth_and_choose_set_bit() {
        // Reference implementation: kth set bit (0-indexed) scanning from LSB->MSB.
        fn select_kth_ref(x: u64, k: u32) -> u32 {
            let mut seen = 0u32;
            for i in 0..64u32 {
                if ((x >> i) & 1) == 1 {
                    if seen == k {
                        return i;
                    }
                    seen += 1;
                }
            }
            assert!(
                k < x.count_ones(),
                "k out of range for x (k={k}, popcnt={})",
                x.count_ones()
            );
            0
        }

        // Basic correctness: exact positions across a variety of masks.
        let masks = [
            1u64,
            1u64 << 31,
            1u64 << 63,
            0b1011u64,                // bits at 0,1,3
            0x00FF_0000u64,           // contiguous block
            0x8000_0001u64,           // ends
            0b1010_0101_0001_0010u64, // mixed
            0xFFFF_FFFF_FFFF_FFFFu64, // all bits set
        ];
        for &m in &masks {
            let cnt = m.count_ones();
            assert!(cnt > 0);
            for k in 0..cnt {
                let got = select_kth_u64(m, k);
                let exp = select_kth_ref(m, k);
                assert_eq!(got, exp, "mask={m:#034b}, k={k}");
                assert_eq!(
                    ((m >> got) & 1),
                    1,
                    "returned bit not set: mask={m:#034b}, got={got}"
                );
            }
        }

        // final fuzz
        let count = 10_000;
        let mut rng = rand::rng();
        for _ in 0..count {
            let mask: u64 = rng.random();
            let k = rng.random::<u32>() % mask.count_ones();
            let got = select_kth_u64(mask, k);
            let expected = select_kth_ref(mask, k);
            assert_eq!(
                got, expected,
                "mask={mask:#066b}, got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_new() {
        let vec: HeaplessVec<i32, 5> = HeaplessVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());
        assert_eq!(HeaplessVec::<i32, 5>::capacity(), 5);
    }

    #[test]
    fn test_capacity() {
        assert_eq!(HeaplessVec::<i32, 0>::capacity(), 0);
        assert_eq!(HeaplessVec::<i32, 1>::capacity(), 1);
        assert_eq!(HeaplessVec::<i32, 10>::capacity(), 10);
        assert_eq!(HeaplessVec::<i32, 100>::capacity(), 100);
    }

    #[test]
    fn test_push_and_len() {
        let mut vec: HeaplessVec<i32, 5> = HeaplessVec::new();

        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());

        vec.push(1);
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.to_vec(), vec![1]);

        vec.push(2);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.to_vec(), vec![1, 2]);

        vec.push(3);
        vec.push(4);
        vec.push(5);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_clear() {
        let mut vec: HeaplessVec<i32, 5> = HeaplessVec::new();

        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.to_vec(), vec![1, 2, 3]);

        vec.clear();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());

        // Should be able to push again after clear
        vec.push(10);
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.to_vec(), vec![10]);
    }

    #[test]
    fn test_retain_all() {
        let mut vec: HeaplessVec<i32, 5> = HeaplessVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        vec.retain(|_| true);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_retain_none() {
        let mut vec: HeaplessVec<i32, 5> = HeaplessVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        vec.retain(|_| false);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_retain_even_numbers() {
        let mut vec: HeaplessVec<i32, 10> = HeaplessVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);
        vec.push(5);
        vec.push(6);

        vec.retain(|&x| x % 2 == 0);
        assert_eq!(vec.len(), 3); // Should keep 2, 4, 6
        assert_eq!(vec.to_vec(), vec![2, 4, 6]);
    }

    #[test]
    fn test_retain_empty_vec() {
        let mut vec: HeaplessVec<i32, 5> = HeaplessVec::new();
        vec.retain(|_| true);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_fill_from_empty_to_empty() {
        let mut dest: HeaplessVec<i32, 5> = HeaplessVec::new();
        let mut src: HeaplessVec<i32, 5> = HeaplessVec::new();

        dest.fill_from(&mut src);

        assert_eq!(dest.len(), 0);
        assert_eq!(dest.to_vec(), Vec::<i32>::new());
        assert_eq!(src.len(), 0);
        assert_eq!(src.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_fill_from_nonempty_to_empty() {
        let mut dest: HeaplessVec<i32, 5> = HeaplessVec::new();
        let mut src: HeaplessVec<i32, 5> = HeaplessVec::new();

        src.push(1);
        src.push(2);
        src.push(3);

        dest.fill_from(&mut src);

        assert_eq!(dest.len(), 3);
        assert_eq!(dest.to_vec(), vec![1, 2, 3]); // Should transfer from END of src
        assert_eq!(src.len(), 0);
        assert_eq!(src.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_fill_from_partial_transfer() {
        let mut dest: HeaplessVec<i32, 5> = HeaplessVec::new();
        let mut src: HeaplessVec<i32, 5> = HeaplessVec::new();

        // Fill dest partially
        dest.push(1);
        dest.push(2);

        // Fill src
        src.push(10);
        src.push(20);
        src.push(30);
        src.push(40);

        dest.fill_from(&mut src);

        // dest had 2, can take 3 more, src had 4
        // Should transfer from END of src: 20, 30, 40
        assert_eq!(dest.len(), 5); // 2 + 3 = 5 (full)
        assert_eq!(dest.to_vec(), vec![1, 2, 20, 30, 40]);
        assert_eq!(src.len(), 1); // 4 - 3 = 1 remaining
        assert_eq!(src.to_vec(), vec![10]); // Only first element remains
    }

    #[test]
    fn test_fill_from_full_destination() {
        let mut dest: HeaplessVec<i32, 3> = HeaplessVec::new();
        let mut src: HeaplessVec<i32, 3> = HeaplessVec::new();

        // Fill dest completely
        dest.push(1);
        dest.push(2);
        dest.push(3);

        // Fill src
        src.push(10);
        src.push(20);

        dest.fill_from(&mut src);

        // dest is full, no transfer should happen
        assert_eq!(dest.len(), 3);
        assert_eq!(dest.to_vec(), vec![1, 2, 3]);
        assert_eq!(src.len(), 2);
        assert_eq!(src.to_vec(), vec![10, 20]);
    }

    #[test]
    fn test_fill_from_exact_fit() {
        let mut dest: HeaplessVec<i32, 5> = HeaplessVec::new();
        let mut src: HeaplessVec<i32, 5> = HeaplessVec::new();

        // Fill dest partially
        dest.push(1);
        dest.push(2);

        // Fill src with exactly what fits
        src.push(10);
        src.push(20);
        src.push(30);

        dest.fill_from(&mut src);

        assert_eq!(dest.len(), 5); // 2 + 3 = 5
        assert_eq!(dest.to_vec(), vec![1, 2, 10, 20, 30]); // All elements transferred from END
        assert_eq!(src.len(), 0); // All transferred
        assert_eq!(src.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_multiple_operations() {
        let mut vec: HeaplessVec<i32, 10> = HeaplessVec::new();

        // Push some elements
        for i in 1..=5 {
            vec.push(i);
        }
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.to_vec(), vec![1, 2, 3, 4, 5]);

        // Retain odd numbers
        vec.retain(|&x| x % 2 == 1);
        assert_eq!(vec.len(), 3); // 1, 3, 5
        assert_eq!(vec.to_vec(), vec![1, 3, 5]);

        // Clear and start over
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());

        // Push again
        vec.push(100);
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.to_vec(), vec![100]);
    }

    #[test]
    fn test_different_types() {
        let mut str_vec: HeaplessVec<&str, 3> = HeaplessVec::new();
        str_vec.push("hello");
        str_vec.push("world");
        assert_eq!(str_vec.len(), 2);
        assert_eq!(str_vec.to_vec(), vec!["hello", "world"]);

        let mut bool_vec: HeaplessVec<bool, 2> = HeaplessVec::new();
        bool_vec.push(true);
        bool_vec.push(false);
        assert_eq!(bool_vec.len(), 2);
        assert_eq!(bool_vec.to_vec(), vec![true, false]);

        bool_vec.retain(|&x| x); // Keep only true values
        assert_eq!(bool_vec.len(), 1);
        assert_eq!(bool_vec.to_vec(), vec![true]);
    }

    #[test]
    fn test_zero_capacity() {
        let vec: HeaplessVec<i32, 0> = HeaplessVec::new();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());
        assert_eq!(HeaplessVec::<i32, 0>::capacity(), 0);

        // fill_from with zero capacity destination
        let src: HeaplessVec<i32, 0> = HeaplessVec::new();

        // Can't push since capacity is 0
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.to_vec(), Vec::<i32>::new());
        assert_eq!(src.len(), 0);
        assert_eq!(src.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_to_vec_method() {
        let mut vec: HeaplessVec<&str, 3> = HeaplessVec::new();

        // Test empty to_vec
        assert_eq!(vec.to_vec(), Vec::<String>::new());

        // Test single element
        vec.push("test");
        assert_eq!(vec.to_vec(), vec!["test"]);

        // Test multiple elements
        vec.push("hello");
        vec.push("world");
        assert_eq!(
            vec.to_vec(),
            vec!["test".to_string(), "hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn test_fill_from_different_sizes() {
        // Test filling a smaller vec from a larger vec
        let mut small_dest: HeaplessVec<i32, 3> = HeaplessVec::new();
        let mut large_src: HeaplessVec<i32, 7> = HeaplessVec::new();

        // Fill the large source
        for i in 1..=5 {
            large_src.push(i);
        }

        small_dest.fill_from(&mut large_src);

        // Should transfer last 3 elements from large_src to fill small_dest
        assert_eq!(small_dest.len(), 3);
        assert_eq!(small_dest.to_vec(), vec![3, 4, 5]); // Last 3 elements
        assert_eq!(large_src.len(), 2);
        assert_eq!(large_src.to_vec(), vec![1, 2]); // First 2 elements remain

        // Test filling a larger vec from a smaller vec
        let mut large_dest: HeaplessVec<i32, 8> = HeaplessVec::new();
        let mut small_src: HeaplessVec<i32, 4> = HeaplessVec::new();

        // Pre-fill large_dest partially
        large_dest.push(100);
        large_dest.push(200);

        // Fill small_src
        small_src.push(10);
        small_src.push(20);
        small_src.push(30);

        large_dest.fill_from(&mut small_src);

        // Should transfer all elements from small_src
        assert_eq!(large_dest.len(), 5); // 2 + 3 = 5
        assert_eq!(large_dest.to_vec(), vec![100, 200, 10, 20, 30]);
        assert_eq!(small_src.len(), 0); // All elements transferred
        assert_eq!(small_src.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_iter() {
        let mut vec: HeaplessVec<i32, 5> = HeaplessVec::new();

        // Test empty vec iteration
        assert_eq!(vec.into_iter().count(), 0);
        assert_eq!(vec.into_iter_mut().len(), 0);

        // Add some elements
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Test iteration count
        assert_eq!(vec.into_iter().count(), 3);

        // Test iteration values
        let mut iter = vec.into_iter();
        assert_eq!(Some(&1), iter.next());
        assert_eq!(Some(&2), iter.next());
        assert_eq!(Some(&3), iter.next());
        assert_eq!(None, iter.next());

        // Test collecting into Vec
        let collected: Vec<i32> = vec.into_iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);

        // Hold references to both ends while iterating the remaining initialized element.
        let mut iter = vec.into_iter_mut();
        let first = iter.next().expect("first element");
        let last = iter.next_back().expect("last element");
        assert_eq!(iter.len(), 1);
        *first += 10;
        *last += 30;
        *iter.next().expect("middle element") += 20;
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
        assert_eq!(vec.to_slice(), &[11, 22, 33]);

        vec.clear();
        assert!(vec.into_iter_mut().next().is_none());
        let mut zero_capacity = HeaplessVec::<i32, 0>::new();
        assert!(zero_capacity.into_iter_mut().next().is_none());
    }

    #[test]
    fn test_bitmask_as_bits_lsb_first_matches_reference() {
        #[allow(
            clippy::needless_range_loop,
            reason = "Indices align board coordinates or parallel buffers; retain the existing traversal and independent reference calculation."
        )]
        fn ref_bits<const N: usize>(x: u64) -> [u8; N] {
            let mut out = [0u8; N];
            for i in 0..N {
                out[i] = ((x >> i) & 1) as u8;
            }
            out
        }

        // A few fixed values that exercise patterns + edge cases.
        let cases = [
            0u64,
            1u64,
            0xFFFF_FFFF_FFFF_FFFFu64,
            0x8000_0000_0000_0000u64,
            0x0123_4567_89AB_CDEFu64,
            0xF0F0_F0F0_F0F0_F0F0u64,
        ];

        for &x in &cases {
            macro_rules! assert_bits {
                ($n:literal) => {{
                    let got = BitMask::<$n>::new_from_u64(x).as_slice();
                    let expected = ref_bits::<$n>(x);
                    assert_eq!(
                        got, expected,
                        "BitMask::<{}> mismatch\nx        = {:#066b}\nexpected = {:?}\ngot      = {:?}",
                        $n, x, expected, got
                    );
                }};
            }

            assert_bits!(1);
            assert_bits!(7);
            assert_bits!(40);
            assert_bits!(64);
        }
    }

    #[test]
    fn test_bitmask_widen_narrow() {
        let m = BitMask::<10>::new_from_u64(0b11_0110_1001);

        // widen preserves the low N bits.
        let w: BitMask<16> = m.widen();
        assert_eq!(w.inner(), 0b11_0110_1001);

        // widen discards garbage bits above the source width.
        let dirty = BitMask::<4>::new_from_u64(0xFF);
        let w2: BitMask<8> = dirty.widen();
        assert_eq!(w2.inner(), 0x0F);

        // narrow truncates to the low M bits.
        let n: BitMask<4> = m.narrow();
        assert_eq!(n.inner(), 0b1001);

        // widen then narrow back round-trips.
        let rt: BitMask<10> = m.widen::<16>().narrow();
        assert_eq!(rt.inner(), m.inner());

        // Degenerate widths: M == N both directions, full 64-bit width.
        assert_eq!(m.widen::<10>().inner(), m.inner());
        assert_eq!(m.narrow::<10>().inner(), m.inner());
        let full = BitMask::<64>::new_from_u64(u64::MAX);
        assert_eq!(full.widen::<64>().inner(), u64::MAX);
        assert_eq!(full.narrow::<1>().inner(), 1);
    }

    #[test]
    fn test_bitmask_as_bits_lsb_first_random_is_0_or_1() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let x: u64 = rng.random();
            let bits = BitMask::<64>::new_from_u64(x).as_slice();
            for &b in &bits {
                assert!(b <= 1, "expected 0/1 bit lane, got {b}");
            }
        }
    }

    // ---- transpose_bits tests ----

    /// Naive reference transpose for verification.
    fn naive_transpose<const ROWS: usize, const COLS: usize, const N: usize>(
        m: &[u8; N],
    ) -> [u8; N] {
        let mut out = [0u8; N];
        for row in 0..ROWS {
            for col in 0..COLS {
                let src = row * COLS + col;
                let dst = col * ROWS + row;
                let bit = (m[src / 8] >> (src % 8)) & 1;
                out[dst / 8] |= bit << (dst % 8);
            }
        }
        out
    }

    #[test]
    fn test_transpose_8x8_involution() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let orig: [u8; 8] = std::array::from_fn(|_| rng.random());
            let mut m = orig;
            transpose_bits::<8, 8, 8>(&mut m);
            transpose_bits::<8, 8, 8>(&mut m);
            assert_eq!(m, orig);
        }
    }

    #[test]
    fn test_transpose_8x8_matches_naive() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let orig: [u8; 8] = std::array::from_fn(|_| rng.random());
            let mut m = orig;
            transpose_bits::<8, 8, 8>(&mut m);
            assert_eq!(m, naive_transpose::<8, 8, 8>(&orig));
        }
    }

    #[test]
    fn test_transpose_4x4_involution() {
        for val in 0..=u16::MAX {
            let orig = val.to_le_bytes();
            let mut m = orig;
            transpose_bits::<4, 4, 2>(&mut m);
            transpose_bits::<4, 4, 2>(&mut m);
            assert_eq!(m, orig, "4x4 involution failed for {val:#06x}");
        }
    }

    #[test]
    fn test_transpose_4x4_matches_naive() {
        for val in 0..=u16::MAX {
            let orig = val.to_le_bytes();
            let mut m = orig;
            transpose_bits::<4, 4, 2>(&mut m);
            assert_eq!(
                m,
                naive_transpose::<4, 4, 2>(&orig),
                "4x4 naive mismatch for {val:#06x}"
            );
        }
    }

    #[test]
    fn test_transpose_nonsquare_2x4() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let orig: [u8; 1] = [rng.random()];
            let mut m = orig;
            transpose_bits::<2, 4, 1>(&mut m);
            assert_eq!(m, naive_transpose::<2, 4, 1>(&orig));
            // Transpose back with swapped dims
            transpose_bits::<4, 2, 1>(&mut m);
            assert_eq!(m, orig);
        }
    }

    #[test]
    fn test_transpose_10x20_involution() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut orig = [0u8; 25];
            for b in orig.iter_mut() {
                *b = rng.random();
            }
            let mut m = orig;
            transpose_bits::<10, 20, 25>(&mut m);
            transpose_bits::<20, 10, 25>(&mut m);
            assert_eq!(m, orig);
        }
    }

    #[test]
    fn test_transpose_10x20_matches_naive() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut orig = [0u8; 25];
            for b in orig.iter_mut() {
                *b = rng.random();
            }
            let mut m = orig;
            transpose_bits::<10, 20, 25>(&mut m);
            assert_eq!(m, naive_transpose::<10, 20, 25>(&orig));
        }
    }

    #[test]
    fn test_transpose_identity_1x1() {
        let mut m = [0x01u8];
        transpose_bits::<1, 1, 1>(&mut m);
        assert_eq!(m, [0x01]);
    }

    #[test]
    fn test_transpose_all_zeros() {
        let mut m = [0u8; 8];
        transpose_bits::<8, 8, 8>(&mut m);
        assert_eq!(m, [0u8; 8]);
    }

    #[test]
    fn test_transpose_all_ones_row() {
        // Row 0 all set in 8x8 → after transpose, bit 0 of every byte set
        let mut m = [0xFFu8, 0, 0, 0, 0, 0, 0, 0];
        transpose_bits::<8, 8, 8>(&mut m);
        assert_eq!(m, [0x01; 8]);
    }
}
