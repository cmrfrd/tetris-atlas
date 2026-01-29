use std::mem::MaybeUninit;
use std::ptr;

use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::hint::spin_loop;
use std::ops::Index;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use proc_macros::inline_conditioned;

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
/// use tetris_game::utils::bits_to_byte;
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
/// use tetris_game::utils::{bits_to_byte, byte_to_bits};
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

    let _ = [()][(!verify_all_u8()) as usize];
};

pub struct BitMask<const N: usize>(u64);

impl<const N: usize> BitMask<N>
where
    [(); N - 1]:,
    [(); 64 - N]:,
{
    #[inline_conditioned(always)]
    pub const fn new_from_u64(x: u64) -> Self {
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
}

/// A spin lock for shared-mutual exclusion.
#[derive(Debug)]
pub struct SpinLock<T: Debug> {
    lock: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Debug> Sync for SpinLock<T> where T: Send {}

pub struct SpinLockGuard<'a, T> {
    lock: &'a AtomicBool,
    data: &'a mut T,
}

impl<'a, T> Drop for SpinLockGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.store(false, AtomicOrdering::Release);
    }
}

impl<T: Debug> SpinLock<T> {
    /// Creates a new `SpinLock` containing the given data.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to protect with the spin lock
    ///
    /// # Returns
    ///
    /// A new `SpinLock` instance in the unlocked state
    pub const fn new(data: T) -> Self {
        SpinLock {
            lock: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquires the lock, blocking the current thread until it becomes available.
    ///
    /// This method will spin in a tight loop until the lock is acquired. The returned
    /// guard will release the lock when dropped.
    ///
    /// # Returns
    ///
    /// A guard that provides mutable access to the protected data
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

    /// Attempts to acquire the lock without blocking.
    ///
    /// If the lock is currently held by another thread, this method returns `None`
    /// immediately instead of spinning.
    ///
    /// # Returns
    ///
    /// `Some(guard)` if the lock was successfully acquired, `None` otherwise
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

/// /////////////////////////////////////////////////////////////////
/// FixedBinMinHeap - Fixed-capacity binary min-heap
///

/// Fast fixed-capacity binary MIN-heap
/// - Stack-allocated, zero heap allocations
/// - Compile-time loop unrolling with early exit
/// - Optimized for beam search (keep best K items)
pub struct FixedBinMinHeap<T: Copy + Ord, const SIZE: usize> {
    data: [MaybeUninit<T>; SIZE],
    len: usize,
}

impl<T: Copy + Ord, const SIZE: usize> FixedBinMinHeap<T, SIZE> {
    // Calculate the minimum number of steps needed to sift up/down the heap
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
        // SAFETY: `FixedBinMinHeap` maintains the invariant that indices `0..self.len`
        // are initialized, and `self.len..SIZE` may be uninitialized.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Return a mutable slice of the initialized elements [0..len)
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `FixedBinMinHeap` maintains the invariant that indices `0..self.len`
        // are initialized, and `self.len..SIZE` may be uninitialized.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    /// Push into heap (must not be full). Unrolled sift-up with early exit.
    #[inline_conditioned(always)]
    pub fn push(&mut self, v: T) {
        debug_assert!(self.len < SIZE, "FixedBinMinHeap::push called when full");
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
    #[inline_conditioned(always)]
    pub fn replace_min(&mut self, v: T) {
        debug_assert!(self.len != 0);
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
    #[inline_conditioned(always)]
    pub fn push_if_better_min_heap(&mut self, v: T) -> bool {
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

    // ----------------- unrolled sift-up/down with early exit -----------------

    #[inline_conditioned(always)]
    unsafe fn sift_up_min_unrolled(&mut self, start: usize) {
        // SAFETY: We maintain the invariant that 0..len are initialized
        let ptr = self.data.as_mut_ptr().cast::<T>();
        let mut i = start;

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

        crate::repeat_idx_unroll!(Self::STEPS, _I, {
            if !unsafe { step_up::<T>(ptr, &mut i) } {
                return; // Early exit when no swap happened
            }
        });
    }

    #[inline(always)]
    unsafe fn sift_down_min_unrolled(&mut self, start: usize) {
        // SAFETY: We maintain the invariant that 0..len are initialized
        let ptr = self.data.as_mut_ptr().cast::<T>();
        let len = self.len;
        let mut i = start;

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
                // Only left child exists
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

        crate::repeat_idx_unroll!(Self::STEPS, _I, {
            if !unsafe { step_down::<T>(ptr, len, &mut i) } {
                return; // Early exit when no swap happened
            }
        });
    }
}

/// HeaplessVec

#[derive(Debug, Clone, Copy)]
pub struct HeaplessVec<T: Copy, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
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
    /// # Safety
    ///
    /// The pointer is only valid for `self.len()` elements and may contain uninitialized data
    /// for indices `self.len()..N`.
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
    /// # Safety
    ///
    /// The caller must ensure the vector is not full before calling this method.
    /// Pushing to a full vector will cause undefined behavior.
    #[inline_conditioned(always)]
    pub fn push(&mut self, item: T) {
        unsafe {
            *self.data.get_unchecked_mut(self.len) = MaybeUninit::new(item); // safe because we know the index is in bounds
            self.len += 1;
        }
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

    /// Applies a function to each element in the vector, allowing mutation.
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
                f(&self.data[i].assume_init_ref());
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
                f(&mut self.data[i].assume_init_mut());
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

    /// Fills this vector with all elements from a `Vec`.
    ///
    /// # Arguments
    ///
    /// * `vec` - The source vector to copy elements from
    ///
    /// # Panics
    ///
    /// Panics if the source vector has more elements than this vector's capacity
    pub fn fill_from_vec(&mut self, vec: &Vec<T>) {
        self.len = vec.len();
        unsafe {
            std::ptr::copy_nonoverlapping(
                vec.as_ptr(),
                self.data.as_mut_ptr() as *mut T,
                vec.len(),
            );
        }
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
        // SAFETY: `len == N` implies all elements `0..N` were initialized.
        // We reuse the safe slice view and then convert it to an array reference.
        self.to_slice().try_into().ok()
    }

    /// Copies the contents into an array **only if** the vector is full.
    ///
    /// Returns `None` when `self.len != N`.
    pub fn to_array(&self) -> Option<[T; N]> {
        self.as_array().map(|a| *a)
    }

    /// Returns a slice containing all elements in the vector.
    ///
    /// # Returns
    ///
    /// A slice view of the vector's elements
    pub fn to_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Returns an iterator over the elements of the vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding immutable references to each element
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.to_slice().iter()
    }

    /// Returns a mutable iterator over the elements of the vector.
    ///
    /// # Returns
    ///
    /// An iterator yielding mutable references to each element
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len).iter_mut()
        }
    }
}

impl<T: Copy, const N: usize> Index<usize> for HeaplessVec<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.data[index].assume_init_ref() }
    }
}

/// Vec Pool
pub struct VecPool<T: Copy + Sized + Debug, const N: usize, const M: usize> {
    pool: [SpinLock<HeaplessVec<T, N>>; M],
}

impl<T: Copy + Sized + Debug, const N: usize, const M: usize> VecPool<T, N, M> {
    /// Creates a new pool of `M` vectors, each with capacity `N`.
    ///
    /// # Returns
    ///
    /// A new `VecPool` with all vectors initialized to empty state
    pub const fn new() -> Self {
        let mut pool: [SpinLock<HeaplessVec<T, N>>; M] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut i = 0;
        while i < M {
            pool[i] = SpinLock::new(HeaplessVec::new());
            i += 1;
        }
        Self { pool }
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

/// Compile-time loop unrolling with const index variable.
///
/// This macro unrolls loops at compile time, allowing the index variable to be `const`,
/// which enables full const evaluation of expressions like `8 * i` and `i % 10`.
///
/// **IMPORTANT**: Each match branch incurs significant compilation time cost due to
/// macro expansion and monomorphization. Only add branches for N values you actually use.
/// Adding unused branches will slow down compilation without providing any benefit.
#[macro_export]
macro_rules! repeat_idx_unroll {
    ($N:expr, $i:ident, $b:block) => {
        match const { $N } {
            0_usize => {}
            1_usize => $crate::repeat_exact_idx!(1, $i, $b),
            2_usize => $crate::repeat_exact_idx!(2, $i, $b),
            3_usize => $crate::repeat_exact_idx!(3, $i, $b),
            4_usize => $crate::repeat_exact_idx!(4, $i, $b),
            5_usize => $crate::repeat_exact_idx!(5, $i, $b),
            6_usize => $crate::repeat_exact_idx!(6, $i, $b),
            7_usize => $crate::repeat_exact_idx!(7, $i, $b),
            8_usize => $crate::repeat_exact_idx!(8, $i, $b),
            10_usize => $crate::repeat_exact_idx!(10, $i, $b),
            16_usize => $crate::repeat_exact_idx!(16, $i, $b),
            25_usize => $crate::repeat_exact_idx!(25, $i, $b),
            32_usize => $crate::repeat_exact_idx!(32, $i, $b),
            40_usize => $crate::repeat_exact_idx!(40, $i, $b),
            64_usize => $crate::repeat_exact_idx!(64, $i, $b),
            _ => panic!(
                "{}",
                concat!("repeat_idx_unroll! supports up to N={", stringify!($N), "}")
            ),
        }
    };
}

/// Right shifts elements in a u32 array based on a mask, performing `I` iterations.
///
/// For each iteration, this function:
/// 1. Finds the most significant bit (MSB) in the mask
/// 2. Right shifts all elements by 1 position from that bit onward
/// 3. Clears the MSB from the mask
///
/// # Type Parameters
///
/// * `N` - The number of u32 elements in the array
/// * `I` - The number of shift iterations to perform
///
/// # Arguments
///
/// * `xs` - Array of u32 values to shift
/// * `m` - Mask indicating which bit positions to shift from
///
/// # Example
/// ```rust
/// use tetris_game::utils::rshift_slice_from_mask_u32;
/// let mut arr = [0b1111_0000_0000_0000_0000_0000_0000_0000_u32; 2];
/// let mask = 0b0000_0000_0000_0000_0000_0000_0000_1111_u32;
/// rshift_slice_from_mask_u32::<2, 4>(&mut arr, mask);
/// // After 4 iterations, each 1 in the mask causes a right shift by 1
/// // So the result shifts everything down by 4 positions
/// assert_eq!(arr, [0b0000_1111_0000_0000_0000_0000_0000_0000_u32; 2]);
/// ```
#[inline_conditioned(always)]
pub const fn rshift_slice_from_mask_u32<const N: usize, const I: usize>(
    xs: &mut [u32; N],
    mut m: u32,
) {
    #[inline_conditioned(always)]
    const fn step<const N: usize>(xs: &mut [u32; N], m: &mut u32) -> bool {
        if *m == 0 {
            return false;
        }
        let pos = m.trailing_zeros();
        let pivot = 1u32 << pos;
        let keep = pivot - 1; // Mask for bits below pos (0 to pos-1)
        let above = !pivot & !keep; // Mask for bits above pos (pos+1 to 31)

        // Remove row at pos: keep bits below, shift bits above down by 1
        repeat_idx_unroll!(N, I, {
            xs[I] = (xs[I] & keep) | ((xs[I] & above) >> 1);
        });

        // Clear the processed bit from the mask and shift mask down too
        *m = (*m & !pivot) >> 1;
        true
    }

    repeat_idx_unroll!(I, _I, {
        if !step(xs, &mut m) {
            return;
        }
    });
}

/// Right shifts a slice of u32s by `n` bits starting from a given index.
///
/// For each u32 in the slice:
/// - Bits before `idx` are kept unchanged
/// - Bits after `idx` are shifted right by `n` positions
///
/// # Arguments
///
/// * `xs` - Array of u32s to shift
/// * `idx` - Starting bit index for the shift
/// * `n` - Number of positions to shift right
///
/// # Example
/// ```rust
/// use tetris_game::utils::rshift_slice_n_from_index_u32;
/// let mut arr = [0b1111_0000_0000_0000_0000_0000_0000_0000_u32; 1];
/// let expected = 0b0000_1111_0000_0000_0000_0000_0000_0000;
/// rshift_slice_n_from_index_u32(&mut arr, 8, 4);
/// assert_eq!(
///     arr[0], expected,
///     "\nExpected {:032b},\n     got {:032b}",
///     expected,
///     arr[0]
/// );
/// ```
#[inline_conditioned(always)]
pub const fn rshift_slice_n_from_index_u32<const N: usize>(
    xs: &mut [u32; N],
    idx: usize,
    n: usize,
) {
    let mask = u32::MAX >> idx;
    repeat_idx_unroll!(N, I, {
        let shifted = (xs[I] >> n) & !mask;
        let kept = xs[I] & mask;
        xs[I] = shifted | kept;
    });
}

/// Right shifts a u32 by `n` bits starting from a given index.
///
/// Bits before `idx` are kept unchanged, while bits from `idx` onward are shifted right.
///
/// # Arguments
///
/// * `x` - The u32 value to shift
/// * `idx` - Starting bit index for the shift (0 = MSB)
/// * `n` - Number of positions to shift right
///
/// # Returns
///
/// The shifted u32 value
///
/// # Example
///
/// ```
/// use tetris_game::utils::rshift_n_from_index;
/// let x = 0b1111_0000_0000_0000_0000_0000_0000_0000u32;
/// let result = rshift_n_from_index(x, 8, 4);
/// assert_eq!(result, 0b0000_1111_0000_0000_0000_0000_0000_0000u32);
/// ```
#[inline_conditioned(always)]
pub const fn rshift_n_from_index(x: u32, idx: usize, n: usize) -> u32 {
    let mask = u32::MAX >> idx;
    let shifted = (x >> n) & !mask;
    let kept = x & mask;
    shifted | kept
}

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
pub fn choose_set_bit_u64<R: rand::Rng + ?Sized>(mask: u64, rng: &mut R) -> Option<u32> {
    (mask != 0).then(|| {
        let n = mask.count_ones(); // 1..=64, fits in u32
        let k = (rng.next_u64() % (n as u64)) as u32; // bias acceptable
        select_kth_u64(mask, k)
    })
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_repeat_idx_generic() {
        const MAX_N: usize = 40;
        let mut test_vec = Vec::new();
        repeat_idx_unroll!(MAX_N, I, {
            test_vec.push(I);
        });
        assert_eq!(test_vec, (0..MAX_N).collect::<Vec<_>>());
    }

    #[test]
    fn test_rshift_n_from_index() {
        assert_eq!(
            rshift_n_from_index(0b1111_0000_0000_0000_0000_0000_0000_0000, 0, 4),
            0b1111_0000_0000_0000_0000_0000_0000_0000
        );
        assert_eq!(
            rshift_n_from_index(0b1111_0000_0000_0000_0000_0000_0000_0000, 4, 4),
            0b0000_0000_0000_0000_0000_0000_0000_0000
        );
        assert_eq!(
            rshift_n_from_index(0b1111_0000_0000_0000_0000_0000_0000_0000, 8, 4),
            0b0000_1111_0000_0000_0000_0000_0000_0000
        );
        assert_eq!(
            rshift_n_from_index(0b1111_0000_0000_0000_0000_0000_0000_1111, 12, 4),
            0b0000_1111_0000_0000_0000_0000_0000_1111
        );
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
            panic!("k out of range for x (k={k}, popcnt={})", x.count_ones());
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
        assert_eq!(vec.iter().count(), 0);

        // Add some elements
        vec.push(1);
        vec.push(2);
        vec.push(3);

        // Test iteration count
        assert_eq!(vec.iter().count(), 3);

        // Test iteration values
        let mut iter = vec.iter();
        assert_eq!(Some(&1), iter.next());
        assert_eq!(Some(&2), iter.next());
        assert_eq!(Some(&3), iter.next());
        assert_eq!(None, iter.next());

        // Test collecting into Vec
        let collected: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_bitmask_as_bits_lsb_first_matches_reference() {
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
                    let got = BitMask::<$n>(x).as_slice();
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
    fn test_bitmask_as_bits_lsb_first_random_is_0_or_1() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let x: u64 = rng.random();
            let bits = BitMask::<64>(x).as_slice();
            for &b in &bits {
                assert!(b <= 1, "expected 0/1 bit lane, got {b}");
            }
        }
    }
}
