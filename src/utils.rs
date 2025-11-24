use std::mem::MaybeUninit;

use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::hint::spin_loop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

use proc_macros::inline_conditioned;

/// Applies the MurmurHash3 64-bit finalization mixer to a value.
///
/// # Arguments
///
/// * `x` - The 64-bit value to mix
///
/// # Returns
///
/// The mixed 64-bit value with improved bit distribution
///
/// # Example
///
/// ```
/// use tetris_atlas::utils::fmix64;
/// let mut x = 12345u64;
/// fmix64(&mut x);
/// assert_eq!(x, 1716623506685013753u64); // known pre-image
/// ```
#[inline_conditioned(always)]
pub const fn fmix64(x: &mut u64) {
    *x ^= *x >> 33;
    *x = x.wrapping_mul(0xff51afd7ed558ccd);
    *x ^= *x >> 33;
    *x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    *x ^= *x >> 33;
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
/// use tetris_atlas::utils::bits_to_byte;
/// let bits = [1, 0, 1, 0, 1, 0, 1, 0];
/// assert_eq!(bits_to_byte(&bits), 0b10101010);
/// ```
#[inline_conditioned(always)]
pub const fn bits_to_byte(bits: &[u8; 8]) -> u8 {
    let packed = u64::from_le_bytes(*bits);

    (((packed) & 1) << 7
        | ((packed >> 8) & 1) << 6
        | ((packed >> 16) & 1) << 5
        | ((packed >> 24) & 1) << 4
        | ((packed >> 32) & 1) << 3
        | ((packed >> 40) & 1) << 2
        | ((packed >> 48) & 1) << 1
        | ((packed >> 56) & 1)) as u8
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
        self.lock.store(false, Ordering::Release);
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
        while self.lock.swap(true, Ordering::Acquire) {
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
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
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

/// HeaplessVec

#[derive(Debug, Clone, Copy)]
pub struct HeaplessVec<T: Copy + Sized + Debug, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
}

impl<T: Copy + Sized + Debug, const N: usize> HeaplessVec<T, N> {
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
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len).iter() }
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
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
    }};
    (17, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep1_at!(16, $i, $b);
    }};
    (18, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep2_at!(16, $i, $b);
    }};
    (19, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep2_at!(16, $i, $b);
        $crate::rep1_at!(18, $i, $b);
    }};
    (20, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep4_at!(16, $i, $b);
    }};
    (21, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep4_at!(16, $i, $b);
        $crate::rep1_at!(20, $i, $b);
    }};
    (22, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep4_at!(16, $i, $b);
        $crate::rep2_at!(20, $i, $b);
    }};
    (23, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep4_at!(16, $i, $b);
        $crate::rep2_at!(20, $i, $b);
        $crate::rep1_at!(22, $i, $b);
    }};
    (24, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
    }};
    (25, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep1_at!(24, $i, $b);
    }};
    (26, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep2_at!(24, $i, $b);
    }};
    (27, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep2_at!(24, $i, $b);
        $crate::rep1_at!(26, $i, $b);
    }};
    (28, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
    }};
    (29, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
        $crate::rep1_at!(28, $i, $b);
    }};
    (30, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
        $crate::rep2_at!(28, $i, $b);
    }};
    (31, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep4_at!(24, $i, $b);
        $crate::rep2_at!(28, $i, $b);
        $crate::rep1_at!(30, $i, $b);
    }};
    (32, $i:ident, $b:block) => {{
        $crate::rep8_at!(0, $i, $b);
        $crate::rep8_at!(8, $i, $b);
        $crate::rep8_at!(16, $i, $b);
        $crate::rep8_at!(24, $i, $b);
    }};
}

#[macro_export]
macro_rules! repeat_idx_generic {
    ($N:expr, $i:ident, $b:block) => {
        match $N {
            0 => {}
            1 => {
                $crate::repeat_exact_idx!(1, $i, $b)
            }
            2 => {
                $crate::repeat_exact_idx!(2, $i, $b)
            }
            3 => {
                $crate::repeat_exact_idx!(3, $i, $b)
            }
            4 => {
                $crate::repeat_exact_idx!(4, $i, $b)
            }
            5 => {
                $crate::repeat_exact_idx!(5, $i, $b)
            }
            6 => {
                $crate::repeat_exact_idx!(6, $i, $b)
            }
            7 => {
                $crate::repeat_exact_idx!(7, $i, $b)
            }
            8 => {
                $crate::repeat_exact_idx!(8, $i, $b)
            }
            9 => {
                $crate::repeat_exact_idx!(9, $i, $b)
            }
            10 => {
                $crate::repeat_exact_idx!(10, $i, $b)
            }
            11 => {
                $crate::repeat_exact_idx!(11, $i, $b)
            }
            12 => {
                $crate::repeat_exact_idx!(12, $i, $b)
            }
            13 => {
                $crate::repeat_exact_idx!(13, $i, $b)
            }
            14 => {
                $crate::repeat_exact_idx!(14, $i, $b)
            }
            15 => {
                $crate::repeat_exact_idx!(15, $i, $b)
            }
            16 => {
                $crate::repeat_exact_idx!(16, $i, $b)
            }
            17 => {
                $crate::repeat_exact_idx!(17, $i, $b)
            }
            18 => {
                $crate::repeat_exact_idx!(18, $i, $b)
            }
            19 => {
                $crate::repeat_exact_idx!(19, $i, $b)
            }
            20 => {
                $crate::repeat_exact_idx!(20, $i, $b)
            }
            21 => {
                $crate::repeat_exact_idx!(21, $i, $b)
            }
            22 => {
                $crate::repeat_exact_idx!(22, $i, $b)
            }
            23 => {
                $crate::repeat_exact_idx!(23, $i, $b)
            }
            24 => {
                $crate::repeat_exact_idx!(24, $i, $b)
            }
            25 => {
                $crate::repeat_exact_idx!(25, $i, $b)
            }
            26 => {
                $crate::repeat_exact_idx!(26, $i, $b)
            }
            27 => {
                $crate::repeat_exact_idx!(27, $i, $b)
            }
            28 => {
                $crate::repeat_exact_idx!(28, $i, $b)
            }
            29 => {
                $crate::repeat_exact_idx!(29, $i, $b)
            }
            30 => {
                $crate::repeat_exact_idx!(30, $i, $b)
            }
            31 => {
                $crate::repeat_exact_idx!(31, $i, $b)
            }
            32 => {
                $crate::repeat_exact_idx!(32, $i, $b)
            }
            _ => panic!("repeat_idx_generic! supports up to N=32"),
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
/// use tetris_atlas::utils::rshift_slice_from_mask_u32;
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
        let keep = pivot - 1; // Mask for bits below pos
        let shift_mask = !keep; // Mask for bits at and above pos

        // Shift each column: bits above pivot shift down, bits at/below pivot stay
        repeat_idx_generic!(N, I, {
            xs[I] = ((xs[I] >> 1) & shift_mask) | (xs[I] & keep);
        });

        // Clear the processed bit from the mask
        *m &= !pivot;
        true
    }

    repeat_idx_generic!(I, _I, {
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
/// use tetris_atlas::utils::rshift_slice_n_from_index_u32;
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
    repeat_idx_generic!(N, I, {
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
/// use tetris_atlas::utils::rshift_n_from_index;
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

#[inline_conditioned(always)]
pub const fn trailing_zeros_all<const N: usize>(xs: [u32; N]) -> [u32; N] {
    let mut trailing_zeros = [0; N];
    repeat_idx_generic!(N, I, {
        trailing_zeros[I] = xs[I].trailing_zeros();
    });
    trailing_zeros
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
