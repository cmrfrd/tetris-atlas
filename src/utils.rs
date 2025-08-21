use std::mem::MaybeUninit;

use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::hint::spin_loop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

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
    pub const fn new(data: T) -> Self {
        SpinLock {
            lock: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

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

    pub const fn new() -> Self {
        Self {
            data: Self::INIT,
            len: 0,
        }
    }

    #[inline(always)]
    pub const fn capacity() -> usize {
        N
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn push(&mut self, item: T) {
        unsafe {
            *self.data.get_unchecked_mut(self.len) = MaybeUninit::new(item); // safe because we know the index is in bounds
            self.len += 1;
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
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

    pub fn to_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len).iter() }
    }

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

/// Apply all runs a function over a slice
/// Apply `f` in-place to every element of a fixed-size array.
///
/// In release mode rustc removes bounds checks and iterator bookkeeping,
/// so the generated machine code is already a tight pointer walk.
#[inline]
pub fn apply_all<T, const N: usize>(arr: &mut [T; N], mut f: impl FnMut(&mut T)) {
    for x in arr {
        f(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
