use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Counter;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOC_MOVES: AtomicUsize = AtomicUsize::new(0);
static REALLOC_MOVED_BYTES: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let now = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every operation forwards the same pointer/layout to System. Counting
// uses atomics, handles allocation failure, and never allocates recursively.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            add(l.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            add(l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        // SAFETY: pointer and original layout are forwarded unchanged.
        unsafe { System.dealloc(p, l) };
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        // SAFETY: the GlobalAlloc caller supplies the live allocation and new size.
        let old = p;
        let p = unsafe { System.realloc(old, l, n) };
        if !p.is_null() {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            if p != old {
                REALLOC_MOVES.fetch_add(1, Ordering::Relaxed);
                REALLOC_MOVED_BYTES.fetch_add(l.size().min(n), Ordering::Relaxed);
            }
            if n >= l.size() {
                add(n - l.size());
            } else {
                LIVE.fetch_sub(l.size() - n, Ordering::Relaxed);
            }
        }
        p
    }
}

pub fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

pub fn begin() {
    PEAK.store(live(), Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOC_MOVES.store(0, Ordering::Relaxed);
    REALLOC_MOVED_BYTES.store(0, Ordering::Relaxed);
}

/// Successful requests since begin. Moved bytes are a logical size bound, not
/// measured copy traffic: System may relocate large allocations by remapping.
pub fn requests() -> (usize, usize, usize, usize) {
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        REALLOCATIONS.load(Ordering::Relaxed),
        REALLOC_MOVES.load(Ordering::Relaxed),
        REALLOC_MOVED_BYTES.load(Ordering::Relaxed),
    )
}

pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}
