use crate::tetris::{TetrisGame, TetrisPiecePlacement, TetrisUint};
use crate::utils::{HeaplessVec, VecPool};
use rayon::iter::ParallelIterator;
use rayon::iter::plumbing::{Folder, Reducer, UnindexedConsumer};
use rayon::{current_num_threads, join_context};
use std::iter::Iterator;
use std::{collections::VecDeque, sync::Arc};

use std::sync::atomic::{AtomicU64, Ordering};

const POOL_SIZE: usize = 8;

const QUEUE_SPLIT_SIZE: usize = 16;
const MEGA_BATCH_SIZE: usize = 512;
const _: () = assert!(
    MEGA_BATCH_SIZE > CHILD_NODE_BUFFER_SIZE,
    "MEGA_BATCH_SIZE must be greater than MAX_PIECE_PLACEMENT_COUNT"
);

const MAX_PIECE_PLACEMENT_COUNT: usize = TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT;
const CHILD_NODE_BUFFER_SIZE: usize = MAX_PIECE_PLACEMENT_COUNT;
static CHILD_NODE_VEC_POOL: VecPool<TetrisExplorerItem, CHILD_NODE_BUFFER_SIZE, POOL_SIZE> =
    VecPool::new();

static ITER_VEC_POOL: VecPool<TetrisExplorerItem, MEGA_BATCH_SIZE, POOL_SIZE> = VecPool::new();

const BLOOM_SIZE_MB: usize = 32;
const BLOOM_SIZE_BYTES: usize = BLOOM_SIZE_MB * 1024 * 1024;
const BITMASK_BITS: usize = BLOOM_SIZE_BYTES * 8;
const BITS_PER_WORD: usize = 64;
const WORDS: usize = BITMASK_BITS / BITS_PER_WORD;

/// A bloom filter for remembering u256 states
pub struct TetrisBloom1 {
    bits: Vec<AtomicU64>,
}

impl TetrisBloom1 {
    pub fn new() -> Self {
        let bits: Vec<AtomicU64> = (0..WORDS).map(|_| AtomicU64::new(0)).collect();
        TetrisBloom1 { bits }
    }

    const fn hash_tetris_uint(x: TetrisUint) -> usize {
        let limbs = x.as_limbs();

        // Mix limbs with different rotations and multipliers
        let mut h = limbs[0];
        h ^= limbs[1].rotate_left(16);
        h ^= limbs[2].rotate_left(32);
        h ^= limbs[3].rotate_left(48);

        // Apply avalanche mixing
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;

        h as usize
    }

    /// Returns true if already seen, false if first time
    pub fn check_and_mark(&self, x: TetrisUint) -> bool {
        let hash = Self::hash_tetris_uint(x);
        let bit_index = hash % BITMASK_BITS;
        let word = bit_index / BITS_PER_WORD;
        let bit = bit_index % BITS_PER_WORD;
        let mask = 1u64 << bit;

        let cell = &self.bits[word];
        let old = cell.fetch_or(mask, Ordering::Relaxed);
        (old & mask) != 0
    }

    pub fn reset(&self) {
        for bit in self.bits.iter() {
            bit.store(0, Ordering::Relaxed);
        }
    }
}

/// An iterator that can be split.
pub trait SplittableIterator: Iterator + Sized {
    /// Split this iterator in two, if possible.
    ///
    /// Returns a newly allocated [`SplittableIterator`] of the second half,
    /// or [`None`], if the iterator is too small to split.
    ///
    /// After the call, [`self`]
    /// will be left containing the first half.
    ///
    /// [`None`]: type@std::option::Option::None
    /// [`self`]: trait@self::SplittableIterator
    fn split(&mut self) -> Option<Self>;
}

/// Converts a [`SplittableIterator`] into a [`rayon::iter::ParallelIterator`].
pub trait IntoParallelIterator: Sized {
    /// Parallelizes this iterator.
    ///
    /// Returns a [`ParallelSplittableIterator`] bridge that implements
    /// [`rayon::iter::ParallelIterator`].
    fn into_par_iter(self) -> ParallelSplittableIterator<Self>;
}

impl<T> IntoParallelIterator for T
where
    T: SplittableIterator + Send,
    T::Item: Send,
{
    fn into_par_iter(self) -> ParallelSplittableIterator<Self> {
        ParallelSplittableIterator::new(self)
    }
}

/// A bridge from a [`SplittableIterator`] to a [`rayon::iter::ParallelIterator`].
pub struct ParallelSplittableIterator<Iter> {
    iter: Iter,
    splits: usize,
}

impl<Iter> ParallelSplittableIterator<Iter>
where
    Iter: SplittableIterator,
{
    /// Creates a new [`ParallelSplittableIterator`] bridge from a [`SplittableIterator`].
    pub fn new(iter: Iter) -> Self {
        Self {
            iter,
            splits: current_num_threads(),
        }
    }

    /// Split the underlying iterator in half.
    fn split(&mut self) -> Option<Self> {
        if self.splits == 0 {
            return None;
        }

        if let Some(split) = self.iter.split() {
            self.splits /= 2;

            Some(Self {
                iter: split,
                splits: self.splits,
            })
        } else {
            None
        }
    }

    /// Bridge to an [`UnindexedConsumer`].
    ///
    /// [`UnindexedConsumer`]: struct@rayon::iter::plumbing::UnindexedConsumer
    fn bridge<C>(&mut self, stolen: bool, consumer: C) -> C::Result
    where
        Iter: Send,
        C: UnindexedConsumer<Iter::Item>,
    {
        // Thief-splitting: start with enough splits to fill the thread pool,
        // and reset every time a job is stolen by another thread.
        if stolen {
            self.splits = current_num_threads();
        }

        let mut folder = consumer.split_off_left().into_folder();

        if self.splits == 0 {
            return folder.consume_iter(&mut self.iter).complete();
        }

        while !folder.full() {
            // Try to split
            if let Some(mut split) = self.split() {
                let (r1, r2) = (consumer.to_reducer(), consumer.to_reducer());
                let left_consumer = consumer.split_off_left();

                let (left, right) = join_context(
                    |ctx| self.bridge(ctx.migrated(), left_consumer),
                    |ctx| split.bridge(ctx.migrated(), consumer),
                );
                return r1.reduce(folder.complete(), r2.reduce(left, right));
            }

            // Otherwise, consume an item and try again
            if let Some(next) = self.iter.next() {
                folder = folder.consume(next);
            } else {
                break;
            }
        }

        folder.complete()
    }
}

impl SplittableIterator for TetrisExplorer {
    fn split(&mut self) -> Option<Self> {
        let queue_len = self.queue.len();
        if queue_len >= QUEUE_SPLIT_SIZE {
            let split = self.queue.split_off(queue_len / 2);
            Some(Self {
                queue: split,
                visited: self.visited.clone(),
                max_depth: self.max_depth,
                filter: self.filter,
            })
        } else {
            None
        }
    }
}

impl rayon::iter::IntoParallelIterator for TetrisExplorer {
    type Iter = ParallelSplittableIterator<Self>;
    type Item = TetrisExplorerItemMegaBatch;

    fn into_par_iter(self) -> Self::Iter {
        ParallelSplittableIterator::new(self)
    }
}

impl<Iter> ParallelIterator for ParallelSplittableIterator<Iter>
where
    Iter: SplittableIterator + Send,
    Iter::Item: Send,
{
    type Item = Iter::Item;

    fn drive_unindexed<C>(mut self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        self.bridge(false, consumer)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub struct TetrisExplorerNode {
    pub prev_game: Option<TetrisGame>,
    pub placement: Option<TetrisPiecePlacement>,
    pub game: TetrisGame,
}

impl TetrisExplorerNode {
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            prev_game: None,
            placement: None,
            game: TetrisGame::new_with_seed(seed),
        }
    }

    pub fn new_with_game(game: TetrisGame) -> Self {
        Self {
            prev_game: None,
            placement: None,
            game,
        }
    }

    pub fn new(prev_game: TetrisGame, placement: TetrisPiecePlacement, game: TetrisGame) -> Self {
        Self {
            prev_game: Some(prev_game),
            placement: Some(placement),
            game,
        }
    }

    fn add_children_batch(
        self,
        depth: usize,
        queue: &mut VecDeque<TetrisExplorerItemBatch>,
        filter: Option<fn(&TetrisExplorerNode) -> bool>,
    ) {
        let mut buf_guard = CHILD_NODE_VEC_POOL.get_lock();
        buf_guard.clear();

        for placement in self.game.current_placements() {
            let mut next_game = self.game;
            let is_lost = next_game.apply_placement(*placement);
            if is_lost.into() || !filter.map_or(true, |f| f(&self)) {
                continue;
            }
            buf_guard.push(TetrisExplorerItem {
                depth,
                node: TetrisExplorerNode::new(self.game, *placement, next_game),
            });
        }

        queue.push_back(TetrisExplorerItemBatch { items: *buf_guard });
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TetrisExplorerItem {
    pub depth: usize,
    pub node: TetrisExplorerNode,
}

#[derive(Debug, Clone, Copy)]
pub struct TetrisExplorerItemBatch {
    pub items: HeaplessVec<TetrisExplorerItem, CHILD_NODE_BUFFER_SIZE>,
}

pub struct TetrisExplorerItemMegaBatch {
    pub items: HeaplessVec<TetrisExplorerItem, MEGA_BATCH_SIZE>,
}

pub struct TetrisExplorer {
    pub visited: Arc<TetrisBloom1>,

    pub queue: VecDeque<TetrisExplorerItemBatch>,

    pub max_depth: Option<usize>,
    pub filter: Option<fn(&TetrisExplorerNode) -> bool>,
}

impl TetrisExplorer {
    pub fn new_with_seed(seed: u64, max_depth: Option<usize>) -> Self {
        Self::new(TetrisExplorerNode::new_with_seed(seed), max_depth)
    }

    pub fn new(root: TetrisExplorerNode, max_depth: Option<usize>) -> Self {
        let visited = Arc::new(TetrisBloom1::new());
        let mut queue = VecDeque::<TetrisExplorerItemBatch>::new();
        let mut items = HeaplessVec::new();
        items.push(TetrisExplorerItem {
            depth: 0,
            node: root,
        });
        queue.push_back(TetrisExplorerItemBatch { items });
        Self {
            visited,
            queue,
            max_depth,
            filter: None,
        }
    }

    /// Add a stateless filter function that will be applied to all items yielded by the explorer
    pub fn with_children_filter(mut self, filter: fn(&TetrisExplorerNode) -> bool) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a stateless filter function that will be applied to all items yielded by the explorer
    pub fn set_children_filter(&mut self, filter: fn(&TetrisExplorerNode) -> bool) {
        self.filter = Some(filter);
    }

    /// Remove the current filter
    pub fn clear_children_filter(&mut self) {
        self.filter = None;
    }
}

impl Iterator for TetrisExplorer {
    type Item = TetrisExplorerItemMegaBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut iter_buf = ITER_VEC_POOL.get_lock();
        iter_buf.clear();

        while iter_buf.len() < MEGA_BATCH_SIZE - MAX_PIECE_PLACEMENT_COUNT - 1 {
            match self.queue.pop_front() {
                Some(item_batch) => {
                    let mut new_items = item_batch.items;
                    new_items.retain(|item| {
                        if let Some(max_depth) = self.max_depth {
                            if item.depth > max_depth {
                                return false;
                            }
                        }

                        let is_visited = self.visited.check_and_mark(item.node.game.board().into());
                        if is_visited {
                            return false;
                        }

                        item.node
                            .add_children_batch(item.depth + 1, &mut self.queue, self.filter);
                        true
                    });
                    iter_buf.fill_from(&mut new_items);
                }
                None => {
                    break;
                }
            }
        }

        if iter_buf.len() == 0 {
            None
        } else {
            Some(TetrisExplorerItemMegaBatch { items: *iter_buf })
        }

        // return ITER_BUFFER.with(|iter_buffer| {
        //     return OVERFLOW_BUFFER.with(|remainder_buffer| {
        //         let mut iter_buf = iter_buffer.borrow_mut();
        //         iter_buf.clear();

        //         // Empty the remainder buffer into the iter buffer
        //         iter_buf.fill_from(&mut remainder_buffer.borrow_mut());

        //         // Keep filling the iter buffer until we reach min_batch_size
        //         while !iter_buf.is_full() {
        //             match self.queue.pop_front() {
        //                 Some(item_batch) => {
        //                     let mut new_items = item_batch.items;
        //                     new_items.retain(|item| {
        //                         if let Some(max_depth) = self.max_depth {
        //                             if item.depth > max_depth {
        //                                 return false;
        //                             }
        //                         }

        //                         let is_visited =
        //                             self.visited.check_and_mark(item.node.game.board().into());
        //                         if is_visited {
        //                             return false;
        //                         }

        //                         item.node.add_children_batch(
        //                             item.depth + 1,
        //                             &mut self.queue,
        //                             self.filter,
        //                         );
        //                         true
        //                     });

        //                     iter_buf.fill_from(&mut new_items);
        //                     if iter_buf.is_full() {
        //                         remainder_buffer.borrow_mut().fill_from(&mut iter_buf);
        //                     }
        //                 }
        //                 None => {
        //                     break;
        //                 }
        //             }
        //         }

        //         if iter_buf.len() == 0 {
        //             None
        //         } else {
        //             Some(TetrisExplorerItemMegaBatch { items: *iter_buf })
        //         }
        //     });
        // });

        // let mut accumulated_item_batch = TetrisExplorerItemBatch { items: SimpleBuffer::new() };

        // // Keep processing batches until we reach min_batch_size or run out of batches
        // while accumulated_item_batch.items.len() < self.min_batch_size {
        //     match self.queue.pop_front() {
        //         Some(item_batch) => {
        //             let mut new_items = item_batch.items;
        //             new_items.retain(|item| {
        //                 if let Some(max_depth) = self.max_depth {
        //                     if item.depth >= max_depth {
        //                         return false;
        //                     }
        //                 }

        //                 let is_visited = self.visited.check_and_mark(item.node.game.board().into());
        //                 if is_visited {
        //                     return false;
        //                 }

        //                 item.node
        //                     .add_children_batch(item.depth + 1, &mut self.queue, self.filter);
        //                 true
        //             });

        //             accumulated_item_batch.items.append(&mut new_items);
        //         }
        //         None => {
        //             // No more batches available
        //             break;
        //         }
        //     }
        // }

        // if accumulated_item_batch.size() == 0 {
        //     None
        // } else {
        //     Some(accumulated_item_batch)
        // }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    const GAME_SEED: u64 = 12345;

    #[test]
    fn test_explorer_count_items_non_parallel() {
        let depth = 1;
        let explorer = TetrisExplorer::new_with_seed(GAME_SEED, Some(depth));
        let total_items = explorer.map(|item| item.items.len()).sum::<usize>();
        assert!(total_items == 18, "Should find 18 items");
    }

    #[test]
    fn test_explorer_find_line_clear_state_non_parallel() {
        let explorer = TetrisExplorer::new_with_seed(GAME_SEED, Some(32));

        let success_state = |item: &TetrisExplorerItem| {
            let lines_cleared = item.node.game.lines_cleared;
            let height = item.node.game.board().height();
            lines_cleared == 1 && height < 15
        };

        let target_state = explorer
            .filter_map(|item| item.items.find(|x| success_state(&x)))
            .next();
        assert!(target_state.is_some(), "Should find a target state");
    }

    #[test]
    fn test_explorer_find_line_clear_state_parallel() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let success_state = |item: &TetrisExplorerItem| item.node.game.piece_count > 8;
        let target_state = pool.install(|| {
            TetrisExplorer::new_with_seed(GAME_SEED, Some(30))
                .with_children_filter(|node| node.game.board().height() < 4)
                .into_par_iter()
                .filter_map(|batch| batch.items.find(|item| success_state(item)))
                .find_any(|_| true)
        });
        assert!(target_state.is_some(), "Should find a target state");
    }
}
