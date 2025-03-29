use borsh::{BorshDeserialize, BorshSerialize, from_slice, to_vec};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::ops::RangeBounds;
use std::{
    collections::{HashSet, VecDeque},
    hash::{Hash, Hasher},
    ops::Bound,
    sync::{Arc, RwLock},
};

use itertools::Itertools;

use crate::tetris_board::{
    BoardRaw, COLS_U8, Column, PiecePlacement, Rotation, TetrisBoard, TetrisPiece, TetrisPieceBag,
};
use rayon::iter::ParallelIterator;
use rayon::iter::plumbing::{Folder, Reducer, UnindexedConsumer};
use rayon::{current_num_threads, join_context};
use std::iter::Iterator;

use std::range::Bound::{Excluded, Included};

use dashmap::DashSet;

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

impl SplittableIterator for AtlasSearch {
    fn split(&mut self) -> Option<Self> {
        let len = self.queue.len();
        if len >= 2 {
            let split = self.queue.split_off(len / 2);
            Some(Self {
                queue: split,
                visited: self.visited.clone(),
                atlas: self.atlas.clone(),
                max_depth: self.max_depth,
            })
        } else {
            None
        }
    }
}

impl rayon::iter::IntoParallelIterator for AtlasSearch {
    type Iter = ParallelSplittableIterator<Self>;
    type Item = AtlasSearchItem;

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

/// The key used to index the tetris atlas composed of (board, bag, placement).
///
/// This is all the information needed to calculate the next tetris board.
#[derive(
    Debug, Hash, PartialEq, Eq, Clone, Default, PartialOrd, BorshSerialize, BorshDeserialize,
)]
pub struct AtlasKey(pub BoardRaw, pub TetrisPieceBag, pub PiecePlacement);

impl Ord for AtlasKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.cmp(&other.0) {
            std::cmp::Ordering::Equal => match self.1.cmp(&other.1) {
                std::cmp::Ordering::Equal => self.2.cmp(&other.2),
                ordering => ordering,
            },
            ordering => ordering,
        }
    }
}

impl AtlasKey {
    pub fn from_board(board: BoardRaw) -> Self {
        Self(board, TetrisPieceBag::default(), PiecePlacement::default())
    }

    /// Get the range so we can query the atlas for all entries
    /// that use the same board.
    pub fn board_range_query(&self) -> AtlasKeyRange {
        AtlasKeyRange {
            start: AtlasKey(
                self.0.clone(),
                TetrisPieceBag::default(),
                PiecePlacement::default(),
            ),
            end: {
                let mut also_board = self.0;
                AtlasKey(
                    *also_board.next_mut(),
                    TetrisPieceBag::default(),
                    PiecePlacement::default(),
                )
            },
        }
    }

    /// Get the range so we can query the atlas for all entries
    /// that use the same board and bag.
    pub fn board_bag_range_query(&self) -> AtlasKeyRange {
        let mut bag = self.1.clone();
        bag.inc();
        AtlasKeyRange {
            start: AtlasKey(self.0.clone(), self.1.clone(), PiecePlacement::default()),
            end: AtlasKey(self.0.clone(), bag, PiecePlacement::default()),
        }
    }
}

pub struct AtlasKeyRange {
    start: AtlasKey,
    end: AtlasKey,
}

impl RangeBounds<AtlasKey> for AtlasKeyRange {
    fn start_bound(&self) -> Bound<&AtlasKey> {
        Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&AtlasKey> {
        Excluded(&self.end)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Default)]
pub struct AtlasNode {
    pub board: TetrisBoard,
    pub bag: TetrisPieceBag,
}

impl AtlasNode {
    pub fn to_key(&self) -> AtlasKey {
        AtlasKey(self.board.play_board, self.bag, PiecePlacement::default())
    }

    /// Take the current node, iterate through every
    pub fn children(&self) -> impl Iterator<Item = (AtlasNode, PiecePlacement)> {
        (!self.board.loss())
            .then(|| {
                self.bag
                    .next_bags()
                    .flat_map(move |(next_bag, next_piece)| {
                        PiecePlacement::all_from_piece(next_piece)
                            .map(move |placement| (next_bag, placement))
                    })
            })
            .into_iter()
            .flatten()
            .map(|(next_bag, placement)| {
                let mut new_board = self.board.clone();
                new_board.play_piece(placement);
                (
                    AtlasNode {
                        board: new_board,
                        bag: next_bag,
                    },
                    placement,
                )
            })
    }

    fn add_children(
        &self,
        depth: usize,
        queue: &mut VecDeque<(usize, Self, Vec<(AtlasNode, PiecePlacement)>)>,
        path: Vec<(AtlasNode, PiecePlacement)>,
    ) {
        queue.extend(self.children().map(|(child_node, placement)| {
            let mut p = path.clone();
            p.push((self.clone(), placement));
            (depth, child_node, p)
        }));
        // for (child_node, placement) in self.children() {
        //     queue.push_back((depth, child_node, {
        //         let mut p = path.clone();
        //         p.push((self.clone(), placement));
        //         p
        //     }));
        // }
    }
}

pub struct Atlas {
    pub inner: BTreeSet<AtlasKey>,
}

impl Atlas {
    pub fn new() -> Self {
        Self {
            inner: BTreeSet::new(),
        }
    }

    pub fn save_atlas(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        let mut writer = BufWriter::new(&mut file);

        for key in self.inner.iter() {
            writer.write_all(&to_vec(&key).unwrap()).unwrap();
        }
        writer.flush().unwrap();
    }

    pub fn load_atlas(path: &str) -> Self {
        let file = File::open(path).unwrap();

        let sizeof_atlas_key = std::mem::size_of::<AtlasKey>();
        let mut atlas = BTreeSet::new();
        let mut reader = BufReader::new(file);

        // Read the entire file into memory
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();

        // Process the buffer in chunks
        let mut i = 0;
        while i + sizeof_atlas_key <= buffer.len() {
            let chunk = &buffer[i..i + sizeof_atlas_key];
            match from_slice::<AtlasKey>(chunk) {
                Ok(key) => {
                    atlas.insert(key);
                }
                Err(e) => {
                    eprintln!("Error deserializing at position {}: {}", i, e);
                    // Try to recover by incrementing by a smaller amount
                    i += 1;
                    continue;
                }
            }
            i += sizeof_atlas_key;
        }

        println!("Loaded {} atlas keys", atlas.len());
        Self { inner: atlas }
    }

    pub fn interactive_traverse(&self) {
        let mut current_node = AtlasNode::default();
        loop {
            // get all the next options
            let query = current_node.to_key();
            let options = self
                .inner
                .range(query.board_bag_range_query())
                .collect_vec();

            println!("Current board: {}", current_node.board);
            println!("Current bag: {}", current_node.bag);
            println!("Num options: {}", options.len());
            options.iter().enumerate().for_each(|(i, e)| {
                let placement = e.2;
                println!("Option {}: {}", i, placement);
            });

            println!("Enter option number: ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input: usize = input.trim().parse().unwrap();

            // Create a new TetrisBoard from the BoardRaw
            let option = options[input];
            let placement = option.2;
            let mut bag = option.1;
            bag.take(placement.piece);
            println!("Playing piece: {}", placement.piece);
            println!("Result bag: {}", bag);
            current_node.board.play_piece(placement);

            assert!(
                current_node.bag.contains(placement.piece),
                "Bag does not contain piece"
            );
            current_node.bag = bag;
        }
    }
}

#[derive(Clone)]
pub struct AtlasSearch {
    pub visited: Arc<DashSet<AtlasNode>>,
    pub atlas: Arc<RwLock<Atlas>>,
    pub queue: VecDeque<(usize, AtlasNode, Vec<(AtlasNode, PiecePlacement)>)>,
    pub max_depth: Option<usize>,
}

impl AtlasSearch {
    pub fn new(root: AtlasNode, max_depth: Option<usize>) -> Self {
        let visited = Arc::new(DashSet::new());
        let atlas = Arc::new(RwLock::new(Atlas::new()));
        let mut queue = VecDeque::<(usize, AtlasNode, Vec<(AtlasNode, PiecePlacement)>)>::new();
        queue.push_back((0, root, Vec::with_capacity(max_depth.unwrap())));
        Self {
            visited,
            atlas,
            queue,
            max_depth,
        }
    }
}

#[derive(Debug)]
pub struct AtlasSearchItem {
    pub depth: usize,
    pub node: AtlasNode,
    pub path: Vec<(AtlasNode, PiecePlacement)>,
}

impl Iterator for AtlasSearch {
    type Item = AtlasSearchItem;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.queue.pop_front() {
            Some((depth, node, path)) => {
                if let Some(max_depth) = self.max_depth {
                    if depth >= max_depth {
                        return Some(AtlasSearchItem { depth, node, path });
                    }
                }

                if node.board.line_height() > 8 {
                    return None;
                }

                // if we are at a happy node, we add it to the atlas
                if node.board.line_height() <= 4 {
                    let mut atlas = self.atlas.write().unwrap();
                    path.iter().for_each(|&(node, placement)| {
                        atlas
                            .inner
                            .insert(AtlasKey(node.board.play_board, node.bag, placement));
                    });

                    // Start a new path from the current node
                    // self.visited.insert(node);
                    node.add_children(
                        0,
                        &mut self.queue,
                        Vec::with_capacity(self.max_depth.unwrap()),
                    );
                    return Some(AtlasSearchItem { depth, node, path });

                    // let in_atlas = self
                    //     .atlas
                    //     .read()
                    //     .unwrap()
                    //     .inner
                    //     .range(node.to_key().board_bag_range_query())
                    //     .next()
                    //     .is_some();
                    // if in_atlas {
                    //     return None;
                    // } else {
                    //     let mut atlas = self.atlas.write().unwrap();
                    //     path.iter().for_each(|&(node, placement)| {
                    //         atlas.inner.insert(AtlasKey(
                    //             node.board.play_board,
                    //             node.bag,
                    //             placement,
                    //         ));
                    //     });

                    //     // Start a new path from the current node
                    //     self.visited.insert(node);
                    //     node.add_children(
                    //         0,
                    //         &mut self.queue,
                    //         Vec::with_capacity(self.max_depth.unwrap()),
                    //     );
                    //     return Some(AtlasSearchItem { depth, node, path });
                    // }
                }

                // // If the current node being searched is already in the atlas,
                // // we can insert our current path also into the atlas
                // // we then make sure we add all current nodes up to this point
                // // to the visited set
                // let in_atlas = self
                //     .atlas
                //     .read()
                //     .unwrap()
                //     .inner
                //     .range(AtlasKey::from_state(node.state).board_bag_range_query())
                //     .next()
                //     .is_some();
                // if in_atlas {
                //     let mut atlas = self.atlas.write().unwrap();
                //     path.iter().for_each(|p: &(AtlasNode, PiecePlacement)| {
                //         atlas.inner.insert(AtlasKey {
                //             board: p.0.state.board.play_board,
                //             bag: p.0.state.bag,
                //             placement: p.1,
                //         });
                //     });
                // }

                // if self.visited.contains(&node) {
                //     return None;
                // } else {
                //     self.visited.insert(node);
                //     node.add_children(depth + 1, &mut self.queue, path.clone());
                //     return Some(AtlasSearchItem { depth, node, path });
                // }

                // self.visited.insert(node);
                node.add_children(depth + 1, &mut self.queue, path.clone());
                return Some(AtlasSearchItem { depth, node, path });
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_query() {
        // test ord
        let a = AtlasKey::default();

        let mut b = AtlasKey::default();
        b.0 = BoardRaw::default().next();
        assert!(a < b);

        let mut c = AtlasKey::default();
        c.0 = BoardRaw::default().next().next();
        assert!(b < c);
    }
}
