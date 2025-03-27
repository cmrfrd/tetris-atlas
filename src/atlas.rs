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

use crate::tetris_board::{BoardRaw, COLS_U8, Rotation, TetrisBoard, TetrisPiece, TetrisPieceBag};
use rayon::iter::ParallelIterator;
use rayon::iter::plumbing::{Folder, Reducer, UnindexedConsumer};
use rayon::{current_num_threads, join_context};
use std::iter::Iterator;

use std::range::Bound::{Excluded, Included};

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

impl SplittableIterator for Dfs {
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

impl rayon::iter::IntoParallelIterator for Dfs {
    type Iter = ParallelSplittableIterator<Self>;
    type Item = DfsItem;

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

#[derive(
    Debug,
    Hash,
    PartialEq,
    Eq,
    Clone,
    Copy,
    Ord,
    PartialOrd,
    Default,
    BorshSerialize,
    BorshDeserialize,
)]
pub struct Column(pub u8);

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column({})", self.0)
    }
}

#[derive(
    Debug,
    Hash,
    PartialEq,
    Eq,
    Clone,
    Copy,
    Ord,
    PartialOrd,
    Default,
    BorshSerialize,
    BorshDeserialize,
)]
pub struct PiecePlacement {
    pub piece: TetrisPiece,
    pub rotation: Rotation,
    pub column: Column,
}

impl Display for PiecePlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PiecePlacement(piece: {}, rotation: {}, column: {})",
            self.piece, self.rotation, self.column
        )
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisState {
    pub board: TetrisBoard,
    pub bag: TetrisPieceBag,
}

impl Default for TetrisState {
    fn default() -> Self {
        Self {
            board: TetrisBoard::default(),
            bag: TetrisPieceBag::default(),
        }
    }
}

impl TetrisState {
    pub fn all_next_placements(
        &self,
    ) -> impl Iterator<Item = (TetrisPieceBag, PiecePlacement)> + '_ {
        self.bag
            .next_bags()
            .flat_map(move |(next_bag, next_piece)| {
                (0..next_piece.num_rotations()).flat_map(move |r| {
                    (0..=(COLS_U8 - next_piece.width(Rotation(r)))).map(move |col| {
                        (
                            next_bag,
                            PiecePlacement {
                                piece: next_piece,
                                rotation: Rotation(r),
                                column: Column(col),
                            },
                        )
                    })
                })
            })
    }
}

#[derive(
    Debug, Hash, PartialEq, Eq, Clone, Ord, PartialOrd, Default, BorshSerialize, BorshDeserialize,
)]
pub struct AtlasKey {
    pub board: BoardRaw,
    pub bag: TetrisPieceBag,
    pub placement: PiecePlacement,
}

impl AtlasKey {
    pub fn from_board(board: BoardRaw) -> Self {
        Self {
            board,
            bag: TetrisPieceBag::default(),
            placement: PiecePlacement::default(),
        }
    }

    pub fn from_state(state: TetrisState) -> Self {
        Self {
            board: state.board.play_board,
            bag: state.bag,
            placement: PiecePlacement::default(),
        }
    }

    pub fn board_range_query(&self) -> AtlasKeyRange {
        AtlasKeyRange {
            end: {
                let mut also_board = self.board.clone();
                also_board.next_mut();
                AtlasKey {
                    board: also_board,
                    bag: TetrisPieceBag::default(),
                    placement: PiecePlacement::default(),
                }
            },
            start: AtlasKey {
                board: self.board,
                bag: TetrisPieceBag::default(),
                placement: PiecePlacement::default(),
            },
        }
    }

    pub fn board_bag_range_query(&self) -> AtlasKeyRange {
        AtlasKeyRange {
            end: {
                let mut also_bag = self.bag.clone();
                also_bag.inc();
                AtlasKey {
                    board: self.board.clone(),
                    bag: also_bag,
                    placement: PiecePlacement::default(),
                }
            },
            start: AtlasKey {
                board: self.board,
                bag: self.bag,
                placement: PiecePlacement::default(),
            },
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct AtlasNode {
    pub state: TetrisState,
}

impl Hash for AtlasNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.board.hash(state);
    }
}

impl AtlasNode {
    pub fn children(&self) -> impl Iterator<Item = (AtlasNode, PiecePlacement)> {
        (!self.state.board.loss())
            .then(|| self.state.all_next_placements())
            .into_iter()
            .flatten()
            .map(|(next_bag, placement)| {
                let mut new_board = self.state.board.clone();
                new_board.play_piece(placement.piece, placement.rotation, placement.column.0);
                (
                    AtlasNode {
                        state: TetrisState {
                            board: new_board,
                            bag: next_bag,
                        },
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
        for (child, placement) in self.children() {
            queue.push_back((depth, child.clone(), {
                let mut p = path.clone();
                p.push((child, placement));
                p
            }));
        }
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
        let mut current_state = TetrisState::default();
        loop {
            // get all the next options
            let query = AtlasKey::from_state(current_state);
            let options = self
                .inner
                .range(query.board_bag_range_query())
                .collect_vec();

            println!("Current board: {}", current_state.board);
            println!("Current bag: {}", current_state.bag);
            println!("Num options: {}", options.len());
            options.iter().enumerate().for_each(|(i, e)| {
                println!("Option {}: {}", i, e.placement);
            });

            println!("Enter option number: ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input: usize = input.trim().parse().unwrap();

            // Create a new TetrisBoard from the BoardRaw
            current_state.board.play_piece(
                options[input].placement.piece,
                options[input].placement.rotation,
                options[input].placement.column.0,
            );
        }
    }
}

#[derive(Clone)]
pub struct Dfs {
    pub visited: Arc<RwLock<HashSet<AtlasNode>>>,
    pub atlas: Arc<RwLock<Atlas>>,
    pub queue: VecDeque<(usize, AtlasNode, Vec<(AtlasNode, PiecePlacement)>)>,
    pub max_depth: Option<usize>,
}

impl Dfs {
    pub fn new(root: AtlasNode, max_depth: Option<usize>) -> Self {
        let visited = Arc::new(RwLock::new(HashSet::new()));
        let atlas = Arc::new(RwLock::new(Atlas::new()));
        let mut queue = VecDeque::<(usize, AtlasNode, Vec<(AtlasNode, PiecePlacement)>)>::new();
        queue.push_back((0, root, Vec::with_capacity(max_depth.unwrap_or(8) + 1)));
        Self {
            visited,
            atlas,
            queue,
            max_depth,
        }
    }
}

#[derive(Debug)]
pub struct DfsItem {
    pub depth: usize,
    pub node: AtlasNode,
    pub path: Vec<(AtlasNode, PiecePlacement)>,
}

impl Iterator for Dfs {
    type Item = DfsItem;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.queue.pop_front() {
            Some((depth, node, path)) => {
                if let Some(max_depth) = self.max_depth {
                    if depth >= max_depth {
                        return Some(DfsItem { depth, node, path });
                    }
                }

                // if we are at a happy node, we add it to the atlas
                if node.state.board.happy_state() {
                    let mut atlas = self.atlas.write().unwrap();
                    path.iter().for_each(|p: &(AtlasNode, PiecePlacement)| {
                        atlas.inner.insert(AtlasKey {
                            board: p.0.state.board.play_board,
                            bag: p.0.state.bag,
                            placement: p.1,
                        });
                    });
                    self.visited.write().unwrap().insert(node);
                    // self.visited
                    //     .write()
                    //     .unwrap()
                    //     .extend(path.iter().map(|p| p.0));
                    node.add_children(depth + 1, &mut self.queue, path.clone());
                    return Some(DfsItem { depth, node, path });
                    // return None;s
                }

                if self.visited.read().unwrap().contains(&node) {
                    return None;
                }

                let in_atlas = self
                    .atlas
                    .read()
                    .unwrap()
                    .inner
                    .range(AtlasKey::from_state(node.state).board_bag_range_query())
                    .next()
                    .is_some();

                // If the current node being searched is already in the atlas,
                // we can insert our current path also into the atlas
                // we then make sure we add all current
                if in_atlas {
                    let mut atlas = self.atlas.write().unwrap();
                    path.iter().for_each(|p: &(AtlasNode, PiecePlacement)| {
                        atlas.inner.insert(AtlasKey {
                            board: p.0.state.board.play_board,
                            bag: p.0.state.bag,
                            placement: p.1,
                        });
                    });
                    self.visited.write().unwrap().insert(node);
                    // self.visited
                    //     .write()
                    //     .unwrap()
                    //     .extend(path.iter().map(|p| p.0));
                    node.add_children(depth + 1, &mut self.queue, path.clone());
                    return Some(DfsItem { depth, node, path });
                    // return None;
                }

                // Don't both revisiting explored nodes, someone
                // else is already exploring this node
                self.visited.write().unwrap().insert(node);
                node.add_children(depth + 1, &mut self.queue, path.clone());
                return Some(DfsItem { depth, node, path });
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::DefaultHasher;

    use super::*;

    #[test]
    fn test_atlas_entry() {
        let state = TetrisState {
            board: TetrisBoard::default(),
            bag: TetrisPieceBag::default(),
        };
        // for (bag, placement) in state.all_next_placements() {
        //     println!(
        //         "bag: {:?}, piece: {:?}, rotation: {:?}, column: {:?}",
        //         bag, placement.piece, placement.rotation, placement.column
        //     );
        // }
        // println!("Num of placements: {}", state.all_next_placements().count());
    }
}
