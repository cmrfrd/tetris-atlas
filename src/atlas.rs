use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
};

use crate::{
    par::parallel_iterator,
    tetris_board::{BoardRaw, Rotation, TetrisBoard, TetrisPiece, TetrisPieceBag, COLS_U8},
};

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Column(u8);

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct PiecePlacement {
    pub piece: TetrisPiece,
    pub rotation: Rotation,
    pub column: Column,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AtlasNode {
    pub state: TetrisState,
    // pub placement: PiecePlacement,
}

impl Hash for AtlasNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.board.hash(state);
    }
}

impl AtlasNode {
    pub fn children(&self) -> impl Iterator<Item = AtlasNode> {
        (!self.state.board.loss())
            .then(|| self.state.all_next_placements())
            .into_iter()
            .flatten()
            .map(|(next_bag, placement)| {
                let mut new_board = self.state.board.clone();
                new_board.play_piece(placement.piece, placement.rotation, placement.column.0);
                AtlasNode {
                    state: TetrisState {
                        board: new_board,
                        bag: next_bag,
                    },
                }
            })
    }
}

pub trait Node
where
    Self: Hash + Eq + Sized + std::fmt::Debug + Clone,
{
    fn add_children(&self, depth: usize, queue: &mut VecDeque<(usize, Self)>);
    fn insertable(&self) -> bool;
}

impl Node for AtlasNode {
    fn add_children(&self, depth: usize, queue: &mut VecDeque<(usize, Self)>) {
        queue.extend(self.children().map(|child| (depth, child)));
    }

    #[inline(always)]
    fn insertable(&self) -> bool {
        self.state.board.happy_state()
    }
}

pub struct Dfs<N>
where
    N: Node,
{
    // pub board_to_bag: Arc<RwLock<HashMap<BoardRaw, TetrisPieceBag>>>,
    pub visited: Arc<RwLock<HashSet<N>>>,
    pub queue: VecDeque<(usize, N)>,
    pub max_depth: Option<usize>,
}

impl<N> Dfs<N>
where
    N: Node,
{
    #[inline]
    pub fn new<R, D>(root: R, max_depth: D) -> Self
    where
        R: Into<N>,
        D: Into<Option<usize>>,
    {
        let visited = Arc::new(RwLock::new(HashSet::new()));
        let mut queue = VecDeque::<(usize, N)>::new();
        let root: N = root.into();
        let max_depth = max_depth.into();
        queue.push_back((0, root.clone()));
        root.add_children(1, &mut queue);
        Self {
            visited,
            queue,
            max_depth,
        }
    }
}

impl<N> Iterator for Dfs<N>
where
    N: Node,
{
    type Item = N;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.queue.pop_front() {
            Some((depth, node)) => {
                if let Some(max_depth) = self.max_depth {
                    if depth >= max_depth {
                        return Some(node);
                    }
                }

                if self.visited.read().unwrap().contains(&node) {
                    return None;
                } else {
                    self.visited.write().unwrap().insert(node.clone());
                    node.add_children(depth + 1, &mut self.queue);
                    return Some(node);
                }

                // // During the search, another thread may have already visited a subchain
                // // of nodes. If so, we can skip the search for that subchain.
                // let subchain_visited = {
                //     let visited = self.visited.read().unwrap();
                //     self.queue
                //         .iter()
                //         .enumerate()
                //         .find(|(_, (_, n))| visited.contains(n))
                // };
                // if let Some((i, _)) = subchain_visited {
                //     self.visited
                //         .write()
                //         .unwrap()
                //         .extend(self.queue.iter().take(i).map(|(_, n)| n.clone()));
                //     return None;
                // }

                // // are we in a happy state?
                // let insertable = node.insertable();
                // let visited = self.visited.read().unwrap().contains(&node);

                // // if it's a happy state and it's visited, exit
                // // we don't need to explore it further
                // if insertable && visited {
                //     return None;
                // }

                // // if we've reached a happy state, and we haven't visited it yet,
                // // insert the whole chain into visited, and continue the search from that happy state
                // if insertable || visited {
                //     println!("Num visited: {}", self.visited.read().unwrap().len());
                //     println!("queue len: {}", self.queue.len());

                //     let mut visited = self.visited.write().unwrap();
                //     visited.insert(node.clone());
                //     visited.extend(self.queue.iter().map(|(_, node)| node.clone()));
                //     drop(visited);

                //     node.add_children(depth + 1, &mut self.queue);
                //     return Some(node);
                // }

                // // if it isn't a happy state,

                // // add children to queue
                // node.add_children(depth + 1, &mut self.queue);
                // Some(node)
            }
            None => None,
        }
    }
}

parallel_iterator!(Dfs<Node>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_entry() {
        let state = TetrisState {
            board: TetrisBoard::default(),
            bag: TetrisPieceBag::default(),
        };
        for (bag, placement) in state.all_next_placements() {
            println!(
                "bag: {:?}, piece: {:?}, rotation: {:?}, column: {:?}",
                bag, placement.piece, placement.rotation, placement.column
            );
        }
        println!("Num of placements: {}", state.all_next_placements().count());
    }
}
