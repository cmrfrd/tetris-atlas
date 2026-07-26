#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_szt3 — exact AND-OR solver for the S/Z/T, 3-column Tetris variant.
//!
//! # Why this exists
//!
//! The full 10-column, 7-piece game has resisted every exact-closure attempt (the
//! carrier wall: >5e6 states in every probe). This binary asks the same question —
//! "does a strategy exist that survives every adversarial bag order forever?" — on a
//! variant small enough to answer EXACTLY: **3 columns, bag = {S, Z, T}**. S and Z are
//! the roughness/hole makers, T is the only charged piece (PieceCharge); a 3-wide well
//! is the minimal arena where all three have both orientations available. Per 3-piece
//! bag the variant places 12 cells = exactly 4 full rows, so survival demands a
//! sustained 4 clears/bag — the clearing-equilibrium question in miniature.
//!
//! # The variant is a real `TetrisGameConfig`
//!
//! The engine (`tetris-game`) is generic over board dimensions and piece set:
//! `apply_piece_placement`, `clear_filled_rows`, `is_lost`, `heights` all key off
//! `C::COLS` / `C::ROWS`. This binary defines `SztCols3<ROWS>` implementing
//! `TetrisGameConfig` (COLS = 3, PIECE_SET = {S,Z,T}) and plays on
//! `TetrisBoard<SztCols3<R>>` directly — the ONLY 10-column-specific thing in the
//! engine is the precomputed placement table, which we filter by piece width.
//!
//! # What it computes
//!
//! State = `(exact board, remaining-bag mask)`; the bag refills to {S,Z,T} when it
//! empties (the 7-bag rule at bag size 3). Two opponent models over the same graph:
//!
//!   * **adv**  — the adversary draws any remaining piece (AND), the player answers
//!     with any legal placement (OR). The greatest fixed point (retrograde death
//!     propagation over the reachable graph) decides the game exactly.
//!   * **coop** — the player also chooses which remaining piece to place (OR over
//!     both). If even this dies, the variant is unplayable regardless of luck.
//!
//! Verdicts per loss line `ROWS` (swept via CLI, each a separate monomorphized
//! config): ALIVE is monotone in ROWS (a strategy that never exceeds height R is
//! valid under any taller ceiling), so the first ALIVE settles the variant playable
//! forever; DEAD at the canonical ceiling settles it unsolvable. On a budget blowup
//! the frontier is treated as dead, which keeps an ALIVE verdict sound and makes a
//! DEAD verdict inconclusive (reported as EXPLODED).
//!
//! When ALIVE, the surviving set is re-verified as a closed carrier by replaying
//! every (state, piece) obligation through the engine from scratch. When DEAD,
//! `--trace` prints a forced-loss line (adversary's killing piece each step, the
//! player's most resistant reply).
//!
//! # One-sided levers (each keeps exactly one verdict sound)
//!
//!   * `--max-holes K` restricts the PLAYER to ≤K buried holes: ALIVE stays sound
//!     for the unrestricted game, DEAD becomes band-relative.
//!   * `--adversary sz-first|greedy` scripts a single deterministic adversary:
//!     DEAD stays sound for the full game, ALIVE becomes a necessary-condition pass.
//!   * On budget explosion, both frontier polarities are solved: pessimistic
//!     (frontier dead) makes ALIVE sound, optimistic (frontier immortal) makes DEAD
//!     sound; only if they disagree is the run inconclusive.
//!
//! `--variant sztj5` switches to the second config: 5 columns, bag = {S, Z, T, J}
//! (16 cells/bag = 3.2 clears/bag demanded; J is the flattening repair piece).
//!
//! The `--engine kill-dfs` prover breaks the GFP memory wall for the DEAD direction:
//! iterative-deepening AND-OR DFS over the adversary's forcing subtree with two
//! path-independent memos (proven kill depths, proven kill-free depths — both
//! depth-indexed, so no graph-history-interaction handling is needed). It settled
//! sztj5 ROWS=6/7/8 in seconds-to-minutes where breadth-first needed billions of
//! states.
//!
//! Run:
//!   cargo run --release -p tetris-playground --bin tetris_szt3
//!   cargo run --release -p tetris-playground --bin tetris_szt3 -- --rows 6,8,10 --trace
//!   cargo run --release -p tetris-playground --bin tetris_szt3 -- --variant sztj5 --max-holes 2
//!   cargo run --release -p tetris-playground --bin tetris_szt3 -- --variant sztj5 \
//!     --engine kill-dfs --rows 8 --kill-depth 30 --budget 300000000

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Parser;
use rustc_hash::FxHashMap;
use tetris_game::{
    TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement,
};

// ---------------------------------------------------------------------------
// The variant config: extending TetrisGameConfig to a 3-column, 3-piece game
// ---------------------------------------------------------------------------

/// 3-column Tetris with bag = {S, Z, T} and the loss line at `ROWS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SztCols3<const ROWS: usize>;

impl<const ROWS: usize> TetrisGameConfig for SztCols3<ROWS> {
    const ROWS: usize = ROWS;
    const COLS: usize = 3;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::from_pieces([
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
    ]);
}

/// 5-column Tetris with bag = {S, Z, T, J} and the loss line at `ROWS`. Adds J — a
/// flattening piece that can repair S/Z damage — and two more columns; one bag is
/// 16 cells = 3.2 rows, so survival needs 3.2 clears/bag on average.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SztjCols5<const ROWS: usize>;

impl<const ROWS: usize> TetrisGameConfig for SztjCols5<ROWS> {
    const ROWS: usize = ROWS;
    const COLS: usize = 5;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::from_pieces([
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
        TetrisPiece::J_PIECE,
    ]);
}

/// All placements of `piece` that fit inside `C::COLS` columns, filtered from the
/// engine's (10-column) placement table. Placement application itself is fully
/// generic — `apply_piece_placement` slices `[column..column+width]` — so fitting
/// horizontally is the only extra condition a narrow board imposes.
fn placements_for<C: TetrisGameConfig>() -> [Vec<TetrisPiecePlacement>; 7] {
    let mut out: [Vec<TetrisPiecePlacement>; 7] = Default::default();
    for piece in TetrisPiece::all() {
        for &pl in TetrisPiecePlacement::all_from_piece(piece) {
            let width = pl.piece.width(pl.orientation.rotation) as usize;
            if pl.orientation.column.index() + width <= C::COLS {
                out[piece.index() as usize].push(pl);
            }
        }
    }
    out
}

/// Iterate the pieces present in a remaining-bag mask, in canonical piece order.
fn pieces_in_mask(mask: u8) -> impl Iterator<Item = TetrisPiece> {
    TetrisPiece::all()
        .into_iter()
        .filter(move |&p| mask & u8::from(p) != 0)
}

/// Which piece(s) the adversary may present from the current bag.
///
/// `All` is the true AND-quantifier. The scripted policies pick ONE piece as a pure
/// function of `(board, bag)` — a single valid adversary strategy, so a player loss
/// against it is a sound DEAD for the full game, while a survival is only a
/// necessary-condition pass (the real adversary is stronger).
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
enum Adversary {
    /// Full adversary: every remaining piece is an AND-obligation.
    All,
    /// Fixed preference S, Z, T, J, O, I, L: the hole-forcers lead every bag.
    SzFirst,
    /// 1-ply minimax: present the piece whose BEST player reply is worst
    /// (holes, then height, then roughness, then mass).
    Greedy,
}

/// A game state: exact board + remaining-bag mask.
type StateKey<C> = (TetrisBoard<C>, u8);
/// A player reply: (survival eval, successor state), kept sorted best-first.
type Reply<C> = (i64, StateKey<C>);

/// Survival eval (lower = better for the player): hole-averse, then low, then flat.
fn eval_board<C: TetrisGameConfig>(b: &TetrisBoard<C>) -> i64
where
    [(); C::COLS]:,
{
    let heights = b.heights();
    (b.total_holes() as i64) * 1_000_000_000
        + (b.height() as i64) * 1_000_000
        + (b.roughness() as i64) * 1_000
        + heights.iter().map(|&h| h as i64).sum::<i64>()
}

/// The adversary's piece choices at a state (1 piece for scripted policies).
fn adversary_choices<C: TetrisGameConfig>(
    board: &TetrisBoard<C>,
    bag: u8,
    placements: &[Vec<TetrisPiecePlacement>; 7],
    adversary: Adversary,
) -> Vec<TetrisPiece>
where
    [(); C::COLS]:,
{
    match adversary {
        Adversary::All => pieces_in_mask(bag).collect(),
        Adversary::SzFirst => {
            const PREF: [TetrisPiece; 7] = [
                TetrisPiece::S_PIECE,
                TetrisPiece::Z_PIECE,
                TetrisPiece::T_PIECE,
                TetrisPiece::J_PIECE,
                TetrisPiece::O_PIECE,
                TetrisPiece::I_PIECE,
                TetrisPiece::L_PIECE,
            ];
            PREF.into_iter()
                .find(|&p| bag & u8::from(p) != 0)
                .into_iter()
                .collect()
        }
        Adversary::Greedy => {
            let mut best_piece: Option<TetrisPiece> = None;
            let mut best_score = i64::MIN;
            for piece in pieces_in_mask(bag) {
                // The player's best reply value; no surviving reply is +inf for us.
                let mut reply = i64::MAX;
                for &pl in &placements[piece.index() as usize] {
                    let mut b = *board;
                    let res = b.apply_piece_placement(pl);
                    if bool::from(res.is_lost) {
                        continue;
                    }
                    reply = reply.min(eval_board(&b));
                }
                if reply > best_score {
                    best_score = reply;
                    best_piece = Some(piece);
                }
            }
            best_piece.into_iter().collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Reachable AND-OR game graph
// ---------------------------------------------------------------------------

/// One OR-node: the player's placement options answering one drawable piece.
#[derive(Clone, Copy)]
struct Group {
    /// `TetrisPiece::index()` of the piece this group answers.
    piece: u8,
    /// Range into `Graph::edges`.
    edge_start: u32,
    edge_len: u8,
}

/// The exact reachable game graph from `(empty board, full bag)`.
///
/// Node ids are dense; node 0 is the initial state. Nodes with id `>= expanded`
/// were interned but never expanded (budget hit) and own no groups.
struct Graph<C: TetrisGameConfig>
where
    [(); C::COLS]:,
{
    keys: Vec<(TetrisBoard<C>, u8)>,
    /// Number of nodes actually expanded (== keys.len() unless `exploded`).
    expanded: usize,
    exploded: bool,
    /// Per-node range into `groups` (len = expanded + 1).
    group_start: Vec<u32>,
    groups: Vec<Group>,
    /// Owning node of each group.
    owner: Vec<u32>,
    /// Deduped successor node ids, grouped per `Group`.
    edges: Vec<u32>,
    /// Reverse CSR over `edges`: for each node, the groups with an edge into it.
    rev_start: Vec<u32>,
    rev_data: Vec<u32>,
    placements: [Vec<TetrisPiecePlacement>; 7],
    full_bag: u8,
    adversary: Adversary,
}

impl<C: TetrisGameConfig> Graph<C>
where
    [(); C::COLS]:,
{
    /// Forward-BFS the reachable state space, recording every (piece, placement)
    /// transition. Placements that top out become missing edges (the OR-node just
    /// has fewer children); a group left empty is an immediately-dead obligation.
    ///
    /// `max_holes` is a PLAYER-side band: placements landing on a board with more
    /// buried holes are treated as unavailable. Restricting the player only shrinks
    /// its strategy space, so an ALIVE verdict stays sound for the unrestricted
    /// game; a DEAD verdict inside a band is inconclusive. Dually, a scripted
    /// `adversary` keeps DEAD sound and makes ALIVE inconclusive.
    fn build(budget: usize, max_holes: Option<u32>, adversary: Adversary) -> Self {
        let placements = placements_for::<C>();
        let full_bag: u8 = C::PIECE_SET.into();
        let init = (TetrisBoard::<C>::new(), full_bag);

        let mut intern: FxHashMap<(TetrisBoard<C>, u8), u32> = FxHashMap::default();
        let mut keys: Vec<(TetrisBoard<C>, u8)> = vec![init];
        intern.insert(init, 0);

        let mut group_start: Vec<u32> = vec![0];
        let mut groups: Vec<Group> = Vec::new();
        let mut owner: Vec<u32> = Vec::new();
        let mut edges: Vec<u32> = Vec::new();
        let mut succs: Vec<u32> = Vec::with_capacity(8);

        let mut next = 0usize;
        let mut exploded = false;
        while next < keys.len() {
            if keys.len() > budget {
                exploded = true;
                break;
            }
            let (board, bag) = keys[next];
            for piece in adversary_choices(&board, bag, &placements, adversary) {
                let mut succ_bag = bag & !u8::from(piece);
                if succ_bag == 0 {
                    succ_bag = full_bag;
                }
                succs.clear();
                for &pl in &placements[piece.index() as usize] {
                    let mut b = board;
                    let res = b.apply_piece_placement(pl);
                    if bool::from(res.is_lost) {
                        continue;
                    }
                    if max_holes.is_some_and(|k| b.total_holes() > k) {
                        continue;
                    }
                    let key = (b, succ_bag);
                    let id = match intern.entry(key) {
                        std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            let id = keys.len() as u32;
                            keys.push(key);
                            e.insert(id);
                            id
                        }
                    };
                    succs.push(id);
                }
                succs.sort_unstable();
                succs.dedup();
                groups.push(Group {
                    piece: piece.index(),
                    edge_start: edges.len() as u32,
                    edge_len: succs.len() as u8,
                });
                owner.push(next as u32);
                edges.extend_from_slice(&succs);
            }
            group_start.push(groups.len() as u32);
            next += 1;
        }

        // Reverse CSR: succ node -> the groups pointing at it.
        let n = keys.len();
        let mut rev_count = vec![0u32; n + 1];
        for &succ in &edges {
            rev_count[succ as usize + 1] += 1;
        }
        for i in 0..n {
            rev_count[i + 1] += rev_count[i];
        }
        let rev_start = rev_count.clone();
        let mut cursor = rev_count;
        let mut rev_data = vec![0u32; edges.len()];
        for (gid, g) in groups.iter().enumerate() {
            let lo = g.edge_start as usize;
            let hi = lo + g.edge_len as usize;
            for &succ in &edges[lo..hi] {
                rev_data[cursor[succ as usize] as usize] = gid as u32;
                cursor[succ as usize] += 1;
            }
        }

        Self {
            keys,
            expanded: next,
            exploded,
            group_start,
            groups,
            owner,
            edges,
            rev_start,
            rev_data,
            placements,
            full_bag,
            adversary,
        }
    }

    fn node_groups(&self, node: usize) -> std::ops::Range<usize> {
        self.group_start[node] as usize..self.group_start[node + 1] as usize
    }
}

// ---------------------------------------------------------------------------
// Greatest-fixed-point solve (retrograde death propagation)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Adversary picks the piece: a node dies when SOME group has no living child.
    Adversarial,
    /// Player picks the piece: a node dies when EVERY group has no living child.
    Cooperative,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Adversarial => "adv",
            Mode::Cooperative => "coop",
        }
    }
}

struct Solved {
    alive: Vec<bool>,
    /// Death-propagation depth per dead node (0 = an obligation with no legal
    /// surviving placement; a diagnostic lower-bound flavor of forced-loss plies).
    depth: Vec<u32>,
    alive_count: usize,
}

/// GFP with a chosen polarity for budget-frontier (unexpanded) nodes.
///
/// `frontier_alive = false` treats them as dead: the surviving set is an
/// under-approximation, so an ALIVE init is sound. `frontier_alive = true` treats
/// them as immortal: the surviving set over-approximates, so a DEAD init is sound
/// (the opponent wins strictly inside the explored region). On a non-exploded graph
/// the polarity is irrelevant.
fn solve<C: TetrisGameConfig>(g: &Graph<C>, mode: Mode, frontier_alive: bool) -> Solved
where
    [(); C::COLS]:,
{
    let n = g.keys.len();
    let mut alive = vec![true; n];
    let mut depth = vec![0u32; n];
    let mut group_alive: Vec<u8> = g.groups.iter().map(|gr| gr.edge_len).collect();
    // Cooperative mode: how many of the node's groups still have a living child.
    let mut live_groups: Vec<u8> = vec![0; if mode == Mode::Cooperative { n } else { 0 }];
    let mut queue: VecDeque<u32> = VecDeque::new();

    let mut kill = |node: u32,
                    d: u32,
                    alive: &mut Vec<bool>,
                    depth: &mut Vec<u32>,
                    queue: &mut VecDeque<u32>| {
        if alive[node as usize] {
            alive[node as usize] = false;
            depth[node as usize] = d;
            queue.push_back(node);
        }
    };

    // Seeds: budget-frontier nodes die or live per the chosen polarity; expanded
    // nodes die immediately per the mode's quantifier over empty groups.
    for node in 0..n {
        if node >= g.expanded {
            if !frontier_alive {
                kill(node as u32, 0, &mut alive, &mut depth, &mut queue);
            }
            continue;
        }
        let gr = g.node_groups(node);
        match mode {
            Mode::Adversarial => {
                if gr.clone().any(|gi| g.groups[gi].edge_len == 0) {
                    kill(node as u32, 0, &mut alive, &mut depth, &mut queue);
                }
            }
            Mode::Cooperative => {
                let live = gr.clone().filter(|&gi| g.groups[gi].edge_len > 0).count();
                live_groups[node] = live as u8;
                if live == 0 {
                    kill(node as u32, 0, &mut alive, &mut depth, &mut queue);
                }
            }
        }
    }

    while let Some(dead) = queue.pop_front() {
        let d = depth[dead as usize];
        let lo = g.rev_start[dead as usize] as usize;
        let hi = g.rev_start[dead as usize + 1] as usize;
        for &gid in &g.rev_data[lo..hi] {
            let ga = &mut group_alive[gid as usize];
            *ga -= 1;
            if *ga == 0 {
                let parent = g.owner[gid as usize];
                match mode {
                    Mode::Adversarial => {
                        kill(parent, d + 1, &mut alive, &mut depth, &mut queue);
                    }
                    Mode::Cooperative => {
                        let lg = &mut live_groups[parent as usize];
                        *lg -= 1;
                        if *lg == 0 {
                            kill(parent, d + 1, &mut alive, &mut depth, &mut queue);
                        }
                    }
                }
            }
        }
    }

    let alive_count = alive.iter().filter(|&&a| a).count();
    Solved {
        alive,
        depth,
        alive_count,
    }
}

// ---------------------------------------------------------------------------
// Independent carrier re-verification (the proof artifact when ALIVE)
// ---------------------------------------------------------------------------

/// Recheck the surviving set against the engine from scratch: for every alive state
/// and every obligated piece, some placement must land inside the alive set. Uses no
/// graph bookkeeping — only the intern map to identify successors.
fn verify_carrier<C: TetrisGameConfig>(g: &Graph<C>, s: &Solved, mode: Mode) -> Result<()>
where
    [(); C::COLS]:,
{
    let mut index: FxHashMap<(TetrisBoard<C>, u8), u32> = FxHashMap::default();
    for (id, &key) in g.keys.iter().enumerate() {
        index.insert(key, id as u32);
    }
    for (node, &(board, bag)) in g.keys.iter().enumerate() {
        if !s.alive[node] {
            continue;
        }
        let mut node_ok = mode == Mode::Adversarial;
        for piece in adversary_choices(&board, bag, &g.placements, g.adversary) {
            let mut succ_bag = bag & !u8::from(piece);
            if succ_bag == 0 {
                succ_bag = g.full_bag;
            }
            let piece_ok = g.placements[piece.index() as usize].iter().any(|&pl| {
                let mut b = board;
                let res = b.apply_piece_placement(pl);
                !bool::from(res.is_lost)
                    && index
                        .get(&(b, succ_bag))
                        .is_some_and(|&id| s.alive[id as usize])
            });
            match mode {
                Mode::Adversarial => {
                    if !piece_ok {
                        bail!(
                            "carrier violation: alive node {node} has no surviving answer to {piece}"
                        );
                    }
                }
                Mode::Cooperative => node_ok |= piece_ok,
            }
        }
        if !node_ok {
            bail!("carrier violation: alive node {node} has no surviving piece choice");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Iterative-deepening kill-prover (directed DFS for a sound DEAD)
// ---------------------------------------------------------------------------

/// Proves `kill within <= d plies` facts by depth-first AND-OR search with two
/// path-independent memo tables:
///
///   * `dead[s] = k`  — the adversary forces a top-out from `s` within `k` plies
///     (absolute fact, reusable on any path);
///   * `lb[s] = d`    — no kill from `s` exists within `d` plies (also absolute,
///     because the statement is depth-indexed — this sidesteps the
///     graph-history-interaction problem entirely; no cycle handling needed).
///
/// The adversary tries every bag piece (worst-first by a greedy eval), so a
/// DEAD result is EXACT for the full game — unlike the scripted adversaries.
/// "No kill within D" is only a survival lower bound of D plies. Explores the
/// forcing subtree instead of the breadth-first arena: memory scales with the
/// states the near-optimal adversary actually visits.
struct KillProver<C: TetrisGameConfig>
where
    [(); C::COLS]:,
{
    placements: [Vec<TetrisPiecePlacement>; 7],
    full_bag: u8,
    max_holes: Option<u32>,
    /// Minimal proven kill depth per state.
    dead: FxHashMap<(TetrisBoard<C>, u8), u16>,
    /// Maximal proven kill-free depth per state.
    lb: FxHashMap<(TetrisBoard<C>, u8), u16>,
    nodes: u64,
    budget: usize,
    aborted: bool,
}

impl<C: TetrisGameConfig> KillProver<C>
where
    [(); C::COLS]:,
{
    fn new(budget: usize, max_holes: Option<u32>) -> Self {
        Self {
            placements: placements_for::<C>(),
            full_bag: C::PIECE_SET.into(),
            max_holes,
            dead: FxHashMap::default(),
            lb: FxHashMap::default(),
            nodes: 0,
            budget,
            aborted: false,
        }
    }

    /// The player's surviving replies to `piece`, best-eval first, or `None` if the
    /// piece kills immediately (no surviving placement).
    fn replies(
        &self,
        board: &TetrisBoard<C>,
        bag: u8,
        piece: TetrisPiece,
    ) -> Option<Vec<Reply<C>>> {
        let mut succ_bag = bag & !u8::from(piece);
        if succ_bag == 0 {
            succ_bag = self.full_bag;
        }
        let mut out: Vec<Reply<C>> = Vec::new();
        for &pl in &self.placements[piece.index() as usize] {
            let mut b = *board;
            let res = b.apply_piece_placement(pl);
            if bool::from(res.is_lost) {
                continue;
            }
            if self.max_holes.is_some_and(|k| b.total_holes() > k) {
                continue;
            }
            out.push((eval_board(&b), (b, succ_bag)));
        }
        if out.is_empty() {
            return None;
        }
        // Sort by (eval, state) so identical successors are adjacent — the dedup is
        // then exact: no state is handed to the search twice from one reply set.
        out.sort_unstable();
        out.dedup_by_key(|&mut (_, k)| k);
        Some(out)
    }

    /// Is there an adversary strategy topping the player out within `d` plies?
    fn kill_within(&mut self, key: (TetrisBoard<C>, u8), d: u32) -> bool {
        if let Some(&kd) = self.dead.get(&key) {
            if u32::from(kd) <= d {
                return true;
            }
        }
        if d == 0 {
            return false;
        }
        if let Some(&l) = self.lb.get(&key) {
            if u32::from(l) >= d {
                return false;
            }
        }
        self.nodes += 1;
        if self.aborted || self.dead.len() + self.lb.len() > self.budget {
            self.aborted = true;
            return false; // treated as "unknown": blocks kill proofs, never fakes them
        }

        let (board, bag) = key;
        // Expand each bag piece once; the adversary tries the worst piece first
        // (immediate kills, then highest player-best eval).
        let mut options: Vec<(i64, Option<Vec<Reply<C>>>)> = Vec::new();
        for piece in pieces_in_mask(bag) {
            let replies = self.replies(&board, bag, piece);
            let badness = replies.as_ref().map_or(i64::MAX, |r| r[0].0);
            options.push((badness, replies));
        }
        options.sort_unstable_by_key(|&(b, _)| std::cmp::Reverse(b));

        for (_, replies) in options {
            let Some(replies) = replies else {
                // A piece with no surviving placement: kill in one ply.
                let e = self.dead.entry(key).or_insert(u16::MAX);
                *e = (*e).min(1);
                return true;
            };
            let mut all_dead = true;
            for &(_, succ) in &replies {
                if !self.kill_within(succ, d - 1) {
                    all_dead = false;
                    break;
                }
            }
            if all_dead {
                let bound = d.min(u32::from(u16::MAX)) as u16;
                let e = self.dead.entry(key).or_insert(u16::MAX);
                *e = (*e).min(bound);
                return true;
            }
            if self.aborted {
                return false;
            }
        }
        if !self.aborted {
            let bound = d.min(u32::from(u16::MAX)) as u16;
            let e = self.lb.entry(key).or_insert(0);
            *e = (*e).max(bound);
        }
        false
    }

    /// Walk one forced line from `key` for display: the killing piece each ply, the
    /// player's most resistant reply per the memo.
    fn print_kill_line(&mut self, mut key: (TetrisBoard<C>, u8), kd: u32) {
        println!("  kill line (adversary piece -> player's most resistant reply):");
        let mut d = kd;
        for step in 0..kd {
            let (board, bag) = key;
            let mut killer: Option<(TetrisPiece, Option<StateKey<C>>)> = None;
            for piece in pieces_in_mask(bag) {
                match self.replies(&board, bag, piece) {
                    None => {
                        killer = Some((piece, None));
                        break;
                    }
                    Some(replies) => {
                        if replies.iter().all(|&(_, s)| self.kill_within(s, d - 1)) {
                            // Most resistant reply: the successor with max proven kill depth.
                            let best = replies
                                .iter()
                                .map(|&(_, s)| s)
                                .max_by_key(|s| self.dead.get(s).copied().unwrap_or(u16::MAX));
                            killer = Some((piece, best));
                            break;
                        }
                    }
                }
            }
            let Some((piece, reply)) = killer else {
                println!("    (line stopped early at step {step})");
                return;
            };
            match reply {
                None => {
                    println!(
                        "    #{step:<3} adversary draws {piece}: NO surviving placement — top-out. {}",
                        render_board(&board)
                    );
                    return;
                }
                Some(succ) => {
                    let (nb, _) = succ;
                    println!(
                        "    #{step:<3} {piece} h={:<2} holes={:<2} {}",
                        nb.height(),
                        nb.total_holes(),
                        render_board(&nb)
                    );
                    key = succ;
                    d -= 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Guided best-first AND-OR solver (anytime; permanent cache; exact backprop)
// ---------------------------------------------------------------------------

/// Anytime AND-OR solver over the full game DAG, no depth budgets.
///
/// AND-nodes are interned by `(board, bag)`; each expanded AND-node owns one
/// OR-node per drawable piece, so the `(board, bag, piece)` triple is
/// materialized exactly once ever — transpositions merge, nothing is searched
/// twice. Expansion order is a badness heap (holes >> height >> roughness),
/// with a proof-focus boost: when an OR-node is down to <= 2 surviving
/// children, those children jump the queue (the closer a refutation is to
/// complete, the sooner it is finished).
///
/// Death is exact and backpropagates immediately through reverse edges:
/// an OR-node dies when its last surviving placement-child dies (or it never
/// had one); its owning AND-node dies with it (the adversary presents that
/// piece); each dead AND-node decrements the surviving-child counters of every
/// parent OR-node in the DAG. Root death = sound DEAD for the full game.
///
/// If the heap drains, the reachable graph is fully expanded and the result is
/// the exact GFP (root alive <=> ALIVE). Otherwise a final pessimistic sweep
/// (frontier treated as dead) can still certify a sound ALIVE.
struct GuidedSolver<C: TetrisGameConfig>
where
    [(); C::COLS]:,
{
    placements: [Vec<TetrisPiecePlacement>; 7],
    full_bag: u8,
    max_holes: Option<u32>,
    intern: FxHashMap<StateKey<C>, u32>,
    keys: Vec<StateKey<C>>,
    /// 0 = unexpanded, 1 = expanded, 2 = dead.
    status: Vec<u8>,
    /// Expanded AND-node -> contiguous range of its OR-nodes.
    or_start: Vec<u32>,
    or_count: Vec<u8>,
    /// Reverse edges: AND-node -> the OR-nodes with an edge into it.
    parents: Vec<Vec<u32>>,
    or_owner: Vec<u32>,
    or_alive: Vec<u32>,
    or_edge_start: Vec<u32>,
    or_edge_len: Vec<u16>,
    edges: Vec<u32>,
    heap: std::collections::BinaryHeap<(i64, u32)>,
    expansions: usize,
    deaths: usize,
}

/// Scarcity weight: children of an OR-node with few surviving replies lead the
/// queue (a proof-number-style priority); board badness breaks ties within a
/// scarcity class.
const FOCUS_BONUS: i64 = 1 << 50;

impl<C: TetrisGameConfig> GuidedSolver<C>
where
    [(); C::COLS]:,
{
    fn new(max_holes: Option<u32>) -> Self {
        let mut s = Self {
            placements: placements_for::<C>(),
            full_bag: C::PIECE_SET.into(),
            max_holes,
            intern: FxHashMap::default(),
            keys: Vec::new(),
            status: Vec::new(),
            or_start: Vec::new(),
            or_count: Vec::new(),
            parents: Vec::new(),
            or_owner: Vec::new(),
            or_alive: Vec::new(),
            or_edge_start: Vec::new(),
            or_edge_len: Vec::new(),
            edges: Vec::new(),
            heap: std::collections::BinaryHeap::new(),
            expansions: 0,
            deaths: 0,
        };
        let root = (TetrisBoard::<C>::new(), s.full_bag);
        let id = s.intern_node(root);
        s.heap.push((0, id));
        s
    }

    fn intern_node(&mut self, key: StateKey<C>) -> u32 {
        match self.intern.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let id = self.keys.len() as u32;
                self.keys.push(key);
                self.status.push(0);
                self.or_start.push(0);
                self.or_count.push(0);
                self.parents.push(Vec::new());
                e.insert(id);
                id
            }
        }
    }

    /// Expand one AND-node: create its OR-nodes and their placement children,
    /// backpropagating any death this completes. Returns true if the root died.
    fn expand(&mut self, id: u32) -> bool {
        if self.status[id as usize] != 0 {
            return false; // stale heap entry
        }
        self.status[id as usize] = 1;
        self.expansions += 1;
        let (board, bag) = self.keys[id as usize];
        self.or_start[id as usize] = self.or_owner.len() as u32;
        let mut newly_dead = false;
        for piece in pieces_in_mask(bag) {
            let mut succ_bag = bag & !u8::from(piece);
            if succ_bag == 0 {
                succ_bag = self.full_bag;
            }
            let mut children: Vec<(i64, StateKey<C>)> = Vec::new();
            for &pl in &self.placements[piece.index() as usize] {
                let mut b = board;
                let res = b.apply_piece_placement(pl);
                if bool::from(res.is_lost) {
                    continue;
                }
                if self.max_holes.is_some_and(|k| b.total_holes() > k) {
                    continue;
                }
                children.push((eval_board(&b), (b, succ_bag)));
            }
            children.sort_unstable();
            children.dedup_by_key(|&mut (_, k)| k);

            let or_id = self.or_owner.len() as u32;
            self.or_owner.push(id);
            self.or_edge_start.push(self.edges.len() as u32);
            self.or_count[id as usize] += 1;
            let mut alive = 0u32;
            let mut fresh: Vec<(i64, u32)> = Vec::with_capacity(children.len());
            for &(badness, child_key) in &children {
                let child = self.intern_node(child_key);
                self.edges.push(child);
                self.parents[child as usize].push(or_id);
                if self.status[child as usize] != 2 {
                    alive += 1;
                    if self.status[child as usize] == 0 {
                        fresh.push((badness, child));
                    }
                }
            }
            self.or_edge_len.push(children.len() as u16);
            self.or_alive.push(alive);
            if alive == 0 {
                // No surviving reply to this piece: the AND-node is dead now.
                newly_dead = true;
            } else {
                // Proof-number-style priority: refutations with few surviving
                // replies are closest to completion, so their children lead the
                // queue; badness breaks ties within a scarcity class.
                for (badness, child) in fresh {
                    self.heap
                        .push((FOCUS_BONUS / i64::from(alive) + badness, child));
                }
            }
        }
        if newly_dead {
            return self.backprop_death(id);
        }
        false
    }

    /// Mark `id` dead and cascade through reverse edges. Returns true if the
    /// root (node 0) died.
    fn backprop_death(&mut self, id: u32) -> bool {
        let mut queue: Vec<u32> = Vec::new();
        let mut root_dead = false;
        let mut kill = |s: &mut Self, n: u32, queue: &mut Vec<u32>, root_dead: &mut bool| {
            if s.status[n as usize] != 2 {
                s.status[n as usize] = 2;
                s.deaths += 1;
                queue.push(n);
                if n == 0 {
                    *root_dead = true;
                }
            }
        };
        kill(self, id, &mut queue, &mut root_dead);
        while let Some(dead) = queue.pop() {
            let parent_list = std::mem::take(&mut self.parents[dead as usize]);
            for &or_id in &parent_list {
                let a = &mut self.or_alive[or_id as usize];
                if *a == 0 {
                    continue; // owning OR-node already refuted
                }
                *a -= 1;
                if *a == 0 {
                    let owner = self.or_owner[or_id as usize];
                    kill(self, owner, &mut queue, &mut root_dead);
                } else {
                    // Re-rank: this refutation just got one child closer to complete.
                    let a = *a;
                    let lo = self.or_edge_start[or_id as usize] as usize;
                    let hi = lo + self.or_edge_len[or_id as usize] as usize;
                    for i in lo..hi {
                        let child = self.edges[i];
                        if self.status[child as usize] == 0 {
                            let (b, _) = self.keys[child as usize];
                            self.heap
                                .push((FOCUS_BONUS / i64::from(a) + eval_board(&b), child));
                        }
                    }
                }
            }
            self.parents[dead as usize] = parent_list;
        }
        root_dead
    }

    /// Exact GFP over the expanded region with the frontier treated as dead.
    /// If the root survives, that is a sound ALIVE for the full game (and if the
    /// heap is empty it is the exact game value). Returns (root_alive, carrier).
    fn pessimistic_sweep(&self) -> (bool, usize) {
        let n = self.keys.len();
        let mut alive: Vec<bool> = (0..n).map(|i| self.status[i] == 1).collect();
        let mut cnt: Vec<u32> = vec![0; self.or_owner.len()];
        for (or_id, c) in cnt.iter_mut().enumerate() {
            let lo = self.or_edge_start[or_id] as usize;
            let hi = lo + self.or_edge_len[or_id] as usize;
            *c = self.edges[lo..hi]
                .iter()
                .filter(|&&ch| alive[ch as usize])
                .count() as u32;
        }
        let mut queue: Vec<u32> = Vec::new();
        for (or_id, &c) in cnt.iter().enumerate() {
            if c == 0 {
                let owner = self.or_owner[or_id];
                if alive[owner as usize] {
                    alive[owner as usize] = false;
                    queue.push(owner);
                }
            }
        }
        while let Some(dead) = queue.pop() {
            for &or_id in &self.parents[dead as usize] {
                let c = &mut cnt[or_id as usize];
                if *c == 0 {
                    continue;
                }
                *c -= 1;
                if *c == 0 {
                    let owner = self.or_owner[or_id as usize];
                    if alive[owner as usize] {
                        alive[owner as usize] = false;
                        queue.push(owner);
                    }
                }
            }
        }
        (alive[0], alive.iter().filter(|&&a| a).count())
    }
}

/// Run the guided solver: expand best-first until the root dies, the graph is
/// exhausted, or the expansion budget is hit; then sweep for a sound ALIVE.
fn run_guided<C: TetrisGameConfig>(cli: &Cli) -> Result<RowOutcome>
where
    [(); C::COLS]:,
{
    let t0 = Instant::now();
    let mut s = GuidedSolver::<C>::new(cli.max_holes);
    let mut verdict = None;
    let mut next_report = 2_000_000;
    while let Some((_, id)) = s.heap.pop() {
        if s.expand(id) {
            println!(
                "ROWS={:<2} guided: root DEAD after {} expansions ({} interned, {} dead, {:.1}s)",
                C::ROWS,
                s.expansions,
                s.keys.len(),
                s.deaths,
                t0.elapsed().as_secs_f64()
            );
            verdict = Some(Verdict::Dead { depth: 0 });
            break;
        }
        if s.expansions >= cli.budget {
            println!(
                "ROWS={:<2} guided: expansion budget {} hit ({} interned, {} dead, {:.1}s)",
                C::ROWS,
                cli.budget,
                s.keys.len(),
                s.deaths,
                t0.elapsed().as_secs_f64()
            );
            break;
        }
        if s.expansions >= next_report {
            next_report += 2_000_000;
            println!(
                "ROWS={:<2} guided: {}M expansions, {} interned, {} dead, heap {}, {:.1}s",
                C::ROWS,
                s.expansions / 1_000_000,
                s.keys.len(),
                s.deaths,
                s.heap.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let exhausted = s.heap.is_empty() && verdict.is_none();
    let verdict = verdict.unwrap_or_else(|| {
        let (root_alive, carrier) = s.pessimistic_sweep();
        if root_alive {
            println!(
                "ROWS={:<2} guided: root ALIVE — {} carrier ({}; {:.1}s)",
                C::ROWS,
                carrier,
                if exhausted {
                    "graph exhausted: exact GFP"
                } else {
                    "sound via pessimistic frontier"
                },
                t0.elapsed().as_secs_f64()
            );
            Verdict::Alive { carrier }
        } else if exhausted {
            println!(
                "ROWS={:<2} guided: graph exhausted and root dead in exact GFP",
                C::ROWS
            );
            Verdict::Dead { depth: 0 }
        } else {
            println!(
                "ROWS={:<2} guided: INCONCLUSIVE at budget ({} proven-dead states banked)",
                C::ROWS,
                s.deaths
            );
            Verdict::Exploded
        }
    });
    Ok(RowOutcome {
        rows: C::ROWS as u32,
        reachable: s.keys.len(),
        edges: s.expansions,
        adv: verdict,
        coop: None,
        secs: t0.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// Continuous parallel AND-OR engine (port of atlas/tetris_atlas_inmemory.rs)
// ---------------------------------------------------------------------------

/// The continuous, fully-parallel AND-OR engine from `tetris_atlas_inmemory`,
/// made generic over the variant config.
///
/// Expansion and backward death-propagation run as one interleaved task stream
/// across all workers: DEATH is the monotone least fixpoint (never retracted, so
/// it may run concurrently with expansion), while the safe verdict is only read
/// at quiescence. Workers drain death tasks first, else pop the lowest-mass
/// state from a `SkipMap` frontier. Interning publishes the `Node` BEFORE the id
/// (under the `DashMap` entry lock) so a freshly-interned id can never be
/// observed without its node — the lost-decrement race documented in the atlas
/// implementation.
///
/// Verdict semantics: root death during the run is a sound DEAD (monotone).
/// Otherwise unexpanded frontier nodes are conservatively killed and the death
/// queue drained to fixpoint; a root that survives that is a sound ALIVE (exact
/// if the frontier drained on its own).
mod parallel {
    use super::*;
    use crossbeam::queue::SegQueue;
    use crossbeam::utils::Backoff;
    use crossbeam_skiplist::SkipMap;
    use dashmap::DashMap;
    use dashmap::mapref::entry::Entry;
    use rustc_hash::FxBuildHasher;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

    type FxDashMap<K, V> = DashMap<K, V, FxBuildHasher>;
    const SHARDS: usize = 256;

    pub struct PNode {
        /// Remaining-bag mask (the board lives in the intern key and frontier).
        bag: u8,
        /// Reverse edges (parent id, piece index); appended concurrently, drained on death.
        preds: Vec<(u32, u8)>,
        /// Surviving-placement count per forced piece; published once, then only decremented.
        live: [i16; 7],
        expanded: bool,
        dead: bool,
    }

    pub struct Engine<C: TetrisGameConfig>
    where
        [(); C::COLS]:,
    {
        placements: [Vec<TetrisPiecePlacement>; 7],
        full_bag: u8,
        max_holes: Option<u32>,
        state_to_id: FxDashMap<StateKey<C>, u32>,
        nodes: FxDashMap<u32, PNode>,
        death_q: SegQueue<u32>,
        frontier: SkipMap<(u32, u32), StateKey<C>>,
        next_id: AtomicU32,
        pub dead_count: AtomicU64,
        pub expanded_count: AtomicU64,
        in_flight: AtomicU64,
        root_dead: AtomicBool,
        budget_stop: AtomicBool,
        budget: usize,
    }

    impl<C: TetrisGameConfig + 'static> Engine<C>
    where
        [(); C::COLS]:,
    {
        pub fn new(budget: usize, max_holes: Option<u32>) -> Arc<Self> {
            let e = Arc::new(Self {
                placements: placements_for::<C>(),
                full_bag: C::PIECE_SET.into(),
                max_holes,
                state_to_id: DashMap::with_hasher_and_shard_amount(FxBuildHasher, SHARDS),
                nodes: DashMap::with_hasher_and_shard_amount(FxBuildHasher, SHARDS),
                death_q: SegQueue::new(),
                frontier: SkipMap::new(),
                next_id: AtomicU32::new(0),
                dead_count: AtomicU64::new(0),
                expanded_count: AtomicU64::new(0),
                in_flight: AtomicU64::new(0),
                root_dead: AtomicBool::new(false),
                budget_stop: AtomicBool::new(false),
                budget,
            });
            let root = (TetrisBoard::<C>::new(), e.full_bag);
            let (id, _) = e.intern(root);
            e.in_flight.fetch_add(1, Ordering::SeqCst);
            e.frontier.insert((0, id), root);
            e
        }

        /// Intern a state. Publishes the node BEFORE the id (entry lock held) so an
        /// id is only ever observable with its node present.
        fn intern(&self, key: StateKey<C>) -> (u32, bool) {
            match self.state_to_id.entry(key) {
                Entry::Occupied(o) => (*o.get(), false),
                Entry::Vacant(v) => {
                    let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                    self.nodes.insert(
                        id,
                        PNode {
                            bag: key.1,
                            preds: Vec::new(),
                            live: [0; 7],
                            expanded: false,
                            dead: false,
                        },
                    );
                    v.insert(id);
                    (id, true)
                }
            }
        }

        pub fn state_count(&self) -> usize {
            self.next_id.load(Ordering::Relaxed) as usize
        }

        /// THE death condition: expanded, not dead, and some forced piece has no
        /// surviving placement left.
        fn dies_now(node: &PNode) -> bool {
            !node.dead
                && node.expanded
                && pieces_in_mask(node.bag).any(|p| node.live[p.index() as usize] == 0)
        }

        fn enqueue_death(&self, id: u32) {
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            self.death_q.push(id);
        }

        fn decrement_live(&self, id: u32, piece_idx: usize) {
            let trigger = {
                let Some(mut node) = self.nodes.get_mut(&id) else {
                    return;
                };
                if node.dead {
                    false
                } else {
                    node.live[piece_idx] -= 1;
                    debug_assert!(node.live[piece_idx] >= 0, "double-decrement on {id}");
                    Self::dies_now(node.value())
                }
            };
            if trigger {
                self.enqueue_death(id);
            }
        }

        fn process_expand(&self, key: StateKey<C>) {
            let (id, _) = self.intern(key);
            if self.nodes.get(&id).map(|n| n.dead).unwrap_or(false) {
                return; // death-first pruning
            }
            let (board, bag) = key;
            let mut live: [i16; 7] = [0; 7];
            let mut pending: Vec<(u32, u8)> = Vec::new();
            let mut new_succs: Vec<(u32, StateKey<C>)> = Vec::new();
            let mut seen: Vec<u32> = Vec::with_capacity(16);
            for piece in pieces_in_mask(bag) {
                let piece_idx = piece.index() as usize;
                let mut succ_bag = bag & !u8::from(piece);
                if succ_bag == 0 {
                    succ_bag = self.full_bag;
                }
                seen.clear();
                for &pl in &self.placements[piece_idx] {
                    let mut b = board;
                    let res = b.apply_piece_placement(pl);
                    if bool::from(res.is_lost) {
                        continue;
                    }
                    if self.max_holes.is_some_and(|k| b.total_holes() > k) {
                        continue;
                    }
                    let succ_key = (b, succ_bag);
                    let (succ_id, is_new) = self.intern(succ_key);
                    if seen.contains(&succ_id) {
                        continue; // one live count + one reverse edge per distinct successor
                    }
                    seen.push(succ_id);
                    live[piece_idx] += 1;
                    pending.push((succ_id, piece_idx as u8));
                    if is_new {
                        new_succs.push((succ_id, succ_key));
                    }
                }
            }
            // Publish live counts + expanded under this node's lock.
            if let Some(mut node) = self.nodes.get_mut(&id) {
                node.live = live;
                node.expanded = true;
            }
            self.expanded_count.fetch_add(1, Ordering::Relaxed);
            for (sid, skey) in new_succs {
                self.in_flight.fetch_add(1, Ordering::SeqCst);
                self.frontier.insert((skey.0.count(), sid), skey);
            }
            // Reverse edges, with a compensating decrement if the successor already died.
            for (succ_id, piece_idx) in pending {
                let succ_dead = {
                    let Some(mut snode) = self.nodes.get_mut(&succ_id) else {
                        continue;
                    };
                    if snode.dead {
                        true
                    } else {
                        snode.preds.push((id, piece_idx));
                        false
                    }
                };
                if succ_dead {
                    self.decrement_live(id, piece_idx as usize);
                }
            }
            // Base-case death check (a state born stuck originates the death wave).
            let should_die = self
                .nodes
                .get(&id)
                .map(|n| Self::dies_now(n.value()))
                .unwrap_or(false);
            if should_die {
                self.enqueue_death(id);
            }
        }

        fn process_die(&self, id: u32) {
            let preds = {
                let Some(mut node) = self.nodes.get_mut(&id) else {
                    return;
                };
                if node.dead {
                    return;
                }
                node.dead = true;
                std::mem::take(&mut node.preds)
            };
            self.dead_count.fetch_add(1, Ordering::Relaxed);
            if id == 0 {
                self.root_dead.store(true, Ordering::SeqCst);
            }
            for (parent, piece_idx) in preds {
                self.decrement_live(parent, piece_idx as usize);
            }
        }

        fn worker_loop(&self) {
            let backoff = Backoff::new();
            loop {
                if self.root_dead.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(d) = self.death_q.pop() {
                    self.process_die(d);
                    self.in_flight.fetch_sub(1, Ordering::SeqCst);
                    backoff.reset();
                } else if self.budget_stop.load(Ordering::Relaxed) {
                    break; // budget: stop expanding, deaths already drained above
                } else if let Some(entry) = self.frontier.pop_front() {
                    if self.state_count() > self.budget {
                        self.budget_stop.store(true, Ordering::Relaxed);
                    }
                    self.process_expand(*entry.value());
                    self.in_flight.fetch_sub(1, Ordering::SeqCst);
                    backoff.reset();
                } else if self.in_flight.load(Ordering::SeqCst) == 0 {
                    break; // quiescence
                } else {
                    backoff.snooze();
                }
            }
        }

        /// Run to quiescence / root-death / budget. Returns (root_dead, exact):
        /// `exact` is true iff the frontier drained without a budget stop.
        pub fn run(self: &Arc<Self>, workers: usize) -> (bool, bool) {
            rayon::scope(|s| {
                for _ in 0..workers {
                    let e = Arc::clone(self);
                    s.spawn(move |_| e.worker_loop());
                }
            });
            // Drain any leftover genuine death tasks first: a root death here is
            // still the monotone (sound) kind, even after a budget stop.
            while let Some(d) = self.death_q.pop() {
                self.process_die(d);
            }
            if self.root_dead.load(Ordering::SeqCst) {
                return (true, true); // monotone: sound no matter when it fired
            }
            let exact = !self.budget_stop.load(Ordering::Relaxed);
            // Conservative finalize: kill every unexpanded survivor, drain to fixpoint.
            // Deaths caused by THIS phase are pessimistic (frontier-as-dead).
            let unexpanded: Vec<u32> = self
                .nodes
                .iter()
                .filter(|e| !e.value().dead && !e.value().expanded)
                .map(|e| *e.key())
                .collect();
            for id in unexpanded {
                self.process_die(id);
            }
            while let Some(d) = self.death_q.pop() {
                self.process_die(d);
            }
            (self.root_dead.load(Ordering::SeqCst), exact)
        }

        /// Independent closure check of the surviving set, replayed through the engine.
        pub fn verify_alive_carrier(&self) -> Result<usize> {
            let alive = |key: &StateKey<C>| -> bool {
                self.state_to_id
                    .get(key)
                    .and_then(|id| self.nodes.get(&id).map(|n| !n.dead && n.expanded))
                    .unwrap_or(false)
            };
            let mut carrier = 0usize;
            for e in self.state_to_id.iter() {
                let (board, bag) = *e.key();
                if !alive(e.key()) {
                    continue;
                }
                carrier += 1;
                for piece in pieces_in_mask(bag) {
                    let mut succ_bag = bag & !u8::from(piece);
                    if succ_bag == 0 {
                        succ_bag = self.full_bag;
                    }
                    let ok = self.placements[piece.index() as usize].iter().any(|&pl| {
                        let mut b = board;
                        let res = b.apply_piece_placement(pl);
                        !bool::from(res.is_lost)
                            && self.max_holes.is_none_or(|k| b.total_holes() <= k)
                            && alive(&(b, succ_bag))
                    });
                    if !ok {
                        bail!("carrier violation: alive state has no surviving answer to {piece}");
                    }
                }
            }
            Ok(carrier)
        }

        /// Is this state interned, expanded, and not proven dead?
        pub fn is_alive(&self, key: &StateKey<C>) -> bool {
            self.state_to_id
                .get(key)
                .and_then(|id| self.nodes.get(&id).map(|n| !n.dead && n.expanded))
                .unwrap_or(false)
        }

        /// Test hook: flip a state's dead flag WITHOUT propagation, to prove the
        /// independent carrier verifier catches inconsistent carriers.
        #[cfg(test)]
        pub fn force_dead_for_test(&self, key: &StateKey<C>) {
            if let Some(id) = self.state_to_id.get(key) {
                if let Some(mut n) = self.nodes.get_mut(&id) {
                    n.dead = true;
                }
            }
        }

        /// Test hook: snapshot every alive state key.
        #[cfg(test)]
        pub fn alive_keys_for_test(&self) -> Vec<StateKey<C>> {
            self.state_to_id
                .iter()
                .filter(|e| {
                    self.nodes
                        .get(e.value())
                        .map(|n| !n.dead && n.expanded)
                        .unwrap_or(false)
                })
                .map(|e| *e.key())
                .collect()
        }
    }
}

/// Run the continuous parallel engine for one ceiling.
fn run_parallel<C: TetrisGameConfig + 'static>(cli: &Cli) -> Result<RowOutcome>
where
    [(); C::COLS]:,
{
    let t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(1))
        .unwrap_or(4);
    let engine = parallel::Engine::<C>::new(cli.budget, cli.max_holes);
    let (root_dead, exact) = engine.run(workers);
    let states = engine.state_count();
    let dead = engine.dead_count.load(std::sync::atomic::Ordering::Relaxed);
    let expanded = engine
        .expanded_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let verdict = if root_dead {
        println!(
            "ROWS={:<2} parallel: root DEAD ({}; {} states, {} expanded, {} dead, {} workers, {:.1}s)",
            C::ROWS,
            if exact { "sound, monotone" } else { "sound" },
            states,
            expanded,
            dead,
            workers,
            t0.elapsed().as_secs_f64()
        );
        Verdict::Dead { depth: 0 }
    } else {
        let carrier = engine.verify_alive_carrier()?;
        println!(
            "ROWS={:<2} parallel: root ALIVE — closed carrier of {carrier} states, re-verified \
             ({}; {} states, {} dead, {} workers, {:.1}s)",
            C::ROWS,
            if exact {
                "exact quiescence"
            } else {
                "sound via conservative finalize at budget"
            },
            states,
            dead,
            workers,
            t0.elapsed().as_secs_f64()
        );
        Verdict::Alive { carrier }
    };
    // A root death after a budget-stop finalize is pessimistic, not exact.
    let verdict = match verdict {
        Verdict::Dead { .. } if !exact => {
            println!(
                "ROWS={:<2} parallel: (death arrived only after the conservative finalize — \
                 INCONCLUSIVE at budget)",
                C::ROWS
            );
            Verdict::Exploded
        }
        v => v,
    };
    Ok(RowOutcome {
        rows: C::ROWS as u32,
        reachable: states,
        edges: expanded as usize,
        adv: verdict,
        coop: None,
        secs: t0.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// Forced-loss trace (when the adversarial verdict is DEAD)
// ---------------------------------------------------------------------------

fn render_board<C: TetrisGameConfig>(b: &TetrisBoard<C>) -> String
where
    [(); C::COLS]:,
{
    let h = b.height().max(1) as usize;
    let mut out = String::new();
    for row in (0..h).rev() {
        out.push('|');
        for col in 0..C::COLS {
            out.push(if b.get_bit(col, row) { '#' } else { '.' });
        }
        out.push('|');
        if row > 0 {
            out.push(' ');
        }
    }
    out
}

/// Walk one forced-loss line from the initial state: at each dead node pick a piece
/// whose whole group is dead, then the player's most resistant reply (max death
/// depth). Ends at a node where some piece has no surviving placement at all.
fn print_trace<C: TetrisGameConfig>(g: &Graph<C>, s: &Solved, max_steps: usize)
where
    [(); C::COLS]:,
{
    println!("  forced-loss trace (adversary piece -> player's most resistant reply):");
    let mut node = 0u32;
    for step in 0..max_steps {
        let (board, bag) = g.keys[node as usize];
        let fatal = g.node_groups(node as usize).find(|&gi| {
            let gr = g.groups[gi];
            let lo = gr.edge_start as usize;
            let hi = lo + gr.edge_len as usize;
            g.edges[lo..hi].iter().all(|&sid| !s.alive[sid as usize])
        });
        let Some(gi) = fatal else {
            println!("    (trace stopped: node {node} has no fatal group?)");
            return;
        };
        let gr = g.groups[gi];
        let piece = TetrisPiece::new(gr.piece);
        if gr.edge_len == 0 {
            println!(
                "    #{step:<3} adversary draws {piece}: NO surviving placement — top-out. {}",
                render_board(&board)
            );
            return;
        }
        let lo = gr.edge_start as usize;
        let hi = lo + gr.edge_len as usize;
        let best = g.edges[lo..hi]
            .iter()
            .copied()
            .max_by_key(|&sid| s.depth[sid as usize]);
        let Some(best) = best else {
            return;
        };
        // Recover a placement realizing board -> best (for display only).
        let mut succ_bag = bag & !u8::from(piece);
        if succ_bag == 0 {
            succ_bag = g.full_bag;
        }
        let target = g.keys[best as usize];
        let pl = g.placements[piece.index() as usize]
            .iter()
            .copied()
            .find(|&pl| {
                let mut b = board;
                let res = b.apply_piece_placement(pl);
                !bool::from(res.is_lost) && (b, succ_bag) == target
            });
        let pl_str = pl.map_or_else(|| "?".to_string(), |p| format!("{p}"));
        let (nb, _) = target;
        println!(
            "    #{step:<3} {piece} -> {pl_str:<24} h={:<2} holes={:<2} {}",
            nb.height(),
            nb.total_holes(),
            render_board(&nb)
        );
        node = best;
    }
    println!("    (trace truncated at {max_steps} steps)");
}

// ---------------------------------------------------------------------------
// Per-ceiling run + CLI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Alive {
        carrier: usize,
    },
    Dead {
        depth: u32,
    },
    /// Kill-prover only: no forced kill exists within this many plies (a survival
    /// lower bound, not an ALIVE certificate).
    NoKill {
        plies: u32,
    },
    Exploded,
}

struct RowOutcome {
    rows: u32,
    reachable: usize,
    edges: usize,
    adv: Verdict,
    coop: Option<Verdict>,
    secs: f64,
}

fn run_one<C: TetrisGameConfig>(cli: &Cli) -> Result<RowOutcome>
where
    [(); C::COLS]:,
{
    let t0 = Instant::now();
    let g = Graph::<C>::build(cli.budget, cli.max_holes, cli.adversary);
    let build_secs = t0.elapsed().as_secs_f64();
    let band = cli
        .max_holes
        .map_or(String::new(), |k| format!(" holes<={k}"));
    let script = if cli.adversary == Adversary::All {
        String::new()
    } else {
        format!(" adversary={:?}", cli.adversary)
    };
    println!(
        "ROWS={:<2}{band}{script} reachable={} edges={} groups={} exploded={} ({build_secs:.2}s build)",
        C::ROWS,
        g.keys.len(),
        g.edges.len(),
        g.groups.len(),
        g.exploded,
    );

    let mut verdicts = [Verdict::Exploded; 2];
    for (slot, mode) in [Mode::Adversarial, Mode::Cooperative]
        .into_iter()
        .enumerate()
    {
        // Pessimistic pass (frontier dead): an alive init is a sound ALIVE.
        let s = solve(&g, mode, false);
        let verdict = if s.alive[0] {
            verify_carrier(&g, &s, mode)?;
            let caveat = if cli.adversary == Adversary::All {
                ", re-verified against the engine"
            } else {
                " vs the SCRIPTED adversary only (necessary condition, not a certificate)"
            };
            println!(
                "  {:<4} ALIVE — closed carrier of {} states{caveat}",
                mode.name(),
                s.alive_count
            );
            Verdict::Alive {
                carrier: s.alive_count,
            }
        } else {
            // Optimistic pass (frontier immortal): a dead init is a sound DEAD even
            // on an exploded graph — the opponent wins inside the explored region.
            let (s, exact) = if g.exploded {
                (solve(&g, mode, true), false)
            } else {
                (s, true)
            };
            if !s.alive[0] {
                println!(
                    "  {:<4} DEAD — init killed at propagation depth {} ({} of {} states survive{})",
                    mode.name(),
                    s.depth[0],
                    s.alive_count,
                    g.keys.len(),
                    if exact { "" } else { "; frontier optimistic" }
                );
                if mode == Mode::Adversarial && cli.trace {
                    print_trace(&g, &s, cli.trace_steps);
                }
                Verdict::Dead { depth: s.depth[0] }
            } else {
                println!(
                    "  {:<4} INCONCLUSIVE at budget (alive optimistically, dead pessimistically)",
                    mode.name()
                );
                Verdict::Exploded
            }
        };
        verdicts[slot] = verdict;
    }

    Ok(RowOutcome {
        rows: C::ROWS as u32,
        reachable: g.keys.len(),
        edges: g.edges.len(),
        adv: verdicts[0],
        coop: Some(verdicts[1]),
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// Iterative-deepening kill-prover run: proves a sound adversarial DEAD by finding a
/// forced kill from the initial state, or reports a survival lower bound.
fn run_kill<C: TetrisGameConfig>(cli: &Cli) -> Result<RowOutcome>
where
    [(); C::COLS]:,
{
    let t0 = Instant::now();
    let mut prover = KillProver::<C>::new(cli.budget, cli.max_holes);
    let init = (TetrisBoard::<C>::new(), prover.full_bag);
    let mut verdict = Verdict::NoKill {
        plies: cli.kill_depth,
    };
    for d in 1..=cli.kill_depth {
        let found = prover.kill_within(init, d);
        if found {
            println!(
                "ROWS={:<2} kill-dfs: DEAD — forced kill within {d} plies \
                 (nodes={}, memo={}+{}, {:.1}s)",
                C::ROWS,
                prover.nodes,
                prover.dead.len(),
                prover.lb.len(),
                t0.elapsed().as_secs_f64()
            );
            if cli.trace {
                prover.print_kill_line(init, d);
            }
            verdict = Verdict::Dead { depth: d };
            break;
        }
        if prover.aborted {
            println!(
                "ROWS={:<2} kill-dfs: BUDGET at depth {d} (nodes={}, memo={}+{}, {:.1}s) — \
                 no kill proven or refuted",
                C::ROWS,
                prover.nodes,
                prover.dead.len(),
                prover.lb.len(),
                t0.elapsed().as_secs_f64()
            );
            verdict = Verdict::Exploded;
            break;
        }
        println!(
            "ROWS={:<2} kill-dfs: no kill within {d} plies (nodes={}, memo={}+{}, {:.1}s)",
            C::ROWS,
            prover.nodes,
            prover.dead.len(),
            prover.lb.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    if let Verdict::NoKill { plies } = verdict {
        println!(
            "ROWS={:<2} kill-dfs: the player SURVIVES every <= {plies}-ply assault \
             (survival lower bound; not an ALIVE certificate)",
            C::ROWS
        );
    }
    Ok(RowOutcome {
        rows: C::ROWS as u32,
        reachable: prover.dead.len() + prover.lb.len(),
        edges: prover.nodes as usize,
        adv: verdict,
        coop: None,
        secs: t0.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
enum Variant {
    /// 3 columns, bag = {S, Z, T}.
    Szt3,
    /// 5 columns, bag = {S, Z, T, J}.
    Sztj5,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
enum Engine {
    /// Reachable-arena enumeration + retrograde GFP (exact, memory-bound).
    Gfp,
    /// Iterative-deepening directed DFS for a forced kill (sound DEAD only).
    KillDfs,
    /// Anytime best-first AND-OR: badness-guided expansion, permanent
    /// (board,bag,piece) cache, exact death backprop, final pessimistic sweep.
    Guided,
    /// Continuous fully-parallel AND-OR (the tetris_atlas_inmemory engine):
    /// interleaved expansion + monotone death propagation on all cores,
    /// quiescence = exact fixpoint, root death = early sound DEAD.
    Parallel,
}

impl Variant {
    fn describe(self) -> &'static str {
        match self {
            Variant::Szt3 => "S/Z/T on 3 columns",
            Variant::Sztj5 => "S/Z/T/J on 5 columns",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "tetris_szt3")]
#[command(about = "Exact AND-OR solver for small piece-set/column-count Tetris variants")]
struct Cli {
    /// Which variant game to solve.
    #[arg(long, value_enum, default_value_t = Variant::Szt3)]
    variant: Variant,

    /// Loss-line heights to sweep (comma separated). ALIVE is monotone upward, so the
    /// sweep stops at the first ALIVE ceiling.
    #[arg(long, value_delimiter = ',', default_values_t = [4u32, 5, 6, 7, 8, 10, 12, 16, 20])]
    rows: Vec<u32>,

    /// Max interned states per solve before declaring EXPLODED.
    #[arg(long, default_value_t = 50_000_000)]
    budget: usize,

    /// Player-side hole band: placements burying more holes than this are treated as
    /// unavailable. ALIVE verdicts stay sound for the unrestricted game; DEAD verdicts
    /// become band-relative (inconclusive globally).
    #[arg(long)]
    max_holes: Option<u32>,

    /// Adversary model: `all` is the true AND-quantifier; `sz-first`/`greedy` script a
    /// single deterministic adversary — a player loss against one is a sound DEAD for
    /// the full game, while a survival is only a necessary-condition pass.
    #[arg(long, value_enum, default_value_t = Adversary::All)]
    adversary: Adversary,

    /// Solving engine: `gfp` enumerates the reachable arena and retrogrades (exact,
    /// memory-bound); `kill-dfs` searches the adversary's forcing subtree by
    /// iterative deepening (sound DEAD far beyond gfp's memory wall; a miss is only
    /// a survival lower bound). kill-dfs is adversarial-only and ignores --adversary.
    #[arg(long, value_enum, default_value_t = Engine::Gfp)]
    engine: Engine,

    /// Max iterative-deepening depth (plies) for the kill-dfs engine.
    #[arg(long, default_value_t = 40)]
    kill_depth: u32,

    /// Print a forced-loss trace for DEAD adversarial verdicts.
    #[arg(long)]
    trace: bool,

    /// Max steps in the forced-loss trace.
    #[arg(long, default_value_t = 120)]
    trace_steps: usize,
}

macro_rules! dispatch_rows {
    ($f:ident, $cfg:ident, $rows:expr, $cli:expr, [$($r:literal),+ $(,)?]) => {
        match $rows {
            $( $r => $f::<$cfg<$r>>($cli), )+
            other => bail!(
                "unsupported --rows value {other} (supported: 2..=20; each is a compiled config)"
            ),
        }
    };
}

/// Print the variant header: columns, bag, per-piece placement counts, and the
/// clearing-rate demand implied by the cell arithmetic.
fn print_banner<C: TetrisGameConfig>(variant: Variant)
where
    [(); C::COLS]:,
{
    let placements = placements_for::<C>();
    let pieces: Vec<TetrisPiece> = pieces_in_mask(C::PIECE_SET.into()).collect();
    let cells = 4 * pieces.len();
    let rows_per_bag = cells as f64 / C::COLS as f64;
    println!(
        "=== tetris_szt3 [{}]: {} columns, {} pieces/bag (TetrisGameConfig variant) ===",
        variant.describe(),
        C::COLS,
        pieces.len(),
    );
    let counts = pieces
        .iter()
        .map(|p| format!("{p}={}", placements[p.index() as usize].len()))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "placements per piece: {counts} | cells/bag={cells} = {rows_per_bag:.1} rows -> survival \
         needs {rows_per_bag:.1} clears/bag",
    );
    println!();
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.variant {
        Variant::Szt3 => print_banner::<SztCols3<20>>(cli.variant),
        Variant::Sztj5 => print_banner::<SztjCols5<20>>(cli.variant),
    }

    let mut outcomes: Vec<RowOutcome> = Vec::new();
    for &r in &cli.rows {
        let out = match (cli.variant, cli.engine) {
            (Variant::Szt3, Engine::Gfp) => dispatch_rows!(
                run_one,
                SztCols3,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Szt3, Engine::KillDfs) => dispatch_rows!(
                run_kill,
                SztCols3,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Sztj5, Engine::Gfp) => dispatch_rows!(
                run_one,
                SztjCols5,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Sztj5, Engine::KillDfs) => dispatch_rows!(
                run_kill,
                SztjCols5,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Szt3, Engine::Guided) => dispatch_rows!(
                run_guided,
                SztCols3,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Sztj5, Engine::Guided) => dispatch_rows!(
                run_guided,
                SztjCols5,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Szt3, Engine::Parallel) => dispatch_rows!(
                run_parallel,
                SztCols3,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
            (Variant::Sztj5, Engine::Parallel) => dispatch_rows!(
                run_parallel,
                SztjCols5,
                r,
                &cli,
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
                ]
            ),
        }?;
        let adv = out.adv;
        outcomes.push(out);
        if let Verdict::Alive { .. } = adv {
            println!("\nALIVE at ROWS={r} is monotone upward — stopping the sweep.");
            break;
        }
        if cli.engine == Engine::Gfp {
            if let Verdict::Exploded = adv {
                println!("\nEXPLODED at ROWS={r} — taller ceilings only grow; stopping the sweep.");
                break;
            }
        }
    }

    println!("\n=== summary ===");
    println!(
        "{:>4} {:>12} {:>12} {:>28} {:>28} {:>8}",
        "ROWS", "states", "edges", "adversarial", "cooperative", "secs"
    );
    for o in &outcomes {
        let fmt = |v: Verdict| match v {
            Verdict::Alive { carrier } => format!("ALIVE (carrier {carrier})"),
            Verdict::Dead { depth } => format!("DEAD (depth {depth})"),
            Verdict::NoKill { plies } => format!("no kill <= {plies} plies"),
            Verdict::Exploded => "EXPLODED".to_string(),
        };
        println!(
            "{:>4} {:>12} {:>12} {:>28} {:>28} {:>8.2}",
            o.rows,
            o.reachable,
            o.edges,
            fmt(o.adv),
            o.coop.map_or("-".to_string(), fmt),
            o.secs
        );
    }

    let alive_r = outcomes.iter().find_map(|o| match o.adv {
        Verdict::Alive { .. } => Some(o.rows),
        _ => None,
    });
    let max_dead = outcomes
        .iter()
        .filter(|o| matches!(o.adv, Verdict::Dead { .. }))
        .map(|o| o.rows)
        .max();
    println!();
    let vname = cli.variant.describe();
    let alive_sound = cli.adversary == Adversary::All;
    let dead_sound = cli.max_holes.is_none();
    let scripted = if alive_sound {
        ""
    } else {
        " (a single scripted adversary suffices)"
    };
    match (alive_r, max_dead) {
        (Some(r), _) if alive_sound => println!(
            "VERDICT: {vname} IS playable forever (adversarially) — a closed \
             carrier exists under ceiling {r}, hence under any taller board."
        ),
        (Some(r), _) => println!(
            "VERDICT: the player survives the scripted adversary under ceiling {r} — a \
             necessary condition only; the full adversary may still win."
        ),
        (None, Some(d)) if dead_sound && d >= 20 => println!(
            "VERDICT: {vname} is NOT playable forever at the canonical \
             20-row ceiling — a top-out is forced{scripted}."
        ),
        (None, Some(d)) if dead_sound => println!(
            "VERDICT so far: {vname} is adversarially DEAD for every ceiling up to {d} rows \
             (each sound){scripted}; no ALIVE ceiling found in this sweep."
        ),
        (None, Some(_)) => println!(
            "VERDICT: no carrier found inside the holes<={} band in this sweep \
             (band-relative DEAD; inconclusive for the unrestricted game).",
            cli.max_holes.unwrap_or(0)
        ),
        (None, None) => println!("VERDICT: no conclusive result in this sweep."),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// O-only on 2 columns: O tiles the width exactly, so every drop clears its two
    /// rows immediately and the game loops on the empty board forever. The solver
    /// must find ALIVE with a tiny carrier (this also exercises self-loop edges).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct OCols2<const ROWS: usize>;

    impl<const ROWS: usize> TetrisGameConfig for OCols2<ROWS> {
        const ROWS: usize = ROWS;
        const COLS: usize = 2;
        const PIECE_SET: TetrisPieceBagState =
            TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE]);
    }

    #[test]
    fn o_only_two_cols_is_alive_with_tiny_carrier() -> Result<()> {
        let g = Graph::<OCols2<4>>::build(1_000, None, Adversary::All);
        assert!(!g.exploded);
        for mode in [Mode::Adversarial, Mode::Cooperative] {
            let s = solve(&g, mode, false);
            assert!(
                s.alive[0],
                "O-only 2-col must be alive in {} mode",
                mode.name()
            );
            verify_carrier(&g, &s, mode)?;
        }
        assert!(
            g.keys.len() <= 4,
            "O-only 2-col carrier should be tiny, got {}",
            g.keys.len()
        );
        Ok(())
    }

    fn check_placement_geometry<C: TetrisGameConfig>()
    where
        [(); C::COLS]:,
    {
        // For each piece, the filtered count must equal sum over distinct rotations of
        // (COLS - width + 1) when the piece fits. The distinct rotations are discovered
        // from the engine's full-width table (every rotation fits at width 10).
        let placements = placements_for::<C>();
        for piece in pieces_in_mask(C::PIECE_SET.into()) {
            let mut widths: Vec<usize> = Vec::new();
            let mut seen = Vec::new();
            for &pl in TetrisPiecePlacement::all_from_piece(piece) {
                if !seen.contains(&pl.orientation.rotation) {
                    seen.push(pl.orientation.rotation);
                    widths.push(pl.piece.width(pl.orientation.rotation) as usize);
                }
            }
            assert_eq!(seen.len(), piece.num_rotations() as usize);
            let expected: usize = widths
                .iter()
                .map(|&w| if w <= C::COLS { C::COLS - w + 1 } else { 0 })
                .sum();
            assert_eq!(
                placements[piece.index() as usize].len(),
                expected,
                "placement count mismatch for {piece}"
            );
        }
    }

    #[test]
    fn placement_filter_matches_engine_geometry() {
        check_placement_geometry::<SztCols3<8>>();
        check_placement_geometry::<SztjCols5<8>>();
    }

    #[test]
    fn width3_t_placements_clear_a_row_from_empty() {
        // Both width-3 T orientations on the empty 3-col board complete a full row:
        // 4 cells minus 3 cleared leaves exactly 1 residual cell.
        let placements = placements_for::<SztCols3<8>>();
        let mut width3 = 0;
        for &pl in &placements[TetrisPiece::T_PIECE.index() as usize] {
            if pl.piece.width(pl.orientation.rotation) != 3 {
                continue;
            }
            width3 += 1;
            let mut b = TetrisBoard::<SztCols3<8>>::new();
            let res = b.apply_piece_placement(pl);
            assert!(!bool::from(res.is_lost));
            assert_eq!(res.lines_cleared, 1, "width-3 T must clear one row: {pl}");
            assert_eq!(b.count(), 1, "one residual cell after the clear: {pl}");
        }
        assert!(width3 >= 1, "T must have a width-3 placement in 3 columns");
    }

    #[test]
    fn variant_placement_matches_standard_board_prefix() {
        // S and Z drops from empty never complete a 3-wide row, so the variant board
        // must match the first three columns of the same placement applied on the
        // canonical 10-column board — placement application is config-generic.
        let placements = placements_for::<SztCols3<20>>();
        for piece in [TetrisPiece::S_PIECE, TetrisPiece::Z_PIECE] {
            for &pl in &placements[piece.index() as usize] {
                let mut narrow = TetrisBoard::<SztCols3<20>>::new();
                let res_n = narrow.apply_piece_placement(pl);
                let mut wide = TetrisBoard::<tetris_game::StandardTetris>::new();
                let res_w = wide.apply_piece_placement(pl);
                assert!(!bool::from(res_n.is_lost) && !bool::from(res_w.is_lost));
                assert_eq!(res_n.lines_cleared, 0);
                assert_eq!(res_w.lines_cleared, 0);
                assert_eq!(
                    narrow.as_limbs().as_slice(),
                    &wide.as_limbs()[..3],
                    "variant/standard divergence for {pl}"
                );
            }
        }
    }

    #[test]
    fn kill_prover_agrees_with_gfp() {
        // O-only 2 cols is ALIVE: no kill exists at any depth.
        let mut prover = KillProver::<OCols2<4>>::new(100_000, None);
        let init = (TetrisBoard::<OCols2<4>>::new(), prover.full_bag);
        for d in 1..=12 {
            assert!(!prover.kill_within(init, d), "spurious kill at depth {d}");
            assert!(!prover.aborted);
        }
        // S/Z/T on 3 cols at ROWS=4 is DEAD (GFP-exact): a forced kill must exist.
        let mut prover = KillProver::<SztCols3<4>>::new(1_000_000, None);
        let init = (TetrisBoard::<SztCols3<4>>::new(), prover.full_bag);
        let found = (1..=12).find(|&d| prover.kill_within(init, d));
        assert!(!prover.aborted);
        assert!(found.is_some(), "kill-dfs must find the GFP-proven kill");
    }

    #[test]
    fn guided_solver_agrees_with_gfp() {
        // O-only 2 cols: the heap must drain (finite closed graph) and the exact
        // GFP over the exhausted graph must find the root alive.
        let mut s = GuidedSolver::<OCols2<4>>::new(None);
        let mut root_dead = false;
        while let Some((_, id)) = s.heap.pop() {
            root_dead |= s.expand(id);
        }
        assert!(!root_dead);
        let (alive, carrier) = s.pessimistic_sweep();
        assert!(alive, "O-only 2-col root must be alive in the exact GFP");
        assert!(carrier >= 1);

        // S/Z/T on 3 cols at ROWS=4 is DEAD: guided expansion + backprop must
        // kill the root without exhausting any budget.
        let mut s = GuidedSolver::<SztCols3<4>>::new(None);
        let mut root_dead = false;
        while let Some((_, id)) = s.heap.pop() {
            if s.expand(id) {
                root_dead = true;
                break;
            }
        }
        assert!(root_dead, "guided solver must prove szt3 R=4 dead");
    }

    /// I-only on ONE column: the vertical I is the only placement; every cell it
    /// places is a full 1-wide row, so it clears instantly back to empty at ANY
    /// ceiling (the engine clears before the loss check). Exactly ALIVE, carrier 1.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct ICols1<const ROWS: usize>;

    impl<const ROWS: usize> TetrisGameConfig for ICols1<ROWS> {
        const ROWS: usize = ROWS;
        const COLS: usize = 1;
        const PIECE_SET: TetrisPieceBagState =
            TetrisPieceBagState::from_pieces([TetrisPiece::I_PIECE]);
    }

    /// O-only on THREE columns: O is 2 wide, a clear needs all 3 columns at one
    /// row, so no row can EVER clear — cells accumulate monotonically and every
    /// ceiling is DEAD (even cooperatively).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct OCols3<const ROWS: usize>;

    impl<const ROWS: usize> TetrisGameConfig for OCols3<ROWS> {
        const ROWS: usize = ROWS;
        const COLS: usize = 3;
        const PIECE_SET: TetrisPieceBagState =
            TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE]);
    }

    /// O+I on TWO columns: a mixed fixture with both alive and dead reachable
    /// states (a vertical I builds a 4-tall tower that bad play tops out on, but
    /// O clears flush pairs and a second I completes the tower's rows).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct OICols2<const ROWS: usize>;

    impl<const ROWS: usize> TetrisGameConfig for OICols2<ROWS> {
        const ROWS: usize = ROWS;
        const COLS: usize = 2;
        const PIECE_SET: TetrisPieceBagState =
            TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE, TetrisPiece::I_PIECE]);
    }

    /// Reference verdict + alive set from the exact sequential GFP.
    fn gfp_reference<C: TetrisGameConfig>() -> (bool, Vec<StateKey<C>>)
    where
        [(); C::COLS]:,
    {
        let g = Graph::<C>::build(1_000_000, None, Adversary::All);
        assert!(!g.exploded, "reference GFP must solve exactly");
        let s = solve(&g, Mode::Adversarial, false);
        let alive_keys = g
            .keys
            .iter()
            .enumerate()
            .filter(|(i, _)| s.alive[*i])
            .map(|(_, &k)| k)
            .collect();
        (s.alive[0], alive_keys)
    }

    /// Run the parallel engine to its verdict and assert full agreement with the
    /// GFP reference: root verdict, alive-set soundness (no false-alive), and a
    /// re-verified carrier when alive.
    fn assert_parallel_matches_gfp<C: TetrisGameConfig + 'static>(workers: usize)
    where
        [(); C::COLS]:,
    {
        let (gfp_root_alive, gfp_alive) = gfp_reference::<C>();
        let engine = parallel::Engine::<C>::new(1_000_000, None);
        let (root_dead, exact) = engine.run(workers);
        assert!(exact, "parallel engine must quiesce on small arenas");
        assert_eq!(
            !root_dead,
            gfp_root_alive,
            "root verdict mismatch (COLS={} ROWS={})",
            C::COLS,
            C::ROWS
        );
        if gfp_root_alive {
            let carrier = engine.verify_alive_carrier();
            assert!(carrier.is_ok(), "carrier must re-verify: {carrier:?}");
            // Soundness: nothing the parallel engine calls alive is dead in truth.
            use std::collections::HashSet;
            let truth: HashSet<StateKey<C>> = gfp_alive.into_iter().collect();
            for key in engine.alive_keys_for_test() {
                assert!(truth.contains(&key), "parallel claims a dead state alive");
            }
        }
    }

    #[test]
    fn i_only_one_col_clears_instantly_alive() {
        // Hand-proven: the 4 cells of a vertical I are 4 full rows -> immediate
        // perfect clear, even under a 2-row ceiling (clears precede the loss check).
        let (root_alive, alive) = gfp_reference::<ICols1<2>>();
        assert!(root_alive);
        assert_eq!(alive.len(), 1, "carrier is exactly the empty board");
        assert_parallel_matches_gfp::<ICols1<2>>(4);
    }

    #[test]
    fn o_only_three_cols_never_clears_dead() {
        // Hand-proven: no clear is geometrically possible -> monotone fill -> dead.
        let (root_alive, _) = gfp_reference::<OCols3<4>>();
        assert!(!root_alive);
        assert_parallel_matches_gfp::<OCols3<4>>(4);
        assert_parallel_matches_gfp::<OCols3<6>>(4);
        // kill-dfs agrees and the kill is shallow (3 O's overflow a 4-row board).
        let mut prover = KillProver::<OCols3<4>>::new(100_000, None);
        let init = (TetrisBoard::<OCols3<4>>::new(), prover.full_bag);
        let found = (1..=8).find(|&d| prover.kill_within(init, d));
        assert!(
            found.is_some_and(|d| d <= 5),
            "kill must be shallow: {found:?}"
        );
    }

    #[test]
    fn mixed_oi_two_cols_engines_agree() {
        // Mixed alive/dead arena: agreement must hold on the root AND on alive
        // membership, at several ceilings and worker counts.
        assert_parallel_matches_gfp::<OICols2<4>>(1);
        assert_parallel_matches_gfp::<OICols2<6>>(4);
        assert_parallel_matches_gfp::<OICols2<8>>(8);
    }

    #[test]
    fn parallel_verdicts_match_gfp_on_variant_games() {
        assert_parallel_matches_gfp::<SztCols3<4>>(4);
        assert_parallel_matches_gfp::<SztCols3<5>>(4);
        assert_parallel_matches_gfp::<SztCols3<6>>(8);
        assert_parallel_matches_gfp::<OCols2<4>>(2);
    }

    #[test]
    fn parallel_verdict_deterministic_across_workers_and_reps() {
        // Race hunting: the verdict must be identical across repetitions and
        // worker counts; for death-free arenas the state count must be too.
        let mut verdicts = Vec::new();
        for rep in 0..12 {
            let workers = [1, 2, 8][rep % 3];
            let engine = parallel::Engine::<SztCols3<6>>::new(1_000_000, None);
            verdicts.push(engine.run(workers));
        }
        assert!(
            verdicts.iter().all(|&v| v == verdicts[0]),
            "nondeterministic verdicts: {verdicts:?}"
        );
        let mut counts = Vec::new();
        for rep in 0..6 {
            let workers = [1, 8][rep % 2];
            let engine = parallel::Engine::<ICols1<2>>::new(1_000, None);
            let (root_dead, exact) = engine.run(workers);
            assert!(!root_dead && exact);
            counts.push(engine.state_count());
        }
        assert!(counts.iter().all(|&c| c == counts[0]));
    }

    #[test]
    fn parallel_budget_finalize_is_conservative() {
        // A budget stop on an all-dead arena must still end with a dead root
        // (any surviving closed set would contradict the exact GFP), and the
        // engine must flag the run as non-exact unless the death was genuine.
        let engine = parallel::Engine::<SztCols3<8>>::new(100, None);
        let (root_dead, _exact) = engine.run(4);
        assert!(root_dead, "no closed alive set exists inside a dead arena");
    }

    #[test]
    fn carrier_verifier_detects_corruption() {
        // Kill every alive state except the root without propagation: the root's
        // obligations now dangle, and the independent verifier must reject it.
        let engine = parallel::Engine::<OICols2<6>>::new(1_000_000, None);
        let (root_dead, exact) = engine.run(4);
        assert!(!root_dead && exact, "fixture must be alive");
        assert!(engine.verify_alive_carrier().is_ok());
        let root = (
            TetrisBoard::<OICols2<6>>::new(),
            u8::from(TetrisPiece::O_PIECE) | u8::from(TetrisPiece::I_PIECE),
        );
        for key in engine.alive_keys_for_test() {
            if key != root {
                engine.force_dead_for_test(&key);
            }
        }
        assert!(
            engine.verify_alive_carrier().is_err(),
            "verifier must reject a corrupted carrier"
        );
    }

    #[test]
    fn parallel_engine_agrees_with_gfp() {
        // O-only 2 cols: quiesces exactly, root ALIVE, carrier re-verified.
        let engine = parallel::Engine::<OCols2<4>>::new(1_000, None);
        let (root_dead, exact) = engine.run(2);
        assert!(!root_dead);
        assert!(exact);
        let carrier = engine.verify_alive_carrier();
        assert!(carrier.is_ok_and(|c| c >= 1));

        // S/Z/T on 3 cols at ROWS=4: GFP-exact DEAD; the parallel engine must
        // prove the root dead (monotone early-exit or at quiescence).
        let engine = parallel::Engine::<SztCols3<4>>::new(10_000, None);
        let (root_dead, _) = engine.run(2);
        assert!(root_dead, "parallel engine must prove szt3 R=4 dead");
    }

    #[test]
    fn bag_mask_refills_to_piece_set() {
        let full: u8 = SztCols3::<8>::PIECE_SET.into();
        assert_eq!(pieces_in_mask(full).count(), 3);
        let mut mask = full;
        for piece in [TetrisPiece::S_PIECE, TetrisPiece::T_PIECE] {
            mask &= !u8::from(piece);
        }
        assert_eq!(
            pieces_in_mask(mask).collect::<Vec<_>>(),
            vec![TetrisPiece::Z_PIECE]
        );
        mask &= !u8::from(TetrisPiece::Z_PIECE);
        assert_eq!(mask, 0, "bag exhausted -> caller refills to PIECE_SET");
    }
}
