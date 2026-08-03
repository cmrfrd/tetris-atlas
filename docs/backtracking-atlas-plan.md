# Backtracking Atlas — Implementation Plan

## Goal

Upgrade `tetris_atlas_narrow` from a forward-only greedy atlas builder to a
**backtracking atlas** that can prove solvability or unsolvability of the
constrained Tetris game. When the placement heuristic picks a bad move that
eventually leads to a dead end, the system backtracks and tries alternatives
until either a complete surviving atlas is found or the root is proven dead.

## Current Architecture

```
Frontier(board, bag) → Dispatcher → Channel → Workers → process_board_state
                                                              ↓
                                                     beam_search picks ONE placement
                                                              ↓
                                                     lookup[(board, piece)] = orientation
                                                              ↓
                                                     frontier.push(new_board, new_bag)
```

**Problem:** If the chosen placement eventually leads to a dead end, there's
no way to undo it and try a different placement.

## New Architecture

### Phase 1: Forward Expansion (modified)

Same as before, but:
- Generate ALL valid placements for each (board, piece), sorted by beam
  search score
- Store the chosen placement + alternatives
- Track reverse edges (child → parent)

### Phase 2: Death Propagation (new)

When a (board, bag) state is dead (all pieces lead to loss or dead states):
- Try the next alternative placement for the parent's (board, piece)
- If alternatives exhausted, mark the parent dead too → propagate upward

## Data Structure Changes

### 1. Replace `lookup` value type

```rust
// Before:
lookup: DashMap<(TetrisBoard, TetrisPiece), TetrisPieceOrientation>

// After:
lookup: DashMap<(TetrisBoard, TetrisPiece), PlacementEntry>

struct PlacementEntry {
    chosen: TetrisPieceOrientation,
    alternatives: Vec<TetrisPieceOrientation>,  // remaining untried, best-first
    result_board: TetrisBoard,
    result_bag: TetrisPieceBagState,
}
```

### 2. Add reverse edge map

```rust
/// child (board, bag) → list of (parent_board, parent_piece) that created it
parents: DashMap<(TetrisBoard, TetrisPieceBagState), Vec<(TetrisBoard, TetrisPiece)>>
```

Populated during forward expansion: when we insert a lookup entry, we also
record the reverse edge.

### 3. Add dead state set

```rust
dead_states: DashSet<(TetrisBoard, TetrisPieceBagState)>
```

### 4. Add death propagation queue

```rust
death_queue: SegQueue<(TetrisBoard, TetrisPieceBagState)>  // lock-free MPMC
```

Workers push dead states here. A dedicated propagation thread (or the
dispatcher) processes them.

### 5. Stats additions

```rust
struct SharedStats {
    // ... existing fields ...
    states_died: AtomicU64,           // states marked dead
    backtrack_retries: AtomicU64,     // times we tried an alternative
    backtrack_successes: AtomicU64,   // alternatives that led to alive states
    death_propagations: AtomicU64,    // total death propagation steps
}
```

## Modified `process_board_state`

```rust
fn process_board_state(board, bag, ...) -> Vec<(TetrisBoard, TetrisPieceBagState)> {
    let mut new_states = Vec::new();

    for (piece, bag_state) in bag.iter_next_states() {
        if !is_allowed_piece(piece) { continue; }
        if lookup.contains_key(&(board, piece)) { continue; }

        // Generate ALL valid placements, sorted by score (not just beam winner)
        let all_placements = generate_scored_placements(board, piece, beam_search);
        if all_placements.is_empty() {
            // No valid placement exists for this piece — record as dead-on-arrival
            continue;
        }

        let chosen = all_placements[0];
        let alternatives = all_placements[1..].to_vec();

        // Apply chosen
        game.board = board;
        let result = game.apply_piece_placement(chosen);
        let new_board = game.board;

        // Skip if result board is already known dead
        if dead_states.contains(&(new_board, bag_state)) {
            // Try alternatives immediately (inline backtrack)
            if let Some(entry) = try_alternatives(board, piece, &alternatives, bag_state, ...) {
                // Found a non-dead alternative
                lookup.insert((board, piece), entry);
                parents[(entry.result_board, bag_state)].push((board, piece));
                new_states.push((entry.result_board, bag_state));
            }
            // else: all alternatives dead → this (board, piece) is dead
            continue;
        }

        let entry = PlacementEntry {
            chosen: chosen.orientation,
            alternatives,
            result_board: new_board,
            result_bag: bag_state,
        };
        lookup.insert((board, piece), entry);
        parents[(new_board, bag_state)].push((board, piece));
        new_states.push((new_board, bag_state));
    }

    // Check: are ALL pieces for this bag dead or unplayable?
    let all_dead = bag.iter_next_states()
        .filter(|(p, _)| is_allowed_piece(*p))
        .all(|(piece, bag_state)| {
            !lookup.contains_key(&(board, piece))  // no entry = dead/unplayable
            || dead_states.contains(&(lookup[(board, piece)].result_board, bag_state))
        });
    if all_dead {
        dead_states.insert((board, bag));
        death_queue.push((board, bag));
    }

    new_states
}
```

## New: `generate_scored_placements`

```rust
fn generate_scored_placements(
    board: TetrisBoard,
    piece: TetrisPiece,
    beam_search: &mut ...,
) -> Vec<TetrisPieceOrientation> {
    let mut results = Vec::new();

    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut test_board = board;
        let result = test_board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST { continue; }
        // Apply filter_fn if configured
        if !filter_fn_check(test_board) { continue; }

        let score = scorer(test_board, result.lines_cleared);
        results.push((placement.orientation, score));
    }

    // Sort by score descending (best first)
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results.into_iter().map(|(o, _)| o).collect()
}
```

This replaces the beam search for placement selection with direct enumeration
of all valid placements scored by the heuristic. Beam search was needed for
the 10-column game where looking ahead matters; for the 4-column game, a simple
1-ply scorer ranking is enough to define a deterministic exploration order.

## New: Death Propagation Thread

```rust
fn death_propagation_thread(
    death_queue: Arc<SegQueue<(TetrisBoard, TetrisPieceBagState)>>,
    lookup: Arc<DashMap<(TetrisBoard, TetrisPiece), PlacementEntry>>,
    parents: Arc<DashMap<(TetrisBoard, TetrisPieceBagState), Vec<...>>>,
    dead_states: Arc<DashSet<(TetrisBoard, TetrisPieceBagState)>>,
    frontier: Arc<PriorityFrontier>,
    stats: Arc<SharedStats>,
) {
    loop {
        let Some((dead_board, dead_bag)) = death_queue.pop() else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };

        stats.death_propagations.fetch_add(1, Ordering::Relaxed);

        // Find all parents that pointed to this dead state
        let Some(parent_list) = parents.get(&(dead_board, dead_bag)) else {
            continue;
        };

        for &(parent_board, parent_piece) in parent_list.value() {
            let Some(mut entry) = lookup.get_mut(&(parent_board, parent_piece)) else {
                continue;
            };

            // Current chosen led to dead state — try next alternative
            while let Some(next_orientation) = entry.alternatives.pop() {
                stats.backtrack_retries.fetch_add(1, Ordering::Relaxed);

                let placement = TetrisPiecePlacement { piece: parent_piece, orientation: next_orientation };
                let mut test_board = parent_board;
                let result = test_board.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST { continue; }

                let new_bag = entry.result_bag; // bag_state is same regardless of placement
                if dead_states.contains(&(test_board, new_bag)) { continue; }

                // Found a surviving alternative!
                entry.chosen = next_orientation;
                entry.result_board = test_board;
                stats.backtrack_successes.fetch_add(1, Ordering::Relaxed);

                // Update parent tracking
                parents.entry((test_board, new_bag))
                    .or_default()
                    .push((parent_board, parent_piece));

                // Re-add to frontier for further expansion
                frontier.push(test_board, new_bag);
                break;
            }

            // If no alternatives left, check if parent (board, bag) is now fully dead
            // (needs checking all pieces for the parent's bag)
            // ... same all_dead check as in process_board_state ...
        }
    }
}
```

## Termination Conditions

1. **Frontier empty + death queue empty + no in-flight work:**
   - If `dead_states` contains root → **UNSOLVABLE** (proven)
   - If `dead_states` does NOT contain root → **SOLVED** (complete atlas)

2. **Ctrl+C:** Save state (including dead_states, parents, alternatives)
   and resume later.

## Logging Changes

```
t=10.0s boards=50000 lookup=45000 frontier=5000 dead=200 retries=150 | ...
```

Add `dead`, `retries`, `backtrack_ok` columns to the periodic stats output
and CSV logger.

## Implementation Order

1. **Add `PlacementEntry` struct** — replace `TetrisPieceOrientation` in lookup
2. **Add `parents` map** — populate during forward expansion
3. **Add `dead_states` set + `death_queue`** — populate when all pieces dead
4. **Add `generate_scored_placements`** — enumerate all placements with scores
5. **Modify `process_board_state`** — store alternatives, check dead children
6. **Add death propagation thread** — consume death_queue, try alternatives
7. **Update stats + logging** — new counters for dead/retries/propagations
8. **Update save/load** — persist PlacementEntry, dead_states, parents
9. **Add termination detection** — check root dead / atlas complete
10. **Test with tight constraints** — verify unsolvability detection works

## Risks / Notes

- **Memory:** `parents` map and `alternatives` vecs add memory overhead.
  For the narrow game (~100K states) this is fine. For the full game it
  would need careful budgeting.

- **Concurrency:** Death propagation modifies lookup entries that workers
  might be reading. Use `DashMap::get_mut` for atomic updates. The death
  propagation thread should be single-threaded to avoid races on the
  alternative selection.

- **Save/load:** The new data structures (PlacementEntry with Vec,
  parents map) need serialization. More complex than the current format
  but straightforward with bincode or manual serialization.

- **Ordering matters:** Better initial placement ordering = less
  backtracking. The direct placement scorer determines how quickly we find
  surviving paths. Poor ordering means more wasted work before finding
  the right placement.
