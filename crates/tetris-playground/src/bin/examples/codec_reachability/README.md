# Codec reachability sampler

`tetris_codec_reachability` continuously samples the board codec, plays legal seven-bag histories
from the empty board, and records independently replayable evidence for refining the carrier.
It uses the canonical hard-drop engine. It does not modify codec membership.

Build once, then run the executable directly from the repository root:

```sh
cargo build --release -p tetris-playground --bin tetris_codec_reachability
target/release/tetris_codec_reachability run --output artifacts/data/codec-study
# Ctrl-C finishes the current batch and saves a checkpoint.
target/release/tetris_codec_reachability run --output artifacts/data/codec-study --resume
target/release/tetris_codec_reachability verify --output artifacts/data/codec-study
```

The default runs until interrupted, with periodic checkpoints every 60 seconds. Only one writer
can open a dataset; stop it before verifying. A bounded example:

```sh
target/release/tetris_codec_reachability run \
  --output artifacts/data/codec-study-bounded --batches 100 --threads 4 --controls
```

Use `run --help` for options. Defaults per batch are 10,000 uniformly sampled parent codec IDs,
all legal placements on each parent, 1,000 episodes **per play mode**, and 10,000 fresh uniform
candidate boards. Each episode has at most 128 pieces. The default seed is `0xb04dc0de`.
`--controls` injects 216 previously proved impossible examples in the first batch; leave it off
for an unbiased count of newly sampled findings. Episodes use random legal draws and placements;
relaxed play favors reducing holes while outside the codec. This is discovery sampling, not a
uniform distribution over reachable boards.

## Interpreting evidence

The four exact bitmaps deliberately preserve different facts:

| Observation | What it establishes |
|---|---|
| Incoming hit | Some sampled codec board has a legal geometric transition to this board. The parent may itself be unreachable; bag history is not established. |
| Strict witness | A legal seven-bag history from empty stays within the height and hole caps and current codec. |
| Relaxed witness | A legal seven-bag history stays within the height cap, but may temporarily exceed the hole budget before returning. |
| Proved impossible | The last-move exclusion below succeeds, independently of sampling coverage. |
| No observation | Unknown. It must not be removed on this evidence. |

A witnessed board is reachable; that does not establish an infinite-survival policy. Conversely,
millions of missed samples do not establish unreachability. Counters are cumulative observations,
not a certified exhaustive Atlas. Incoming transitions include every available piece without
requiring a reachable seven-bag predecessor.

The implemented impossibility rule applies to width greater than four. If the final board reaches
the height cap and has multiple separated runs in its top row, the last move could not have cleared
a row: after a clear, old cells would be below the cap and that top row would consist of one
contiguous tetromino row. The checker exhaustively removes every translated tetromino subset and
replays its hard drop. If none gives a valid zero-clear predecessor with no internal empty row,
the board is unreachable from empty under this height cap. Earlier hole excursions cannot rescue
it. The reference checker uses engine placements and is rerun before evidence is saved and by
`verify`. See [the codec reachability analysis](../../../../../tetris-codec/REACHABILITY.md).

## Files and restart behavior

Each output directory contains:

- `manifest.json`: model, workload, source fingerprints, original command, and Git provenance.
- `findings.jsonl`: raw columns for every discovered impossibility finding and selected reachable
  witnesses. Witnesses include piece/rotation/column actions and remaining bag. Up to four random
  paths per mode per batch are saved by default; a first hit on a watched target is always saved.
- `candidates.csv`: a fixed uniform watchlist (10,000 by default), with evolving incoming,
  strict, relaxed, and impossible flags, plus raw columns. Missing flags mean unknown.
- `checkpoint.bin`: checksummed RNG streams, counters, watchlist, four bitmaps, and committed
  journal offset. It is written atomically after flushing the journal.
- `summary.json`: cumulative totals and the most recent invocation's elapsed seconds and workers.

Resume requires matching workload and compiled engine/codec/sampler fingerprints. Specify the same
sampling options again; thread count, batch/time limits, and checkpoint interval can change.
`--batches` means additional batches in this invocation. Changing code or codec parameters requires
a fresh dataset. Keep the old binary to continue old datasets. `verify` can replay old raw findings
after codec IDs or sampler code change, provided the engine and physical model still match; numeric
codec IDs are descriptive, while raw boards and actions are authoritative. Strict witnesses also
check admission of every prefix by the current codec.

After a crash, bytes beyond the last durable checkpoint's journal offset are preserved in
`uncheckpointed-*.jsonl.part`, then rolled back and deterministically resampled. A corrupt checkpoint
or shortened journal causes an error, never a silent reset. Source fingerprints and checksums detect
accidental mismatch/corruption; they are not cryptographic authentication.

## Configuration and cost

Change `Game`, `ROWS`, and `MAX_HOLES` at the top of `state.rs` and rebuild to investigate another
codec. Defaults are `BoardCodec<StandardTetris, 4, 2>`: 392,431,346 IDs. This proof implementation
supports standard seven tetrominoes at widths 5–10. The four bitmaps use **196,215,680 bytes
(187.1 MiB)** at default parameters. They are shared by workers; there is no per-worker universe
copy. Larger carriers can become impractical; `--max-bitmap-mib` defaults to 512 and rejects an
oversized allocation. Paths and the growing journal add storage beyond bitmap memory.

`--threads 0` selects up to four available workers. Parent expansions run in parallel with fixed
input IDs, shared atomic bitmaps, and local reductions; legal-play streams stay deterministic.
See [PERFORMANCE.md](PERFORMANCE.md) for the measured baseline, scaling, and validation.

The default codec now excludes every board caught by the implemented last-move rule;
that checker should find no further default-carrier exclusions of the same class. Sampling
still records legal coverage and unknown targets for developing additional proven rules.
Old datasets require the matching old binary for resume; raw findings remain verifiable.

After a reviewed engine-source refactor, `verify --output DIR --recheck-engine` explicitly
replays raw facts against the current engine instead of requiring the old source fingerprint.
Review the engine change first; this does not establish equivalence with an old engine contract.
Model dimensions and complete action/proof replay remain mandatory. Normal verify stays strict,
and resume never bypasses source/workload matching. The rule-derived codec has no exclusion
assets to regenerate; sampling is for discovering additional necessary rules.
