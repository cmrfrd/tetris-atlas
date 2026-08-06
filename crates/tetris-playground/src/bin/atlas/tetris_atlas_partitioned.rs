#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! Partition-owned, disk-backed greedy closed-carrier atlas builder.

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;
use rocksdb::statistics::StatsLevel;
use rocksdb::{
    AsColumnFamilyRef, BlockBasedIndexType, BlockBasedOptions, BoundColumnFamily, Cache,
    ColumnFamilyDescriptor, DB, DBCompactionStyle, IteratorMode, Options, ReadOptions,
    SliceTransform, WriteBatch, WriteBufferManager, WriteOptions,
};
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::cmp::Ordering;
use std::fs;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_search::{TetrisBoardScoreState, height_mse_board_score};
use tetris_utils::HeaplessVec;

#[global_allocator]
static GLOBAL_ALLOCATOR: MiMalloc = MiMalloc;

const PACKED_COLUMN_BYTES: usize = 3;
const STATE_KEY_LEN: usize = StandardTetris::COLS * PACKED_COLUMN_BYTES + 1;
const NODE_RECORD_LEN: usize = TetrisPiece::NUM_PIECES;
const REDGE_KEY_LEN: usize = STATE_KEY_LEN * 2 + 2;
const WORK_KEY_LEN: usize = 11;
const VIRTUAL_SHARDS: usize = 256;
const COMMAND_KIND_COUNT: usize = 6;

const SLOT_PENDING_BASE: u8 = 0x40;
const SLOT_UNEXPANDED: u8 = 0xFE;
const SLOT_EXHAUSTED: u8 = 0xFF;

const PRIORITY_DEATH: u8 = 0;
const PRIORITY_EDGE: u8 = 1;
const PRIORITY_EXPAND: u8 = 2;
const WORK_PRIORITY_COUNT: usize = PRIORITY_EXPAND as usize + 1;

const CF_NODE: &str = "node";
const CF_REDGE: &str = "redge";
const CF_WORK: &str = "work";

const META_SCHEMA_VERSION: &[u8] = b"schema-version";
const META_STATUS: &[u8] = b"status";
const META_OWNER_MAP: &[u8] = b"owner-map";
const META_CARRIER: &[u8] = b"carrier";
const META_SCORER: &[u8] = b"scorer";
const META_SOURCE_COMMIT: &[u8] = b"source-commit";
const META_SUMMARY: &[u8] = b"summary";
const META_PRODUCER_SEQUENCE_PREFIX: &[u8; 13] = b"producer-seq-";
const META_WORK_HEAD_PREFIX: &[u8; 10] = b"work-head-";
const SCHEMA_VERSION: &[u8] = b"partitioned-greedy-v2";
const STATUS_BUILDING: &[u8] = b"building";
const STATUS_PAUSED: &[u8] = b"paused";
const STATUS_COMPLETE: &[u8] = b"complete";
const STATUS_SEED_DEAD: &[u8] = b"seed-dead";
const STATUS_ABORTED: &[u8] = b"aborted";

const WORK_BATCH_COMMANDS: usize = 1024;
const MAX_CANDIDATES_PER_COMMAND: usize = TetrisPiecePlacement::NUM_PLACEMENTS;
const MAX_CANDIDATES_PER_BATCH: usize = WORK_BATCH_COMMANDS * MAX_CANDIDATES_PER_COMMAND;
const MAX_CHOICE_REQUESTS_PER_BATCH: usize = WORK_BATCH_COMMANDS * TetrisPiece::NUM_PIECES;
const MAX_REDGES_PER_DRAIN: usize = 4_096;
const MAX_WRITE_BATCH_BYTES: usize = 8 * 1024 * 1024;
const MAX_COMMAND_WRITE_BYTES: usize = 700_000;
const MAX_NODE_OVERLAY_ENTRIES: usize = MAX_CANDIDATES_PER_BATCH + WORK_BATCH_COMMANDS;
const MAX_REDGE_MUTATIONS_PER_BATCH: usize = 16_384;
const MAX_OUTPUT_COMMANDS_PER_BATCH: usize = 16_384;
const WORK_RECLAIM_SEQUENCE_QUANTUM: u64 = 1 << 20;
const _: () = assert!(
    WORK_RECLAIM_SEQUENCE_QUANTUM > 1 && WORK_RECLAIM_SEQUENCE_QUANTUM.is_power_of_two(),
    "work reclamation quantum must be a power of two greater than one"
);
const MIB_BYTES: usize = 1024 * 1024;
const DEFAULT_RAM_MIB: usize = 2048;
const PER_WRITER_MEMORY_RESERVE_BYTES: usize = 48 * MIB_BYTES;
const WRITER_STACK_BYTES: usize = 2 * MIB_BYTES;
const _: () = assert!(
    PER_WRITER_MEMORY_RESERVE_BYTES >= WRITER_STACK_BYTES + MAX_WRITE_BATCH_BYTES,
    "writer reserve must cover its stack and maximum write batch"
);
const NATIVE_OVERHEAD_BYTES: usize = 64 * 1024 * 1024;
const DEAD_STATE_BLOOM_BYTES: usize = 1024 * 1024 * 1024;
const DEAD_STATE_BLOOM_HASHES: usize = 4;
const MAX_OPEN_FILES: i32 = 1_024;
const MAX_BACKGROUND_JOBS: i32 = 4;
const MAX_SUBCOMPACTIONS: u32 = 1;
const PARK_DURATION: Duration = Duration::from_millis(2);
const LOG_INTERVAL: Duration = Duration::from_secs(3);
const STORAGE_LOG_INTERVAL: Duration = Duration::from_secs(30);

const MAX_CARRIER_HEIGHT: u32 = 4;
const MAX_CARRIER_HOLES: u32 = 2;

fn carrier_description() -> String {
    format!("height<={MAX_CARRIER_HEIGHT};holes<={MAX_CARRIER_HOLES}")
}

const COMMAND_EXPAND: u8 = 0;
const COMMAND_ATTACH_EDGE: u8 = 1;
const COMMAND_ATTACH_RESULT: u8 = 2;
const COMMAND_CHILD_DEAD: u8 = 3;
const COMMAND_DETACH_EDGE: u8 = 4;
const COMMAND_DRAIN_DEAD: u8 = 5;

#[derive(Parser)]
#[command(name = "tetris-atlas-partitioned")]
#[command(about = "Partition-owned RocksDB closed-carrier atlas builder")]
struct Cli {
    /// Path to the RocksDB directory.
    #[arg(short, long)]
    db_path: PathBuf,

    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Create a new atlas, or resume a cleanly paused build.
    Create {
        /// Number of partition-owner writer threads.
        #[arg(long)]
        writers: Option<usize>,
        /// Shared RocksDB/application memory budget.
        #[arg(long, default_value_t = DEFAULT_RAM_MIB)]
        ram_mib: usize,
    },
    /// Inspect or replay a completed atlas.
    Explore {
        /// Shared RocksDB/application memory budget.
        #[arg(long, default_value_t = DEFAULT_RAM_MIB)]
        ram_mib: usize,
        /// Piece sequence to replay, for example "OISTZLJ".
        #[arg(long, default_value = "")]
        pieces: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StateKey([u8; STATE_KEY_LEN]);

impl StateKey {
    fn from_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> Result<Self> {
        let bag_byte = u8::from(bag);
        validate_bag_byte(bag_byte)?;

        let mut out = [0u8; STATE_KEY_LEN];
        for (column, limb) in board.as_limbs().into_iter().enumerate() {
            ensure!(
                limb >> StandardTetris::ROWS == 0,
                "board column {column} has bits above row {}",
                StandardTetris::ROWS
            );
            let offset = column * PACKED_COLUMN_BYTES;
            let bytes = limb.to_le_bytes();
            out[offset..offset + PACKED_COLUMN_BYTES]
                .copy_from_slice(&bytes[..PACKED_COLUMN_BYTES]);
        }
        out[STATE_KEY_LEN - 1] = bag_byte;
        Ok(Self(out))
    }

    fn from_slice(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == STATE_KEY_LEN,
            "state key length is {}, expected {STATE_KEY_LEN}",
            bytes.len()
        );
        let mut out = [0u8; STATE_KEY_LEN];
        out.copy_from_slice(bytes);
        let key = Self(out);
        key.decode()?;
        Ok(key)
    }

    fn decode(self) -> Result<(TetrisBoard, TetrisPieceBagState)> {
        let mut limbs = [0u32; StandardTetris::COLS];
        for (column, limb) in limbs.iter_mut().enumerate() {
            let offset = column * PACKED_COLUMN_BYTES;
            *limb = u32::from_le_bytes([self.0[offset], self.0[offset + 1], self.0[offset + 2], 0]);
            ensure!(
                *limb >> StandardTetris::ROWS == 0,
                "state key column {column} has bits above row {}",
                StandardTetris::ROWS
            );
        }
        let bag_byte = self.0[STATE_KEY_LEN - 1];
        validate_bag_byte(bag_byte)?;
        Ok((
            TetrisBoard::from_limbs(limbs),
            TetrisPieceBagState::from(bag_byte),
        ))
    }

    fn bag(self) -> Result<TetrisPieceBagState> {
        let bag = self.0[STATE_KEY_LEN - 1];
        validate_bag_byte(bag)?;
        Ok(TetrisPieceBagState::from(bag))
    }

    const fn as_bytes(&self) -> &[u8; STATE_KEY_LEN] {
        &self.0
    }
}

fn validate_bag_byte(bag: u8) -> Result<()> {
    ensure!(bag != 0, "state key bag must not be empty");
    ensure!(
        bag & !TetrisPieceBagState::FULL_MASK == 0,
        "state key bag contains unknown piece bits"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoveSlot {
    Active(u8),
    Pending(u8),
    Unexpanded,
    Exhausted,
}

impl MoveSlot {
    fn decode(piece: TetrisPiece, byte: u8) -> Result<Self> {
        match byte {
            SLOT_UNEXPANDED => Ok(Self::Unexpanded),
            SLOT_EXHAUSTED => Ok(Self::Exhausted),
            0..=0x27 => {
                validate_orientation(piece, byte)?;
                Ok(Self::Active(byte))
            }
            SLOT_PENDING_BASE..=0x67 => {
                let orientation = byte - SLOT_PENDING_BASE;
                validate_orientation(piece, orientation)?;
                Ok(Self::Pending(orientation))
            }
            _ => bail!("invalid node slot byte {byte:#04x}"),
        }
    }

    const fn encode(self) -> u8 {
        match self {
            Self::Active(orientation) => orientation,
            Self::Pending(orientation) => SLOT_PENDING_BASE + orientation,
            Self::Unexpanded => SLOT_UNEXPANDED,
            Self::Exhausted => SLOT_EXHAUSTED,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NodeRecord([u8; NODE_RECORD_LEN]);

impl NodeRecord {
    const INIT: Self = Self([SLOT_UNEXPANDED; NODE_RECORD_LEN]);

    fn from_slice(bytes: &[u8], bag: TetrisPieceBagState) -> Result<Self> {
        ensure!(
            bytes.len() == NODE_RECORD_LEN,
            "node record length is {}, expected {NODE_RECORD_LEN}",
            bytes.len()
        );
        let mut record = [0u8; NODE_RECORD_LEN];
        record.copy_from_slice(bytes);
        let value = Self(record);
        value.validate(bag)?;
        Ok(value)
    }

    fn validate(self, bag: TetrisPieceBagState) -> Result<()> {
        for piece in TetrisPiece::all() {
            let slot = MoveSlot::decode(piece, self.0[piece.index() as usize])?;
            ensure!(
                bag.contains(piece) || slot == MoveSlot::Unexpanded,
                "non-forced piece {piece:?} has a persisted move"
            );
        }
        Ok(())
    }

    fn slot(self, piece: TetrisPiece) -> Result<MoveSlot> {
        MoveSlot::decode(piece, self.0[piece.index() as usize])
    }

    fn set_slot(&mut self, piece: TetrisPiece, slot: MoveSlot) -> Result<()> {
        match slot {
            MoveSlot::Active(orientation) | MoveSlot::Pending(orientation) => {
                validate_orientation(piece, orientation)?;
            }
            MoveSlot::Unexpanded | MoveSlot::Exhausted => {}
        }
        self.0[piece.index() as usize] = slot.encode();
        Ok(())
    }

    fn is_dead(self, bag: TetrisPieceBagState) -> Result<bool> {
        for piece in bag.iter_pieces() {
            if self.slot(piece)? == MoveSlot::Exhausted {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn is_resolved(self, bag: TetrisPieceBagState) -> Result<bool> {
        for piece in bag.iter_pieces() {
            if !matches!(self.slot(piece)?, MoveSlot::Active(_)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn has_pending(self, bag: TetrisPieceBagState) -> Result<bool> {
        for piece in bag.iter_pieces() {
            if matches!(self.slot(piece)?, MoveSlot::Pending(_)) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    const fn as_bytes(&self) -> &[u8; NODE_RECORD_LEN] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ClosureGauges {
    live_states: i64,
    states_with_unexpanded: i64,
    states_with_pending: i64,
    locally_resolved_states: i64,
    forced_slots_total: i64,
    forced_slots_unexpanded: i64,
    forced_slots_pending: i64,
    forced_slots_active: i64,
}

impl ClosureGauges {
    fn from_record(record: Option<NodeRecord>, bag: TetrisPieceBagState) -> Result<Self> {
        let Some(record) = record else {
            return Ok(Self::default());
        };
        if record.is_dead(bag)? {
            return Ok(Self::default());
        }

        let mut gauges = Self {
            live_states: 1,
            ..Self::default()
        };
        for piece in bag.iter_pieces() {
            gauges.forced_slots_total += 1;
            match record.slot(piece)? {
                MoveSlot::Active(_) => gauges.forced_slots_active += 1,
                MoveSlot::Pending(_) => gauges.forced_slots_pending += 1,
                MoveSlot::Unexpanded => gauges.forced_slots_unexpanded += 1,
                MoveSlot::Exhausted => {
                    bail!("nondead node contains an exhausted forced slot");
                }
            }
        }
        gauges.states_with_unexpanded = i64::from(gauges.forced_slots_unexpanded != 0);
        gauges.states_with_pending = i64::from(gauges.forced_slots_pending != 0);
        gauges.locally_resolved_states =
            i64::from(gauges.forced_slots_active == gauges.forced_slots_total);
        Ok(gauges)
    }

    fn delta_from(self, before: Self) -> Self {
        Self {
            live_states: self.live_states - before.live_states,
            states_with_unexpanded: self.states_with_unexpanded - before.states_with_unexpanded,
            states_with_pending: self.states_with_pending - before.states_with_pending,
            locally_resolved_states: self.locally_resolved_states - before.locally_resolved_states,
            forced_slots_total: self.forced_slots_total - before.forced_slots_total,
            forced_slots_unexpanded: self.forced_slots_unexpanded - before.forced_slots_unexpanded,
            forced_slots_pending: self.forced_slots_pending - before.forced_slots_pending,
            forced_slots_active: self.forced_slots_active - before.forced_slots_active,
        }
    }

    fn checked_add(&mut self, other: Self) -> Result<()> {
        self.live_states = self
            .live_states
            .checked_add(other.live_states)
            .context("live-state recovery count overflow")?;
        self.states_with_unexpanded = self
            .states_with_unexpanded
            .checked_add(other.states_with_unexpanded)
            .context("unexpanded-state recovery count overflow")?;
        self.states_with_pending = self
            .states_with_pending
            .checked_add(other.states_with_pending)
            .context("pending-state recovery count overflow")?;
        self.locally_resolved_states = self
            .locally_resolved_states
            .checked_add(other.locally_resolved_states)
            .context("resolved-state recovery count overflow")?;
        self.forced_slots_total = self
            .forced_slots_total
            .checked_add(other.forced_slots_total)
            .context("forced-slot recovery count overflow")?;
        self.forced_slots_unexpanded = self
            .forced_slots_unexpanded
            .checked_add(other.forced_slots_unexpanded)
            .context("unexpanded-slot recovery count overflow")?;
        self.forced_slots_pending = self
            .forced_slots_pending
            .checked_add(other.forced_slots_pending)
            .context("pending-slot recovery count overflow")?;
        self.forced_slots_active = self
            .forced_slots_active
            .checked_add(other.forced_slots_active)
            .context("active-slot recovery count overflow")?;
        Ok(())
    }
}

impl ClosureGauges {
    fn add(&mut self, other: Self) {
        self.live_states += other.live_states;
        self.states_with_unexpanded += other.states_with_unexpanded;
        self.states_with_pending += other.states_with_pending;
        self.locally_resolved_states += other.locally_resolved_states;
        self.forced_slots_total += other.forced_slots_total;
        self.forced_slots_unexpanded += other.forced_slots_unexpanded;
        self.forced_slots_pending += other.forced_slots_pending;
        self.forced_slots_active += other.forced_slots_active;
    }
}

fn validate_piece_index(index: u8) -> Result<TetrisPiece> {
    ensure!(
        (index as usize) < TetrisPiece::NUM_PIECES,
        "invalid piece index {index}"
    );
    Ok(TetrisPiece::from_index(index))
}

fn producer_sequence_meta_key(producer: u8) -> [u8; 14] {
    let mut key = [0u8; 14];
    key[..META_PRODUCER_SEQUENCE_PREFIX.len()].copy_from_slice(META_PRODUCER_SEQUENCE_PREFIX);
    key[META_PRODUCER_SEQUENCE_PREFIX.len()] = producer;
    key
}

fn work_head_meta_key(owner: u8, priority: u8, producer: u8) -> [u8; 13] {
    let mut key = [0u8; 13];
    key[..META_WORK_HEAD_PREFIX.len()].copy_from_slice(META_WORK_HEAD_PREFIX);
    key[META_WORK_HEAD_PREFIX.len()..].copy_from_slice(&[owner, priority, producer]);
    key
}

fn decode_u64_metadata(bytes: &[u8], name: &str) -> Result<u64> {
    let encoded: [u8; 8] = bytes
        .try_into()
        .with_context(|| format!("{name} metadata must contain eight bytes"))?;
    Ok(u64::from_be_bytes(encoded))
}

fn validate_orientation(piece: TetrisPiece, index: u8) -> Result<TetrisPieceOrientation> {
    ensure!(
        (index as usize) < TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
        "invalid orientation index {index}"
    );
    let mask = TetrisPieceOrientation::ORIENTATION_MASKS_BY_PIECE[piece.index() as usize];
    ensure!(
        mask & (1u64 << index) != 0,
        "orientation {index} does not belong to piece {piece:?}"
    );
    Ok(TetrisPieceOrientation::from_index(index))
}

fn cf<'db>(db: &'db DB, name: &str) -> Result<Arc<BoundColumnFamily<'db>>> {
    db.cf_handle(name)
        .with_context(|| format!("missing {name} column family"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ReverseEdgeKey([u8; REDGE_KEY_LEN]);

impl ReverseEdgeKey {
    fn new(
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    ) -> Result<Self> {
        validate_orientation(piece, orientation.index())?;
        let mut out = [0u8; REDGE_KEY_LEN];
        out[..STATE_KEY_LEN].copy_from_slice(succ.as_bytes());
        out[STATE_KEY_LEN..STATE_KEY_LEN * 2].copy_from_slice(pred.as_bytes());
        out[STATE_KEY_LEN * 2] = piece.index();
        out[STATE_KEY_LEN * 2 + 1] = orientation.index();
        Ok(Self(out))
    }

    fn from_slice(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == REDGE_KEY_LEN,
            "reverse-edge key length is {}, expected {REDGE_KEY_LEN}",
            bytes.len()
        );
        let mut out = [0u8; REDGE_KEY_LEN];
        out.copy_from_slice(bytes);
        let key = Self(out);
        key.decode()?;
        Ok(key)
    }

    fn decode(self) -> Result<(StateKey, StateKey, TetrisPiece, TetrisPieceOrientation)> {
        let succ = StateKey::from_slice(&self.0[..STATE_KEY_LEN])?;
        let pred = StateKey::from_slice(&self.0[STATE_KEY_LEN..STATE_KEY_LEN * 2])?;
        let piece = validate_piece_index(self.0[STATE_KEY_LEN * 2])?;
        let orientation = validate_orientation(piece, self.0[STATE_KEY_LEN * 2 + 1])?;
        Ok((succ, pred, piece, orientation))
    }

    const fn as_bytes(&self) -> &[u8; REDGE_KEY_LEN] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct WorkKey([u8; WORK_KEY_LEN]);

impl WorkKey {
    fn new(owner: u8, priority: u8, producer: u8, sequence: u64) -> Result<Self> {
        ensure!(
            priority <= PRIORITY_EXPAND,
            "invalid work priority {priority}"
        );
        ensure!(sequence != u64::MAX, "work sequence u64::MAX is reserved");
        let mut out = [0u8; WORK_KEY_LEN];
        out[0] = owner;
        out[1] = priority;
        out[2] = producer;
        out[3..].copy_from_slice(&sequence.to_be_bytes());
        Ok(Self(out))
    }

    fn from_slice(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == WORK_KEY_LEN,
            "work key length is {}, expected {WORK_KEY_LEN}",
            bytes.len()
        );
        let sequence = u64::from_be_bytes(bytes[3..].try_into().context("work sequence")?);
        Self::new(bytes[0], bytes[1], bytes[2], sequence)
    }

    const fn owner(self) -> u8 {
        self.0[0]
    }

    const fn priority(self) -> u8 {
        self.0[1]
    }

    const fn producer(self) -> u8 {
        self.0[2]
    }

    fn sequence(self) -> u64 {
        let mut sequence = [0u8; 8];
        sequence.copy_from_slice(&self.0[3..]);
        u64::from_be_bytes(sequence)
    }

    const fn as_bytes(&self) -> &[u8; WORK_KEY_LEN] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkClaimCursor {
    next_sequence: [[u64; VIRTUAL_SHARDS]; WORK_PRIORITY_COUNT],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WorkCommitStats {
    heads_advanced: u64,
    reclaim_ranges: u64,
}

impl WorkClaimCursor {
    const fn new() -> Self {
        Self::filled(0)
    }

    const fn filled(next_sequence: u64) -> Self {
        Self {
            next_sequence: [[next_sequence; VIRTUAL_SHARDS]; WORK_PRIORITY_COUNT],
        }
    }

    const fn next(&self, priority: u8, producer: u8) -> u64 {
        self.next_sequence[priority as usize][producer as usize]
    }

    fn set(&mut self, priority: u8, producer: u8, next_sequence: u64) {
        self.next_sequence[priority as usize][producer as usize] = next_sequence;
    }

    fn advance_committed(&mut self, processed: &[WorkKey]) -> Result<()> {
        for key in processed {
            let next = key
                .sequence()
                .checked_add(1)
                .context("work claim cursor overflow")?;
            let cursor = &mut self.next_sequence[key.priority() as usize][key.producer() as usize];
            *cursor = (*cursor).max(next);
        }
        Ok(())
    }

    fn append_committed<C: AsColumnFamilyRef>(
        &self,
        batch: &mut WriteBatch,
        work_cf: &C,
        processed: &[WorkKey],
    ) -> Result<WorkCommitStats> {
        let mut stats = WorkCommitStats::default();
        let mut consumed_lane: Option<((u8, u8, u8), WorkKey, WorkKey)> = None;
        for &key in processed {
            let lane = (key.owner(), key.priority(), key.producer());
            if let Some((current_lane, first, last)) = &mut consumed_lane {
                ensure!(*last < key, "processed work keys are not strictly ordered");
                if *current_lane == lane {
                    *last = key;
                    continue;
                }
                self.append_consumed_lane(batch, work_cf, *first, *last, &mut stats)?;
            }
            consumed_lane = Some((lane, key, key));
        }
        if let Some((_, first, last)) = consumed_lane {
            self.append_consumed_lane(batch, work_cf, first, last, &mut stats)?;
        }
        Ok(stats)
    }

    fn append_consumed_lane<C: AsColumnFamilyRef>(
        &self,
        batch: &mut WriteBatch,
        work_cf: &C,
        first: WorkKey,
        last: WorkKey,
        stats: &mut WorkCommitStats,
    ) -> Result<()> {
        ensure!(
            (first.owner(), first.priority(), first.producer())
                == (last.owner(), last.priority(), last.producer()),
            "consumed work range crosses lanes"
        );
        let old_head = self.next(first.priority(), first.producer());
        ensure!(
            first.sequence() >= old_head,
            "processed work precedes the durable lane head"
        );
        let new_head = last
            .sequence()
            .checked_add(1)
            .context("work head overflow")?;
        batch.put(
            work_head_meta_key(first.owner(), first.priority(), first.producer()),
            new_head.to_be_bytes(),
        );
        stats.heads_advanced += 1;

        let old_reclaim = old_head / WORK_RECLAIM_SEQUENCE_QUANTUM * WORK_RECLAIM_SEQUENCE_QUANTUM;
        let new_reclaim = new_head / WORK_RECLAIM_SEQUENCE_QUANTUM * WORK_RECLAIM_SEQUENCE_QUANTUM;
        if new_reclaim > old_reclaim {
            let start = WorkKey::new(
                first.owner(),
                first.priority(),
                first.producer(),
                old_reclaim,
            )?;
            let end = WorkKey::new(
                first.owner(),
                first.priority(),
                first.producer(),
                new_reclaim,
            )?;
            batch.delete_range_cf(work_cf, start.as_bytes(), end.as_bytes());
            stats.reclaim_ranges += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Expand {
        state: StateKey,
    },
    AttachEdge {
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    AttachResult {
        accepted: bool,
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    ChildDead {
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    DetachEdge {
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    DrainDead {
        succ: StateKey,
    },
}

struct EncodedCommand {
    bytes: [u8; 66],
    len: usize,
}

impl EncodedCommand {
    const fn len(&self) -> usize {
        self.len
    }
}

impl AsRef<[u8]> for EncodedCommand {
    fn as_ref(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl Command {
    fn encode(self) -> Result<EncodedCommand> {
        let mut out = EncodedCommand {
            bytes: [0u8; 66],
            len: 0,
        };
        let mut append = |bytes: &[u8]| {
            let end = out.len + bytes.len();
            out.bytes[out.len..end].copy_from_slice(bytes);
            out.len = end;
        };
        append(&[self.kind_index() as u8]);
        match self {
            Self::Expand { state } | Self::DrainDead { succ: state } => {
                append(state.as_bytes());
            }
            Self::AttachEdge {
                succ,
                pred,
                piece,
                orientation,
            }
            | Self::ChildDead {
                succ,
                pred,
                piece,
                orientation,
            }
            | Self::DetachEdge {
                succ,
                pred,
                piece,
                orientation,
            } => {
                validate_orientation(piece, orientation.index())?;
                append(succ.as_bytes());
                append(pred.as_bytes());
                append(&[piece.index(), orientation.index()]);
            }
            Self::AttachResult {
                accepted,
                succ,
                pred,
                piece,
                orientation,
            } => {
                validate_orientation(piece, orientation.index())?;
                append(&[u8::from(accepted)]);
                append(succ.as_bytes());
                append(pred.as_bytes());
                append(&[piece.index(), orientation.index()]);
            }
        }
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let (&tag, payload) = bytes
            .split_first()
            .context("command payload must contain a tag")?;
        match tag {
            COMMAND_EXPAND => {
                ensure!(payload.len() == STATE_KEY_LEN, "invalid Expand length");
                Ok(Self::Expand {
                    state: StateKey::from_slice(payload)?,
                })
            }
            COMMAND_ATTACH_EDGE | COMMAND_CHILD_DEAD | COMMAND_DETACH_EDGE => {
                ensure!(
                    payload.len() == STATE_KEY_LEN * 2 + 2,
                    "invalid edge command length"
                );
                let succ = StateKey::from_slice(&payload[..STATE_KEY_LEN])?;
                let pred = StateKey::from_slice(&payload[STATE_KEY_LEN..STATE_KEY_LEN * 2])?;
                let piece = validate_piece_index(payload[STATE_KEY_LEN * 2])?;
                let orientation = validate_orientation(piece, payload[STATE_KEY_LEN * 2 + 1])?;
                Ok(if tag == COMMAND_ATTACH_EDGE {
                    Self::AttachEdge {
                        succ,
                        pred,
                        piece,
                        orientation,
                    }
                } else if tag == COMMAND_CHILD_DEAD {
                    Self::ChildDead {
                        succ,
                        pred,
                        piece,
                        orientation,
                    }
                } else {
                    Self::DetachEdge {
                        succ,
                        pred,
                        piece,
                        orientation,
                    }
                })
            }
            COMMAND_ATTACH_RESULT => {
                ensure!(
                    payload.len() == STATE_KEY_LEN * 2 + 3,
                    "invalid AttachResult length"
                );
                ensure!(payload[0] <= 1, "invalid AttachResult accepted byte");
                let succ = StateKey::from_slice(&payload[1..1 + STATE_KEY_LEN])?;
                let pred =
                    StateKey::from_slice(&payload[1 + STATE_KEY_LEN..1 + STATE_KEY_LEN * 2])?;
                let piece = validate_piece_index(payload[1 + STATE_KEY_LEN * 2])?;
                let orientation = validate_orientation(piece, payload[2 + STATE_KEY_LEN * 2])?;
                Ok(Self::AttachResult {
                    accepted: payload[0] == 1,
                    succ,
                    pred,
                    piece,
                    orientation,
                })
            }
            COMMAND_DRAIN_DEAD => {
                ensure!(payload.len() == STATE_KEY_LEN, "invalid DrainDead length");
                Ok(Self::DrainDead {
                    succ: StateKey::from_slice(payload)?,
                })
            }
            _ => bail!("invalid command tag {tag}"),
        }
    }

    const fn priority(self) -> u8 {
        match self {
            Self::DrainDead { .. }
            | Self::ChildDead { .. }
            | Self::AttachResult {
                accepted: false, ..
            } => PRIORITY_DEATH,
            Self::AttachEdge { .. }
            | Self::DetachEdge { .. }
            | Self::AttachResult { accepted: true, .. } => PRIORITY_EDGE,
            Self::Expand { .. } => PRIORITY_EXPAND,
        }
    }

    const fn target(self) -> StateKey {
        match self {
            Self::Expand { state } => state,
            Self::AttachEdge { succ, .. }
            | Self::DetachEdge { succ, .. }
            | Self::DrainDead { succ } => succ,
            Self::AttachResult { pred, .. } | Self::ChildDead { pred, .. } => pred,
        }
    }

    const fn node_read_key(&self) -> Option<&StateKey> {
        match self {
            Self::Expand { state } => Some(state),
            Self::AttachEdge { succ, .. } | Self::DrainDead { succ } => Some(succ),
            Self::AttachResult { pred, .. } | Self::ChildDead { pred, .. } => Some(pred),
            Self::DetachEdge { .. } => None,
        }
    }

    const fn source(self) -> StateKey {
        match self {
            Self::Expand { state } => state,
            Self::AttachEdge { pred, .. } | Self::DetachEdge { pred, .. } => pred,
            Self::AttachResult { succ, .. }
            | Self::ChildDead { succ, .. }
            | Self::DrainDead { succ } => succ,
        }
    }

    const fn kind_index(self) -> usize {
        match self {
            Self::Expand { .. } => COMMAND_EXPAND as usize,
            Self::AttachEdge { .. } => COMMAND_ATTACH_EDGE as usize,
            Self::AttachResult { .. } => COMMAND_ATTACH_RESULT as usize,
            Self::ChildDead { .. } => COMMAND_CHILD_DEAD as usize,
            Self::DetachEdge { .. } => COMMAND_DETACH_EDGE as usize,
            Self::DrainDead { .. } => COMMAND_DRAIN_DEAD as usize,
        }
    }
}

const fn command_scratch_requirements(command: Command) -> (usize, usize) {
    match command {
        Command::Expand { .. } | Command::AttachResult { .. } | Command::ChildDead { .. } => (0, 8),
        Command::AttachEdge { .. } => (1, 2),
        Command::DetachEdge { .. } => (1, 0),
        Command::DrainDead { .. } => (MAX_REDGES_PER_DRAIN, MAX_REDGES_PER_DRAIN + 1),
    }
}

#[derive(Clone)]
struct OwnerMap {
    owners: [u8; VIRTUAL_SHARDS],
    writer_count: usize,
}

impl OwnerMap {
    fn balanced(writer_count: usize) -> Result<Self> {
        ensure!(
            writer_count > 0 && writer_count <= VIRTUAL_SHARDS,
            "writer count must be in 1..={VIRTUAL_SHARDS}"
        );
        let mut owners = [0u8; VIRTUAL_SHARDS];
        for (shard, owner) in owners.iter_mut().enumerate() {
            *owner = (shard % writer_count) as u8;
        }
        Ok(Self {
            owners,
            writer_count,
        })
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == VIRTUAL_SHARDS + 2,
            "owner map length is {}, expected {}",
            bytes.len(),
            VIRTUAL_SHARDS + 2
        );
        let writer_count = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
        let expected = Self::balanced(writer_count)?;
        ensure!(
            bytes[2..] == expected.owners,
            "owner map does not match the schema's balanced assignment"
        );
        Ok(expected)
    }

    fn as_bytes(&self) -> [u8; VIRTUAL_SHARDS + 2] {
        let mut out = [0u8; VIRTUAL_SHARDS + 2];
        out[..2].copy_from_slice(&(self.writer_count as u16).to_be_bytes());
        out[2..].copy_from_slice(&self.owners);
        out
    }

    fn owner(&self, state: StateKey) -> u8 {
        self.owners[state_shard(state) as usize]
    }

    fn validate_work(&self, key: WorkKey, command: Command) -> Result<()> {
        ensure!(
            (key.owner() as usize) < self.writer_count,
            "work owner {} is outside writer count {}",
            key.owner(),
            self.writer_count
        );
        ensure!(
            (key.producer() as usize) < self.writer_count,
            "work producer {} is outside writer count {}",
            key.producer(),
            self.writer_count
        );
        ensure!(
            key.priority() == command.priority(),
            "work priority does not match command"
        );
        ensure!(
            key.owner() == self.owner(command.target()),
            "work owner does not own command target"
        );
        ensure!(
            key.producer() == self.owner(command.source()),
            "work producer does not own command source"
        );
        Ok(())
    }
}

fn state_hash(state: StateKey) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in state.0 {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    hash
}

fn state_shard(state: StateKey) -> u8 {
    state_hash(state) as u8
}

fn mix_bloom_hash(mut hash: u64) -> u64 {
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^ (hash >> 31)
}

struct DeadStateBloom {
    words: Box<[AtomicU64]>,
    word_mask: usize,
    set_bits: AtomicU64,
}

impl DeadStateBloom {
    fn new(byte_budget: usize) -> Result<Self> {
        let word_bytes = std::mem::size_of::<AtomicU64>();
        let available_words = byte_budget / word_bytes;
        ensure!(
            available_words != 0,
            "dead-state Bloom budget must hold at least one word"
        );
        let word_count = 1usize << available_words.ilog2();
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .context("allocate dead-state Bloom filter")?;
        words.resize_with(word_count, || AtomicU64::new(0));
        Ok(Self {
            words: words.into_boxed_slice(),
            word_mask: word_count - 1,
            set_bits: AtomicU64::new(0),
        })
    }

    fn probe(&self, state: StateKey) -> (usize, u64) {
        let hash = state_hash(state);
        let word = hash as usize & self.word_mask;
        let positions = mix_bloom_hash(hash ^ 0x9e3779b97f4a7c15);
        let mut mask = 0;
        for index in 0..DEAD_STATE_BLOOM_HASHES {
            let bit = (positions >> (index * 6)) & 0x3f;
            mask |= 1 << bit;
        }
        (word, mask)
    }

    fn might_contain(&self, state: StateKey) -> bool {
        let (word, mask) = self.probe(state);
        self.words[word].load(AtomicOrdering::Relaxed) & mask == mask
    }

    fn insert(&self, state: StateKey) -> u64 {
        let (word, mask) = self.probe(state);
        let previous = self.words[word].fetch_or(mask, AtomicOrdering::Relaxed);
        let newly_set = u64::from((mask & !previous).count_ones());
        atomic_add_relaxed(&self.set_bits, newly_set);
        newly_set
    }

    fn allocated_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<AtomicU64>()
    }

    fn capacity_bits(&self) -> u64 {
        (self.allocated_bytes() as u64) * 8
    }

    fn set_bits(&self) -> u64 {
        self.set_bits.load(AtomicOrdering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
struct CommandCounts {
    expand: u64,
    attach_edge: u64,
    attach_result: u64,
    child_dead: u64,
    detach_edge: u64,
    drain_dead: u64,
}

impl CommandCounts {
    const fn from_array(values: [u64; COMMAND_KIND_COUNT]) -> Self {
        Self {
            expand: values[COMMAND_EXPAND as usize],
            attach_edge: values[COMMAND_ATTACH_EDGE as usize],
            attach_result: values[COMMAND_ATTACH_RESULT as usize],
            child_dead: values[COMMAND_CHILD_DEAD as usize],
            detach_edge: values[COMMAND_DETACH_EDGE as usize],
            drain_dead: values[COMMAND_DRAIN_DEAD as usize],
        }
    }

    const fn to_array(self) -> [u64; COMMAND_KIND_COUNT] {
        let mut values = [0; COMMAND_KIND_COUNT];
        values[COMMAND_EXPAND as usize] = self.expand;
        values[COMMAND_ATTACH_EDGE as usize] = self.attach_edge;
        values[COMMAND_ATTACH_RESULT as usize] = self.attach_result;
        values[COMMAND_CHILD_DEAD as usize] = self.child_dead;
        values[COMMAND_DETACH_EDGE as usize] = self.detach_edge;
        values[COMMAND_DRAIN_DEAD as usize] = self.drain_dead;
        values
    }

    const fn total(self) -> u64 {
        self.expand
            + self.attach_edge
            + self.attach_result
            + self.child_dead
            + self.detach_edge
            + self.drain_dead
    }
}

/// Declares an atomic counter block and its plain serializable twin with
/// field-wise `add` and `snapshot`.
macro_rules! stat_pair {
    (struct $atomic:ident / $snapshot:ident { $($field:ident),* $(,)? }) => {
        #[derive(Default)]
        struct $atomic {
            $($field: AtomicU64,)*
        }

        #[derive(Clone, Copy, Debug, Default, Serialize)]
        struct $snapshot {
            $($field: u64,)*
        }

        impl $atomic {
            fn add(&self, delta: &$snapshot) {
                $(atomic_add_relaxed(&self.$field, delta.$field);)*
            }

            fn snapshot(&self) -> $snapshot {
                $snapshot {
                    $($field: self.$field.load(AtomicOrdering::Relaxed),)*
                }
            }
        }
    };
}

/// Declares an atomic gauge block and its plain serializable twin with
/// field-wise `store` and `snapshot`; `max` fields keep their high-water mark.
macro_rules! gauge_pair {
    (
        struct $atomic:ident / $snapshot:ident {
            max: { $($max_field:ident),* $(,)? },
            latest: { $($latest_field:ident),* $(,)? } $(,)?
        }
    ) => {
        #[derive(Default)]
        struct $atomic {
            $($max_field: AtomicU64,)*
            $($latest_field: AtomicU64,)*
        }

        #[derive(Clone, Copy, Debug, Default, Serialize)]
        struct $snapshot {
            $($max_field: u64,)*
            $($latest_field: u64,)*
        }

        impl $atomic {
            fn store(&self, value: &$snapshot) {
                $(self.$max_field.fetch_max(value.$max_field, AtomicOrdering::Relaxed);)*
                $(self.$latest_field.store(value.$latest_field, AtomicOrdering::Relaxed);)*
            }

            fn snapshot(&self) -> $snapshot {
                $snapshot {
                    $($max_field: self.$max_field.load(AtomicOrdering::Relaxed),)*
                    $($latest_field: self.$latest_field.load(AtomicOrdering::Relaxed),)*
                }
            }
        }
    };
}

/// Builds a snapshot struct by relaxed-loading the named fields from an atomic
/// source with matching field names.
macro_rules! snapshot_fields {
    ($source:expr, $snapshot:ident { $($field:ident),* $(,)? }) => {
        $snapshot {
            $($field: ($source).$field.load(AtomicOrdering::Relaxed),)*
        }
    };
}

#[derive(Default)]
struct AtomicCommandCounts {
    values: [AtomicU64; COMMAND_KIND_COUNT],
}

impl AtomicCommandCounts {
    fn new(initial: [u64; COMMAND_KIND_COUNT]) -> Self {
        Self {
            values: std::array::from_fn(|index| AtomicU64::new(initial[index])),
        }
    }

    fn add(&self, values: &[u64; COMMAND_KIND_COUNT]) {
        for (counter, value) in self.values.iter().zip(values) {
            atomic_add_relaxed(counter, *value);
        }
    }

    fn snapshot(&self) -> CommandCounts {
        CommandCounts::from_array(std::array::from_fn(|index| {
            self.values[index].load(AtomicOrdering::Relaxed)
        }))
    }
}

#[derive(Default)]
struct AtomicCommandStats {
    carried_in: AtomicCommandCounts,
    produced: AtomicCommandCounts,
    processed: AtomicCommandCounts,
    stale: AtomicCommandCounts,
}

#[derive(Clone, Debug, Serialize)]
struct CommandStatsSnapshot {
    carried_in: CommandCounts,
    produced: CommandCounts,
    processed: CommandCounts,
    stale: CommandCounts,
}

impl CommandStatsSnapshot {
    fn backlog(&self) -> CommandCounts {
        let carried_in = self.carried_in.to_array();
        let produced = self.produced.to_array();
        let processed = self.processed.to_array();
        CommandCounts::from_array(std::array::from_fn(|kind| {
            carried_in[kind]
                .saturating_add(produced[kind])
                .saturating_sub(processed[kind])
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchStopReason {
    WriteBytes,
    Candidates,
    ReverseEdges,
    Outputs,
}

#[derive(Default)]
struct BatchMetrics {
    commands_produced: [u64; COMMAND_KIND_COUNT],
    commands_processed: [u64; COMMAND_KIND_COUNT],
    commands_stale: [u64; COMMAND_KIND_COUNT],
    nodes_created: u64,
    nodes_dead: u64,
    attach_accepted: u64,
    attach_rejected: u64,
    expansion: ExpansionStatsSnapshot,
    candidates: CandidateStatsSnapshot,
    node_reads: NodeReadStatsSnapshot,
    dead_filter: DeadFilterCounters,
    reverse_edges: ReverseEdgeStatsSnapshot,
    closure_delta: ClosureGauges,
    reroutes: u64,
    reroutes_from_active: u64,
    reroutes_from_pending: u64,
    work_heads_advanced: u64,
    work_reclaim_ranges: u64,
    stop_reason: Option<BatchStopReason>,
}

stat_pair! {
    struct AtomicNodeReadStats / NodeReadStatsSnapshot {
        lookups,
        overlay_hits,
        db_gets,
        db_found,
        db_missing,
        prefetch_calls,
        prefetch_keys,
        candidate_lookups,
        candidate_overlay_hits,
        candidate_db_gets,
    }
}

stat_pair! {
    struct AtomicDeadFilterStats / DeadFilterCounters {
        checks,
        definite_negatives,
        maybe_positives,
        verified_dead,
        false_positives,
        inserts,
        bits_set,
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct DeadFilterStatsSnapshot {
    bytes: u64,
    hashes: usize,
    capacity_bits: u64,
    set_bits: u64,
    occupancy: Option<f64>,
    checks: u64,
    definite_negatives: u64,
    maybe_positives: u64,
    verified_dead: u64,
    false_positives: u64,
    inserts: u64,
}

stat_pair! {
    struct AtomicExpansionStats / ExpansionStatsSnapshot {
        doomed,
        planned_attachments_published,
        planned_attachments_avoided,
    }
}

stat_pair! {
    struct AtomicCandidateStats / CandidateStatsSnapshot {
        prepared_choice_requests,
        prepared_choice_unique,
        prepared_placements,
        prepared_choices_unused,
        unplanned_fallbacks,
        examined,
        rejected_lost,
        rejected_height,
        rejected_holes,
        rejected_known_dead,
        viable,
        reuse_probes,
        existing_overlay,
        existing_database,
        may_exist,
        definitely_new,
        chosen_likely_existing,
        chosen_definitely_new,
    }
}

stat_pair! {
    struct AtomicReverseEdgeStats / ReverseEdgeStatsSnapshot {
        inserted,
        detached,
        drained,
        drain_continuations,
    }
}

#[derive(Default)]
struct AtomicBatchStats {
    claim_calls: AtomicU64,
    empty_claims: AtomicU64,
    claimed_commands_total: AtomicU64,
    claimed_commands_max: AtomicU64,
    claim_micros_total: AtomicU64,
    claim_micros_max: AtomicU64,
    process_micros_total: AtomicU64,
    process_micros_max: AtomicU64,
    committed: AtomicU64,
    commands_total: AtomicU64,
    commands_max: AtomicU64,
    bytes_total: AtomicU64,
    bytes_max: AtomicU64,
    commit_micros_total: AtomicU64,
    commit_micros_max: AtomicU64,
    work_heads_advanced: AtomicU64,
    work_reclaim_ranges: AtomicU64,
    stopped_by_write_bytes: AtomicU64,
    stopped_by_candidates: AtomicU64,
    stopped_by_reverse_edges: AtomicU64,
    stopped_by_outputs: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct BatchStatsSnapshot {
    claim_calls: u64,
    empty_claims: u64,
    claimed_commands_total: u64,
    claimed_commands_max: u64,
    claim_micros_total: u64,
    claim_micros_max: u64,
    process_micros_total: u64,
    process_micros_max: u64,
    committed: u64,
    commands_total: u64,
    commands_max: u64,
    bytes_total: u64,
    bytes_max: u64,
    commit_micros_total: u64,
    commit_micros_max: u64,
    work_heads_advanced: u64,
    work_reclaim_ranges: u64,
    stopped_by_write_bytes: u64,
    stopped_by_candidates: u64,
    stopped_by_reverse_edges: u64,
    stopped_by_outputs: u64,
}

#[derive(Default)]
struct AtomicClosureStats {
    live_states: AtomicI64,
    states_with_unexpanded: AtomicI64,
    states_with_pending: AtomicI64,
    locally_resolved_states: AtomicI64,
    forced_slots_total: AtomicI64,
    forced_slots_unexpanded: AtomicI64,
    forced_slots_pending: AtomicI64,
    forced_slots_active: AtomicI64,
    reroutes: AtomicU64,
    reroutes_from_active: AtomicU64,
    reroutes_from_pending: AtomicU64,
}

impl AtomicClosureStats {
    fn new(initial: ClosureGauges) -> Self {
        Self {
            live_states: AtomicI64::new(initial.live_states),
            states_with_unexpanded: AtomicI64::new(initial.states_with_unexpanded),
            states_with_pending: AtomicI64::new(initial.states_with_pending),
            locally_resolved_states: AtomicI64::new(initial.locally_resolved_states),
            forced_slots_total: AtomicI64::new(initial.forced_slots_total),
            forced_slots_unexpanded: AtomicI64::new(initial.forced_slots_unexpanded),
            forced_slots_pending: AtomicI64::new(initial.forced_slots_pending),
            forced_slots_active: AtomicI64::new(initial.forced_slots_active),
            ..Self::default()
        }
    }

    fn add(&self, metrics: &BatchMetrics) {
        atomic_add_i64_relaxed(&self.live_states, metrics.closure_delta.live_states);
        atomic_add_i64_relaxed(
            &self.states_with_unexpanded,
            metrics.closure_delta.states_with_unexpanded,
        );
        atomic_add_i64_relaxed(
            &self.states_with_pending,
            metrics.closure_delta.states_with_pending,
        );
        atomic_add_i64_relaxed(
            &self.locally_resolved_states,
            metrics.closure_delta.locally_resolved_states,
        );
        atomic_add_i64_relaxed(
            &self.forced_slots_total,
            metrics.closure_delta.forced_slots_total,
        );
        atomic_add_i64_relaxed(
            &self.forced_slots_unexpanded,
            metrics.closure_delta.forced_slots_unexpanded,
        );
        atomic_add_i64_relaxed(
            &self.forced_slots_pending,
            metrics.closure_delta.forced_slots_pending,
        );
        atomic_add_i64_relaxed(
            &self.forced_slots_active,
            metrics.closure_delta.forced_slots_active,
        );
        atomic_add_relaxed(&self.reroutes, metrics.reroutes);
        atomic_add_relaxed(&self.reroutes_from_active, metrics.reroutes_from_active);
        atomic_add_relaxed(&self.reroutes_from_pending, metrics.reroutes_from_pending);
    }

    fn snapshot(&self) -> ClosureOwnerSnapshot {
        ClosureOwnerSnapshot {
            live_states: nonnegative_atomic(&self.live_states),
            states_with_unexpanded: nonnegative_atomic(&self.states_with_unexpanded),
            states_with_pending: nonnegative_atomic(&self.states_with_pending),
            locally_resolved_states: nonnegative_atomic(&self.locally_resolved_states),
            forced_slots_total: nonnegative_atomic(&self.forced_slots_total),
            forced_slots_unexpanded: nonnegative_atomic(&self.forced_slots_unexpanded),
            forced_slots_pending: nonnegative_atomic(&self.forced_slots_pending),
            forced_slots_active: nonnegative_atomic(&self.forced_slots_active),
            reroutes: self.reroutes.load(AtomicOrdering::Relaxed),
            reroutes_from_active: self.reroutes_from_active.load(AtomicOrdering::Relaxed),
            reroutes_from_pending: self.reroutes_from_pending.load(AtomicOrdering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ClosureOwnerSnapshot {
    live_states: u64,
    states_with_unexpanded: u64,
    states_with_pending: u64,
    locally_resolved_states: u64,
    forced_slots_total: u64,
    forced_slots_unexpanded: u64,
    forced_slots_pending: u64,
    forced_slots_active: u64,
    reroutes: u64,
    reroutes_from_active: u64,
    reroutes_from_pending: u64,
}

impl ClosureOwnerSnapshot {
    fn add(&mut self, other: Self) {
        self.live_states += other.live_states;
        self.states_with_unexpanded += other.states_with_unexpanded;
        self.states_with_pending += other.states_with_pending;
        self.locally_resolved_states += other.locally_resolved_states;
        self.forced_slots_total += other.forced_slots_total;
        self.forced_slots_unexpanded += other.forced_slots_unexpanded;
        self.forced_slots_pending += other.forced_slots_pending;
        self.forced_slots_active += other.forced_slots_active;
        self.reroutes += other.reroutes;
        self.reroutes_from_active += other.reroutes_from_active;
        self.reroutes_from_pending += other.reroutes_from_pending;
    }
}

#[derive(Default)]
struct AtomicOwnerStats {
    outstanding: AtomicU64,
    commands_processed: AtomicU64,
    commands_produced: AtomicU64,
    batches: AtomicU64,
    candidates_examined: AtomicU64,
    batch_bytes_total: AtomicU64,
    batch_commands_max: AtomicU64,
    batch_bytes_max: AtomicU64,
    commit_micros_total: AtomicU64,
    commit_micros_max: AtomicU64,
    busy_micros: AtomicU64,
    idle_micros: AtomicU64,
    idle_polls: AtomicU64,
    closure: AtomicClosureStats,
}

impl AtomicOwnerStats {
    fn new(outstanding: u64, initial_closure: ClosureGauges) -> Self {
        Self {
            outstanding: AtomicU64::new(outstanding),
            closure: AtomicClosureStats::new(initial_closure),
            ..Self::default()
        }
    }

    fn snapshot(&self, owner: usize) -> OwnerStatsSnapshot {
        OwnerStatsSnapshot {
            owner,
            outstanding: self.outstanding.load(AtomicOrdering::SeqCst),
            commands_processed: self.commands_processed.load(AtomicOrdering::Relaxed),
            commands_produced: self.commands_produced.load(AtomicOrdering::Relaxed),
            batches: self.batches.load(AtomicOrdering::Relaxed),
            candidates_examined: self.candidates_examined.load(AtomicOrdering::Relaxed),
            batch_bytes_total: self.batch_bytes_total.load(AtomicOrdering::Relaxed),
            batch_commands_max: self.batch_commands_max.load(AtomicOrdering::Relaxed),
            batch_bytes_max: self.batch_bytes_max.load(AtomicOrdering::Relaxed),
            commit_micros_total: self.commit_micros_total.load(AtomicOrdering::Relaxed),
            commit_micros_max: self.commit_micros_max.load(AtomicOrdering::Relaxed),
            busy_micros: self.busy_micros.load(AtomicOrdering::Relaxed),
            idle_micros: self.idle_micros.load(AtomicOrdering::Relaxed),
            idle_polls: self.idle_polls.load(AtomicOrdering::Relaxed),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct OwnerStatsSnapshot {
    owner: usize,
    outstanding: u64,
    commands_processed: u64,
    commands_produced: u64,
    batches: u64,
    candidates_examined: u64,
    batch_bytes_total: u64,
    batch_commands_max: u64,
    batch_bytes_max: u64,
    commit_micros_total: u64,
    commit_micros_max: u64,
    busy_micros: u64,
    idle_micros: u64,
    idle_polls: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ClosureStatsSnapshot {
    nodes_not_dead: u64,
    states_with_unexpanded: u64,
    states_with_pending: u64,
    locally_resolved_states: u64,
    unresolved_states: u64,
    discovered_state_resolution_ratio: Option<f64>,
    forced_slots_total: u64,
    forced_slots_unexpanded: u64,
    forced_slots_pending: u64,
    forced_slots_active: u64,
    discovered_slot_resolution_ratio: Option<f64>,
    expand_backlog: u64,
    coordination_backlog: u64,
    run_expansion_reproduction: Option<f64>,
    run_novelty_rate: Option<f64>,
    total_death_fraction: Option<f64>,
    run_death_cascade_factor: Option<f64>,
    reroutes: u64,
    reroutes_from_active: u64,
    reroutes_from_pending: u64,
    proof_deficits: ProofDeficitSnapshot,
}

#[derive(Clone, Debug, Serialize)]
struct ProofDeficitSnapshot {
    pending_work: u64,
    unresolved_states: u64,
    states_with_unexpanded: u64,
    states_with_pending: u64,
    child_dead_backlog: u64,
    drain_dead_backlog: u64,
}

impl ClosureStatsSnapshot {
    fn from_runtime(
        gauges: ClosureOwnerSnapshot,
        commands: &CommandStatsSnapshot,
        nodes_created: u64,
        nodes_dead: u64,
        nodes_created_this_run: u64,
        nodes_dead_this_run: u64,
        outstanding_work: u64,
        reverse_edges_drained: u64,
    ) -> Self {
        let backlog = commands.backlog();
        let unresolved_states = gauges
            .live_states
            .saturating_sub(gauges.locally_resolved_states);
        Self {
            nodes_not_dead: gauges.live_states,
            states_with_unexpanded: gauges.states_with_unexpanded,
            states_with_pending: gauges.states_with_pending,
            locally_resolved_states: gauges.locally_resolved_states,
            unresolved_states,
            discovered_state_resolution_ratio: ratio(
                gauges.locally_resolved_states,
                gauges.live_states,
            ),
            forced_slots_total: gauges.forced_slots_total,
            forced_slots_unexpanded: gauges.forced_slots_unexpanded,
            forced_slots_pending: gauges.forced_slots_pending,
            forced_slots_active: gauges.forced_slots_active,
            discovered_slot_resolution_ratio: ratio(
                gauges.forced_slots_active,
                gauges.forced_slots_total,
            ),
            expand_backlog: backlog.expand,
            coordination_backlog: outstanding_work.saturating_sub(backlog.expand),
            run_expansion_reproduction: ratio(nodes_created_this_run, commands.processed.expand),
            run_novelty_rate: ratio(nodes_created_this_run, commands.processed.attach_edge),
            total_death_fraction: ratio(nodes_dead, nodes_created),
            run_death_cascade_factor: ratio(reverse_edges_drained, nodes_dead_this_run),
            reroutes: gauges.reroutes,
            reroutes_from_active: gauges.reroutes_from_active,
            reroutes_from_pending: gauges.reroutes_from_pending,
            proof_deficits: ProofDeficitSnapshot {
                pending_work: outstanding_work,
                unresolved_states,
                states_with_unexpanded: gauges.states_with_unexpanded,
                states_with_pending: gauges.states_with_pending,
                child_dead_backlog: backlog.child_dead,
                drain_dead_backlog: backlog.drain_dead,
            },
        }
    }
}

gauge_pair! {
    struct AtomicStorageStats / StorageStatsSnapshot {
        max: { peak_rss_bytes },
        latest: {
            block_cache_usage_bytes,
            block_cache_pinned_bytes,
            memtable_usage_bytes,
            pending_compaction_bytes,
            live_data_bytes,
            live_sst_bytes,
            node_keys_estimate,
            reverse_edge_keys_estimate,
            work_keys_estimate,
            node_table_reader_bytes,
            reverse_edge_table_reader_bytes,
            work_table_reader_bytes,
            level_zero_files,
            running_compactions,
            running_flushes,
            background_errors,
            delayed_write_rate,
            write_stopped,
            block_cache_hits,
            block_cache_misses,
            block_cache_data_hits,
            block_cache_data_misses,
            block_cache_index_hits,
            block_cache_index_misses,
            block_cache_filter_hits,
            block_cache_filter_misses,
            bloom_filter_useful,
            bloom_filter_full_positive,
            bloom_filter_full_true_positive,
            memtable_hits,
            memtable_misses,
            get_hits_l0,
            get_hits_l1,
            get_hits_l2_and_up,
            db_seeks,
            db_nexts,
            iterator_bytes_read,
            keys_read,
            bytes_read,
            stall_micros,
            bytes_written,
            compaction_read_bytes,
            compaction_write_bytes,
            flush_write_bytes,
        },
    }
}

#[derive(Default)]
struct RuntimeStats {
    outstanding_work: AtomicU64,
    nodes_created: AtomicU64,
    nodes_dead: AtomicU64,
    nodes_created_this_run: AtomicU64,
    nodes_dead_this_run: AtomicU64,
    attach_accepted: AtomicU64,
    attach_rejected: AtomicU64,
    commands: AtomicCommandStats,
    expansion: AtomicExpansionStats,
    candidates: AtomicCandidateStats,
    node_reads: AtomicNodeReadStats,
    dead_filter: AtomicDeadFilterStats,
    reverse_edges: AtomicReverseEdgeStats,
    batches: AtomicBatchStats,
    owners: Vec<AtomicOwnerStats>,
    max_queue_skew: AtomicU64,
    storage: AtomicStorageStats,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeStatsSnapshot {
    outstanding_work: u64,
    nodes_created: u64,
    nodes_dead: u64,
    nodes_not_dead: u64,
    nodes_created_this_run: u64,
    nodes_dead_this_run: u64,
    attach_accepted: u64,
    attach_rejected: u64,
    commands: CommandStatsSnapshot,
    expansion: ExpansionStatsSnapshot,
    candidates: CandidateStatsSnapshot,
    node_reads: NodeReadStatsSnapshot,
    dead_filter: DeadFilterStatsSnapshot,
    reverse_edges: ReverseEdgeStatsSnapshot,
    closure: ClosureStatsSnapshot,
    batches: BatchStatsSnapshot,
    queue_min: u64,
    queue_max: u64,
    queue_skew: u64,
    max_queue_skew: u64,
    owners: Vec<OwnerStatsSnapshot>,
    storage: StorageStatsSnapshot,
}

impl RuntimeStats {
    fn new(writer_count: usize, seed_owner: u8) -> Self {
        let seed_closure = ClosureGauges {
            live_states: 1,
            states_with_unexpanded: 1,
            forced_slots_total: TetrisPiece::NUM_PIECES as i64,
            forced_slots_unexpanded: TetrisPiece::NUM_PIECES as i64,
            ..ClosureGauges::default()
        };
        let owners = (0..writer_count)
            .map(|owner| {
                let owns_seed = owner == seed_owner as usize;
                AtomicOwnerStats::new(
                    u64::from(owns_seed),
                    if owns_seed {
                        seed_closure
                    } else {
                        ClosureGauges::default()
                    },
                )
            })
            .collect();
        let mut produced = [0; COMMAND_KIND_COUNT];
        produced[COMMAND_EXPAND as usize] = 1;
        Self {
            outstanding_work: AtomicU64::new(1),
            nodes_created: AtomicU64::new(1),
            commands: AtomicCommandStats {
                produced: AtomicCommandCounts::new(produced),
                ..AtomicCommandStats::default()
            },
            owners,
            ..Self::default()
        }
    }

    fn recovered(recovery: &RecoverySnapshot) -> Self {
        let owners = recovery
            .owner_outstanding
            .iter()
            .copied()
            .zip(recovery.owner_closure.iter().copied())
            .map(|(outstanding, closure)| AtomicOwnerStats::new(outstanding, closure))
            .collect();
        Self {
            outstanding_work: AtomicU64::new(recovery.outstanding_work),
            nodes_created: AtomicU64::new(recovery.nodes_created),
            nodes_dead: AtomicU64::new(recovery.nodes_dead),
            commands: AtomicCommandStats {
                carried_in: AtomicCommandCounts::new(recovery.commands_carried_in),
                ..AtomicCommandStats::default()
            },
            dead_filter: AtomicDeadFilterStats {
                inserts: AtomicU64::new(recovery.nodes_dead),
                bits_set: AtomicU64::new(recovery.dead_filter_bits_set),
                ..AtomicDeadFilterStats::default()
            },
            owners,
            ..Self::default()
        }
    }

    fn reserve_outputs(
        &self,
        output_count: u64,
        outputs_by_owner: &[u64; VIRTUAL_SHARDS],
    ) -> Result<()> {
        atomic_checked_add(&self.outstanding_work, output_count)?;
        for (owner, &count) in outputs_by_owner.iter().enumerate().take(self.owners.len()) {
            if count != 0 {
                atomic_checked_add(&self.owners[owner].outstanding, count)?;
            }
        }
        Ok(())
    }

    fn record_claim(&self, claimed_commands: u64, elapsed: Duration) {
        let micros = duration_micros(elapsed);
        self.batches
            .claim_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        if claimed_commands == 0 {
            self.batches
                .empty_claims
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
        self.batches
            .claimed_commands_total
            .fetch_add(claimed_commands, AtomicOrdering::Relaxed);
        self.batches
            .claimed_commands_max
            .fetch_max(claimed_commands, AtomicOrdering::Relaxed);
        self.batches
            .claim_micros_total
            .fetch_add(micros, AtomicOrdering::Relaxed);
        self.batches
            .claim_micros_max
            .fetch_max(micros, AtomicOrdering::Relaxed);
    }

    fn record_process(&self, elapsed: Duration) {
        let micros = duration_micros(elapsed);
        self.batches
            .process_micros_total
            .fetch_add(micros, AtomicOrdering::Relaxed);
        self.batches
            .process_micros_max
            .fetch_max(micros, AtomicOrdering::Relaxed);
    }

    fn record_commit(
        &self,
        owner: usize,
        metrics: &BatchMetrics,
        input_count: u64,
        output_count: u64,
        batch_bytes: u64,
        commit_micros: u64,
    ) -> Result<()> {
        atomic_checked_sub(&self.outstanding_work, input_count)?;
        atomic_checked_sub(&self.owners[owner].outstanding, input_count)?;
        self.commands.produced.add(&metrics.commands_produced);
        self.commands.processed.add(&metrics.commands_processed);
        self.commands.stale.add(&metrics.commands_stale);
        atomic_add_relaxed(&self.nodes_created, metrics.nodes_created);
        atomic_add_relaxed(&self.nodes_dead, metrics.nodes_dead);
        atomic_add_relaxed(&self.nodes_created_this_run, metrics.nodes_created);
        atomic_add_relaxed(&self.nodes_dead_this_run, metrics.nodes_dead);
        atomic_add_relaxed(&self.attach_accepted, metrics.attach_accepted);
        atomic_add_relaxed(&self.attach_rejected, metrics.attach_rejected);
        self.expansion.add(&metrics.expansion);
        self.candidates.add(&metrics.candidates);
        self.node_reads.add(&metrics.node_reads);
        self.dead_filter.add(&metrics.dead_filter);
        self.reverse_edges.add(&metrics.reverse_edges);
        self.batches.committed.fetch_add(1, AtomicOrdering::Relaxed);
        self.batches
            .commands_total
            .fetch_add(input_count, AtomicOrdering::Relaxed);
        self.batches
            .commands_max
            .fetch_max(input_count, AtomicOrdering::Relaxed);
        self.batches
            .bytes_total
            .fetch_add(batch_bytes, AtomicOrdering::Relaxed);
        self.batches
            .bytes_max
            .fetch_max(batch_bytes, AtomicOrdering::Relaxed);
        self.batches
            .commit_micros_total
            .fetch_add(commit_micros, AtomicOrdering::Relaxed);
        self.batches
            .commit_micros_max
            .fetch_max(commit_micros, AtomicOrdering::Relaxed);
        atomic_add_relaxed(
            &self.batches.work_heads_advanced,
            metrics.work_heads_advanced,
        );
        atomic_add_relaxed(
            &self.batches.work_reclaim_ranges,
            metrics.work_reclaim_ranges,
        );
        let stopped_by = match metrics.stop_reason {
            Some(BatchStopReason::WriteBytes) => Some(&self.batches.stopped_by_write_bytes),
            Some(BatchStopReason::Candidates) => Some(&self.batches.stopped_by_candidates),
            Some(BatchStopReason::ReverseEdges) => Some(&self.batches.stopped_by_reverse_edges),
            Some(BatchStopReason::Outputs) => Some(&self.batches.stopped_by_outputs),
            None => None,
        };
        if let Some(counter) = stopped_by {
            counter.fetch_add(1, AtomicOrdering::Relaxed);
        }

        let owner_stats = &self.owners[owner];
        owner_stats.closure.add(metrics);
        owner_stats
            .commands_processed
            .fetch_add(input_count, AtomicOrdering::Relaxed);
        owner_stats
            .commands_produced
            .fetch_add(output_count, AtomicOrdering::Relaxed);
        owner_stats.batches.fetch_add(1, AtomicOrdering::Relaxed);
        atomic_add_relaxed(
            &owner_stats.candidates_examined,
            metrics.candidates.examined,
        );
        owner_stats
            .batch_bytes_total
            .fetch_add(batch_bytes, AtomicOrdering::Relaxed);
        owner_stats
            .batch_commands_max
            .fetch_max(input_count, AtomicOrdering::Relaxed);
        owner_stats
            .batch_bytes_max
            .fetch_max(batch_bytes, AtomicOrdering::Relaxed);
        owner_stats
            .commit_micros_total
            .fetch_add(commit_micros, AtomicOrdering::Relaxed);
        owner_stats
            .commit_micros_max
            .fetch_max(commit_micros, AtomicOrdering::Relaxed);
        Ok(())
    }

    fn record_busy(&self, owner: usize, elapsed: Duration) {
        self.owners[owner]
            .busy_micros
            .fetch_add(duration_micros(elapsed), AtomicOrdering::Relaxed);
    }

    fn record_idle(&self, owner: usize, elapsed: Duration) {
        self.owners[owner]
            .idle_micros
            .fetch_add(duration_micros(elapsed), AtomicOrdering::Relaxed);
        self.owners[owner]
            .idle_polls
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    fn snapshot(&self) -> RuntimeStatsSnapshot {
        let owners: Vec<_> = self
            .owners
            .iter()
            .enumerate()
            .map(|(owner, stats)| stats.snapshot(owner))
            .collect();
        let queue_min = owners
            .iter()
            .map(|owner| owner.outstanding)
            .min()
            .unwrap_or(0);
        let queue_max = owners
            .iter()
            .map(|owner| owner.outstanding)
            .max()
            .unwrap_or(0);
        let queue_skew = queue_max - queue_min;
        self.max_queue_skew
            .fetch_max(queue_skew, AtomicOrdering::Relaxed);
        let nodes_created = self.nodes_created.load(AtomicOrdering::Relaxed);
        let nodes_dead = self.nodes_dead.load(AtomicOrdering::Relaxed);
        let nodes_created_this_run = self.nodes_created_this_run.load(AtomicOrdering::Relaxed);
        let nodes_dead_this_run = self.nodes_dead_this_run.load(AtomicOrdering::Relaxed);
        let outstanding_work = self.outstanding_work.load(AtomicOrdering::SeqCst);
        let commands = CommandStatsSnapshot {
            carried_in: self.commands.carried_in.snapshot(),
            produced: self.commands.produced.snapshot(),
            processed: self.commands.processed.snapshot(),
            stale: self.commands.stale.snapshot(),
        };
        let reverse_edges = self.reverse_edges.snapshot();
        let mut closure_gauges = ClosureOwnerSnapshot::default();
        for owner in &self.owners {
            closure_gauges.add(owner.closure.snapshot());
        }
        let closure = ClosureStatsSnapshot::from_runtime(
            closure_gauges,
            &commands,
            nodes_created,
            nodes_dead,
            nodes_created_this_run,
            nodes_dead_this_run,
            outstanding_work,
            reverse_edges.drained,
        );
        let dead_filter = {
            let counters = self.dead_filter.snapshot();
            let capacity_bits = (DEAD_STATE_BLOOM_BYTES as u64) * 8;
            DeadFilterStatsSnapshot {
                bytes: DEAD_STATE_BLOOM_BYTES as u64,
                hashes: DEAD_STATE_BLOOM_HASHES,
                capacity_bits,
                set_bits: counters.bits_set,
                occupancy: ratio(counters.bits_set, capacity_bits),
                checks: counters.checks,
                definite_negatives: counters.definite_negatives,
                maybe_positives: counters.maybe_positives,
                verified_dead: counters.verified_dead,
                false_positives: counters.false_positives,
                inserts: counters.inserts,
            }
        };
        RuntimeStatsSnapshot {
            outstanding_work,
            nodes_created,
            nodes_dead,
            nodes_not_dead: nodes_created.saturating_sub(nodes_dead),
            nodes_created_this_run,
            nodes_dead_this_run,
            attach_accepted: self.attach_accepted.load(AtomicOrdering::Relaxed),
            attach_rejected: self.attach_rejected.load(AtomicOrdering::Relaxed),
            commands,
            expansion: self.expansion.snapshot(),
            candidates: self.candidates.snapshot(),
            node_reads: self.node_reads.snapshot(),
            dead_filter,
            reverse_edges,
            closure,
            batches: snapshot_fields!(
                self.batches,
                BatchStatsSnapshot {
                    claim_calls,
                    empty_claims,
                    claimed_commands_total,
                    claimed_commands_max,
                    claim_micros_total,
                    claim_micros_max,
                    process_micros_total,
                    process_micros_max,
                    committed,
                    commands_total,
                    commands_max,
                    bytes_total,
                    bytes_max,
                    commit_micros_total,
                    commit_micros_max,
                    work_heads_advanced,
                    work_reclaim_ranges,
                    stopped_by_write_bytes,
                    stopped_by_candidates,
                    stopped_by_reverse_edges,
                    stopped_by_outputs,
                }
            ),
            queue_min,
            queue_max,
            queue_skew,
            max_queue_skew: self.max_queue_skew.load(AtomicOrdering::Relaxed),
            owners,
            storage: self.storage.snapshot(),
        }
    }
}

fn duration_micros(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CandidateReuse {
    DefinitelyNew,
    MayExist,
    Existing,
}

impl CandidateReuse {
    const fn is_likely_existing(self) -> bool {
        !matches!(self, Self::DefinitelyNew)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateState {
    Dead,
    Viable(CandidateReuse),
}

#[derive(Clone, Copy)]
struct Candidate {
    reuse: CandidateReuse,
    score: f32,
    orientation: TetrisPieceOrientation,
    succ: StateKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ChoiceKey {
    pred: StateKey,
    piece: TetrisPiece,
}

#[derive(Clone, Copy)]
enum PreparedPlacement {
    Lost,
    RejectedHeight,
    RejectedHoles,
    Admissible {
        succ: StateKey,
        score: f32,
        orientation: TetrisPieceOrientation,
    },
}

struct PreparedChoice {
    key: ChoiceKey,
    placement_start: usize,
    placement_end: usize,
    used: bool,
}

const MAX_EXPANSION_MOVES: usize = TetrisPiece::NUM_PIECES;

#[derive(Clone, Copy)]
struct PlannedMove {
    piece: TetrisPiece,
    candidate: Candidate,
}

type ExpansionPlan = HeaplessVec<PlannedMove, MAX_EXPANSION_MOVES>;

fn candidate_order(left: &Candidate, right: &Candidate) -> Ordering {
    left.reuse
        .cmp(&right.reuse)
        .then_with(|| left.score.total_cmp(&right.score))
        .then_with(|| right.orientation.index().cmp(&left.orientation.index()))
}

#[inline]
const fn is_admissible(board: TetrisBoard) -> bool {
    board.height() <= MAX_CARRIER_HEIGHT && board.total_holes() <= MAX_CARRIER_HOLES
}

fn successor(
    state: StateKey,
    piece: TetrisPiece,
    orientation: TetrisPieceOrientation,
) -> Result<Option<StateKey>> {
    let (mut board, bag) = state.decode()?;
    ensure!(
        bag.contains(piece),
        "piece is not forced by predecessor bag"
    );
    validate_orientation(piece, orientation.index())?;
    let result = board.apply_piece_placement(TetrisPiecePlacement { piece, orientation });
    if result.is_lost == IsLost::LOST {
        return Ok(None);
    }
    let mut next_bag = bag;
    next_bag.remove(piece);
    if next_bag.is_empty() {
        next_bag.fill();
    }
    if !is_admissible(board) {
        return Ok(None);
    }
    Ok(Some(StateKey::from_board_bag(board, next_bag)?))
}

fn score_board(board: TetrisBoard, recent_lines_cleared: u32) -> Result<f32> {
    let score = height_mse_board_score(&TetrisBoardScoreState {
        board,
        recent_lines_cleared,
    });
    ensure!(!score.is_nan(), "height_mse_board_score returned NaN");
    Ok(score)
}

fn choice_context(
    pred: StateKey,
    piece: TetrisPiece,
) -> Result<(TetrisBoard, TetrisPieceBagState)> {
    let (board, bag) = pred.decode()?;
    ensure!(bag.contains(piece), "piece is not forced by predecessor");
    let mut next_bag = bag;
    next_bag.remove(piece);
    if next_bag.is_empty() {
        next_bag.fill();
    }
    Ok((board, next_bag))
}

fn prepare_placement(
    board: TetrisBoard,
    next_bag: TetrisPieceBagState,
    placement: TetrisPiecePlacement,
) -> Result<PreparedPlacement> {
    let mut next_board = board;
    let result = next_board.apply_piece_placement(placement);
    if result.is_lost == IsLost::LOST {
        return Ok(PreparedPlacement::Lost);
    }
    if next_board.height() > MAX_CARRIER_HEIGHT {
        return Ok(PreparedPlacement::RejectedHeight);
    }
    if next_board.total_holes() > MAX_CARRIER_HOLES {
        return Ok(PreparedPlacement::RejectedHoles);
    }
    Ok(PreparedPlacement::Admissible {
        succ: StateKey::from_board_bag(next_board, next_bag)?,
        score: score_board(next_board, result.lines_cleared)?,
        orientation: placement.orientation,
    })
}

#[derive(Clone, Copy)]
struct ClaimedCommand {
    key: WorkKey,
    command: Command,
}

type ClaimedBatch = HeaplessVec<ClaimedCommand, WORK_BATCH_COMMANDS>;
type ProcessedBatch = HeaplessVec<WorkKey, WORK_BATCH_COMMANDS>;
type DirtyNodeBatch = HeaplessVec<StateKey, WORK_BATCH_COMMANDS>;

#[derive(Clone, Copy)]
struct NodeOverlayEntry {
    original: Option<NodeRecord>,
    record: Option<NodeRecord>,
    dirty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OutputCommand {
    key: WorkKey,
    command: Command,
}

struct WorkerScratch {
    nodes: FxHashMap<StateKey, NodeOverlayEntry>,
    dirty_nodes: DirtyNodeBatch,
    node_prefetch_plan: Vec<StateKey>,
    choice_requests: Vec<ChoiceKey>,
    prepared_choices: Vec<PreparedChoice>,
    prepared_placements: Vec<PreparedPlacement>,
    redges: FxHashMap<ReverseEdgeKey, bool>,
    outputs: Vec<OutputCommand>,
    processed: ProcessedBatch,
    ordered_redges: Vec<ReverseEdgeKey>,
    drain_edges: Vec<ReverseEdgeKey>,
}

impl WorkerScratch {
    fn new() -> Result<Self> {
        let mut nodes = FxHashMap::default();
        nodes
            .try_reserve(MAX_NODE_OVERLAY_ENTRIES)
            .context("allocate node overlay scratch")?;
        let mut redges = FxHashMap::default();
        redges
            .try_reserve(MAX_REDGE_MUTATIONS_PER_BATCH)
            .context("allocate reverse-edge overlay scratch")?;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(MAX_OUTPUT_COMMANDS_PER_BATCH)
            .context("allocate output command scratch")?;
        let mut node_prefetch_plan = Vec::new();
        node_prefetch_plan
            .try_reserve_exact(MAX_NODE_OVERLAY_ENTRIES)
            .context("allocate node prefetch scratch")?;
        let mut choice_requests = Vec::new();
        choice_requests
            .try_reserve_exact(MAX_CHOICE_REQUESTS_PER_BATCH)
            .context("allocate choice request scratch")?;
        let mut prepared_choices = Vec::new();
        prepared_choices
            .try_reserve_exact(MAX_CHOICE_REQUESTS_PER_BATCH)
            .context("allocate prepared choice scratch")?;
        let mut prepared_placements = Vec::new();
        prepared_placements
            .try_reserve_exact(MAX_CANDIDATES_PER_BATCH)
            .context("allocate prepared placement scratch")?;
        let mut ordered_redges = Vec::new();
        ordered_redges
            .try_reserve_exact(MAX_REDGE_MUTATIONS_PER_BATCH)
            .context("allocate ordered reverse-edge scratch")?;
        let mut drain_edges = Vec::new();
        drain_edges
            .try_reserve_exact(MAX_REDGES_PER_DRAIN + 1)
            .context("allocate reverse-edge drain scratch")?;
        Ok(Self {
            nodes,
            dirty_nodes: DirtyNodeBatch::new(),
            node_prefetch_plan,
            choice_requests,
            prepared_choices,
            prepared_placements,
            redges,
            outputs,
            processed: ProcessedBatch::new(),
            ordered_redges,
            drain_edges,
        })
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.dirty_nodes.clear();
        self.node_prefetch_plan.clear();
        self.choice_requests.clear();
        self.prepared_choices.clear();
        self.prepared_placements.clear();
        self.redges.clear();
        self.outputs.clear();
        self.processed.clear();
        self.ordered_redges.clear();
        self.drain_edges.clear();
    }
}

struct BatchProcessor<'a> {
    db: &'a DB,
    node_cf: Arc<BoundColumnFamily<'a>>,
    read_options: ReadOptions,
    owners: &'a OwnerMap,
    dead_filter: &'a DeadStateBloom,
    owner: u8,
    next_sequence: &'a mut u64,
    scratch: &'a mut WorkerScratch,
    estimated_bytes: usize,
    metrics: BatchMetrics,
    seed_died: bool,
}

impl<'a> BatchProcessor<'a> {
    fn new(
        db: &'a DB,
        owners: &'a OwnerMap,
        dead_filter: &'a DeadStateBloom,
        owner: u8,
        next_sequence: &'a mut u64,
        scratch: &'a mut WorkerScratch,
    ) -> Result<Self> {
        scratch.clear();
        let node_cf = cf(&db, CF_NODE)?;
        Ok(Self {
            db,
            node_cf,
            read_options: ReadOptions::default(),
            owners,
            dead_filter,
            owner,
            next_sequence,
            scratch,
            estimated_bytes: 0,
            metrics: BatchMetrics::default(),
            seed_died: false,
        })
    }

    fn process_claimed(&mut self, claimed: &[ClaimedCommand]) -> Result<()> {
        self.prepare_and_prefetch_batch(claimed)?;
        for item in claimed {
            if !self.scratch.processed.is_empty() {
                let (redges, outputs) = command_scratch_requirements(item.command);
                let stop_reason =
                    if self.estimated_bytes + MAX_COMMAND_WRITE_BYTES > MAX_WRITE_BATCH_BYTES {
                        Some(BatchStopReason::WriteBytes)
                    } else if self.metrics.candidates.examined + MAX_CANDIDATES_PER_COMMAND as u64
                        > MAX_CANDIDATES_PER_BATCH as u64
                    {
                        Some(BatchStopReason::Candidates)
                    } else if self.scratch.redges.len() + redges > MAX_REDGE_MUTATIONS_PER_BATCH {
                        Some(BatchStopReason::ReverseEdges)
                    } else if self.scratch.outputs.len() + outputs > MAX_OUTPUT_COMMANDS_PER_BATCH {
                        Some(BatchStopReason::Outputs)
                    } else {
                        None
                    };
                if let Some(stop_reason) = stop_reason {
                    self.metrics.stop_reason = Some(stop_reason);
                    break;
                }
            }
            self.owners.validate_work(item.key, item.command)?;
            ensure!(
                item.key.owner() == self.owner,
                "writer {} received work owned by {}",
                self.owner,
                item.key.owner()
            );
            self.estimated_bytes += WORK_KEY_LEN;
            self.apply(item.command)?;
            self.metrics.commands_processed[item.command.kind_index()] += 1;
            ensure!(
                self.scratch.processed.try_push(item.key),
                "processed work exceeds fixed batch capacity"
            );
        }
        self.metrics.candidates.prepared_choices_unused = self
            .scratch
            .prepared_choices
            .iter()
            .filter(|choice| !choice.used)
            .count() as u64;
        ensure!(
            !self.scratch.processed.is_empty(),
            "claimed batch made no progress"
        );
        ensure!(
            self.estimated_bytes <= MAX_WRITE_BATCH_BYTES,
            "projected write batch exceeds {MAX_WRITE_BATCH_BYTES} bytes"
        );
        ensure!(
            self.metrics.candidates.examined <= MAX_CANDIDATES_PER_BATCH as u64,
            "candidate scratch limit exceeded"
        );
        Ok(())
    }

    fn push_prefetch_key(&mut self, state: StateKey) -> Result<()> {
        ensure!(
            self.scratch.node_prefetch_plan.len() < MAX_NODE_OVERLAY_ENTRIES,
            "node prefetch plan exceeds fixed batch capacity"
        );
        self.scratch.node_prefetch_plan.push(state);
        Ok(())
    }

    fn push_choice_request(&mut self, key: ChoiceKey) -> Result<()> {
        ensure!(
            self.scratch.choice_requests.len() < MAX_CHOICE_REQUESTS_PER_BATCH,
            "choice request plan exceeds fixed batch capacity"
        );
        self.scratch.choice_requests.push(key);
        Ok(())
    }

    fn request_choice_if_forced(&mut self, pred: StateKey, piece: TetrisPiece) -> Result<()> {
        if pred.bag()?.contains(piece) {
            self.push_choice_request(ChoiceKey { pred, piece })?;
        }
        Ok(())
    }

    fn collect_batch_read_requests(&mut self, claimed: &[ClaimedCommand]) -> Result<()> {
        for item in claimed {
            if let Some(&state) = item.command.node_read_key() {
                self.push_prefetch_key(state)?;
            }
            match item.command {
                Command::Expand { state } => {
                    for piece in state.bag()?.iter_pieces() {
                        self.push_choice_request(ChoiceKey { pred: state, piece })?;
                    }
                }
                Command::AttachResult {
                    accepted: false,
                    pred,
                    piece,
                    ..
                }
                | Command::ChildDead { pred, piece, .. } => {
                    self.request_choice_if_forced(pred, piece)?;
                }
                Command::AttachEdge { .. }
                | Command::AttachResult { accepted: true, .. }
                | Command::DetachEdge { .. }
                | Command::DrainDead { .. } => {}
            }
        }
        self.metrics.candidates.prepared_choice_requests =
            self.scratch.choice_requests.len() as u64;
        self.scratch.choice_requests.sort_unstable();
        self.scratch.choice_requests.dedup();
        self.metrics.candidates.prepared_choice_unique = self.scratch.choice_requests.len() as u64;
        Ok(())
    }

    fn prepare_choice(&mut self, key: ChoiceKey) -> Result<()> {
        ensure!(
            self.scratch.prepared_choices.len() < MAX_CHOICE_REQUESTS_PER_BATCH,
            "prepared choices exceed fixed batch capacity"
        );
        let placement_start = self.scratch.prepared_placements.len();
        let (board, next_bag) = choice_context(key.pred, key.piece)?;
        for &placement in TetrisPiecePlacement::all_from_piece(key.piece) {
            ensure!(
                self.scratch.prepared_placements.len() < MAX_CANDIDATES_PER_BATCH,
                "prepared placements exceed fixed batch capacity"
            );
            let prepared = prepare_placement(board, next_bag, placement)?;
            if let PreparedPlacement::Admissible { succ, .. } = prepared {
                self.push_prefetch_key(succ)?;
            }
            self.scratch.prepared_placements.push(prepared);
        }
        self.scratch.prepared_choices.push(PreparedChoice {
            key,
            placement_start,
            placement_end: self.scratch.prepared_placements.len(),
            used: false,
        });
        Ok(())
    }

    fn prepare_and_prefetch_batch(&mut self, claimed: &[ClaimedCommand]) -> Result<()> {
        self.collect_batch_read_requests(claimed)?;
        for index in 0..self.scratch.choice_requests.len() {
            self.prepare_choice(self.scratch.choice_requests[index])?;
        }
        self.metrics.candidates.prepared_placements = self.scratch.prepared_placements.len() as u64;

        self.scratch.node_prefetch_plan.sort_unstable();
        self.scratch.node_prefetch_plan.dedup();
        if self.scratch.node_prefetch_plan.is_empty() {
            return Ok(());
        }
        ensure!(
            self.scratch.nodes.len() + self.scratch.node_prefetch_plan.len()
                <= MAX_NODE_OVERLAY_ENTRIES,
            "node prefetch exceeds fixed overlay capacity"
        );
        let results = self.db.batched_multi_get_cf_opt(
            &self.node_cf,
            self.scratch
                .node_prefetch_plan
                .iter()
                .map(StateKey::as_bytes),
            true,
            &self.read_options,
        );
        ensure!(
            results.len() == self.scratch.node_prefetch_plan.len(),
            "RocksDB returned the wrong batched node-read count"
        );
        self.metrics.node_reads.prefetch_calls += 1;
        self.metrics.node_reads.prefetch_keys += results.len() as u64;
        let scratch = &mut *self.scratch;
        for (&state, result) in scratch.node_prefetch_plan.iter().zip(results) {
            let record = result
                .context("batched read node record")?
                .map(|bytes| NodeRecord::from_slice(bytes.as_ref(), state.bag()?))
                .transpose()?;
            self.metrics.node_reads.db_gets += 1;
            if record.is_some() {
                self.metrics.node_reads.db_found += 1;
            } else {
                self.metrics.node_reads.db_missing += 1;
            }
            scratch.nodes.insert(
                state,
                NodeOverlayEntry {
                    original: record,
                    record,
                    dirty: false,
                },
            );
        }
        Ok(())
    }

    fn apply(&mut self, command: Command) -> Result<()> {
        match command {
            Command::Expand { state } => self.expand(state),
            Command::AttachEdge {
                succ,
                pred,
                piece,
                orientation,
            } => self.attach_edge(succ, pred, piece, orientation),
            Command::AttachResult {
                accepted,
                succ,
                pred,
                piece,
                orientation,
            } => self.attach_result(accepted, succ, pred, piece, orientation),
            Command::ChildDead {
                succ,
                pred,
                piece,
                orientation,
            } => self.child_dead(succ, pred, piece, orientation),
            Command::DetachEdge {
                succ,
                pred,
                piece,
                orientation,
            } => self.detach_edge(succ, pred, piece, orientation),
            Command::DrainDead { succ } => self.drain_dead(succ),
        }
    }

    fn load_node(&mut self, state: StateKey) -> Result<Option<NodeRecord>> {
        self.metrics.node_reads.lookups += 1;
        if let Some(entry) = self.scratch.nodes.get(&state) {
            self.metrics.node_reads.overlay_hits += 1;
            return Ok(entry.record);
        }
        self.load_node_from_db(state)
    }

    fn load_node_from_db(&mut self, state: StateKey) -> Result<Option<NodeRecord>> {
        ensure!(
            self.scratch.nodes.len() < MAX_NODE_OVERLAY_ENTRIES,
            "node overlay exceeds fixed batch capacity"
        );
        let record = self
            .db
            .get_pinned_cf_opt(&self.node_cf, state.as_bytes(), &self.read_options)
            .context("read node record")?
            .map(|bytes| NodeRecord::from_slice(bytes.as_ref(), state.bag()?))
            .transpose()?;
        self.metrics.node_reads.db_gets += 1;
        if record.is_some() {
            self.metrics.node_reads.db_found += 1;
        } else {
            self.metrics.node_reads.db_missing += 1;
        }
        self.scratch.nodes.insert(
            state,
            NodeOverlayEntry {
                original: record,
                record,
                dirty: false,
            },
        );
        Ok(record)
    }

    fn put_node(&mut self, state: StateKey, record: NodeRecord) -> Result<()> {
        ensure!(
            self.owners.owner(state) == self.owner,
            "writer {} attempted to mutate a node owned by {}",
            self.owner,
            self.owners.owner(state)
        );
        record.validate(state.bag()?)?;
        ensure!(
            self.scratch.nodes.contains_key(&state)
                || self.scratch.nodes.len() < MAX_NODE_OVERLAY_ENTRIES,
            "node overlay exceeds fixed batch capacity"
        );
        match self.scratch.nodes.get_mut(&state) {
            Some(entry) => {
                if !entry.dirty {
                    ensure!(
                        self.scratch.dirty_nodes.try_push(state),
                        "dirty nodes exceed processed command capacity"
                    );
                }
                entry.record = Some(record);
                entry.dirty = true;
            }
            None => {
                ensure!(
                    self.scratch.dirty_nodes.try_push(state),
                    "dirty nodes exceed processed command capacity"
                );
                self.scratch.nodes.insert(
                    state,
                    NodeOverlayEntry {
                        original: None,
                        record: Some(record),
                        dirty: true,
                    },
                );
            }
        }
        self.estimated_bytes += STATE_KEY_LEN + NODE_RECORD_LEN;
        Ok(())
    }

    fn output(&mut self, command: Command) -> Result<()> {
        ensure!(
            self.owners.owner(command.source()) == self.owner,
            "writer {} cannot produce command sourced by owner {}",
            self.owner,
            self.owners.owner(command.source())
        );
        ensure!(
            self.scratch.outputs.len() < MAX_OUTPUT_COMMANDS_PER_BATCH,
            "output commands exceed fixed batch capacity"
        );
        ensure!(
            *self.next_sequence != u64::MAX,
            "producer {} exhausted its work sequence",
            self.owner
        );
        let key = WorkKey::new(
            self.owners.owner(command.target()),
            command.priority(),
            self.owner,
            *self.next_sequence,
        )?;
        *self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .context("producer work sequence overflow")?;
        self.owners.validate_work(key, command)?;
        self.estimated_bytes += WORK_KEY_LEN + command.encode()?.len();
        self.scratch.outputs.push(OutputCommand { key, command });
        self.metrics.commands_produced[command.kind_index()] += 1;
        Ok(())
    }

    fn record_candidate_reuse(&mut self, reuse: CandidateReuse) {
        match reuse {
            CandidateReuse::DefinitelyNew => self.metrics.candidates.definitely_new += 1,
            CandidateReuse::MayExist => self.metrics.candidates.may_exist += 1,
            CandidateReuse::Existing => {}
        }
    }

    fn record_published_candidate(&mut self, reuse: CandidateReuse) {
        if reuse.is_likely_existing() {
            self.metrics.candidates.chosen_likely_existing += 1;
        } else {
            self.metrics.candidates.chosen_definitely_new += 1;
        }
    }

    fn classify_candidate(&mut self, state: StateKey) -> Result<CandidateState> {
        self.metrics.node_reads.lookups += 1;
        self.metrics.node_reads.candidate_lookups += 1;
        if let Some(entry) = self.scratch.nodes.get(&state) {
            self.metrics.node_reads.overlay_hits += 1;
            self.metrics.node_reads.candidate_overlay_hits += 1;
            let Some(record) = entry.record else {
                self.record_candidate_reuse(CandidateReuse::DefinitelyNew);
                return Ok(CandidateState::Viable(CandidateReuse::DefinitelyNew));
            };
            if record.is_dead(state.bag()?)? {
                return Ok(CandidateState::Dead);
            }
            self.metrics.candidates.existing_overlay += 1;
            return Ok(CandidateState::Viable(CandidateReuse::Existing));
        }

        self.metrics.dead_filter.checks += 1;
        if self.dead_filter.might_contain(state) {
            self.metrics.dead_filter.maybe_positives += 1;
            self.metrics.node_reads.candidate_db_gets += 1;
            let Some(record) = self.load_node_from_db(state)? else {
                self.metrics.dead_filter.false_positives += 1;
                self.record_candidate_reuse(CandidateReuse::DefinitelyNew);
                return Ok(CandidateState::Viable(CandidateReuse::DefinitelyNew));
            };
            if record.is_dead(state.bag()?)? {
                self.metrics.dead_filter.verified_dead += 1;
                return Ok(CandidateState::Dead);
            }
            self.metrics.dead_filter.false_positives += 1;
            self.metrics.candidates.existing_database += 1;
            return Ok(CandidateState::Viable(CandidateReuse::Existing));
        }

        self.metrics.dead_filter.definite_negatives += 1;
        self.metrics.candidates.reuse_probes += 1;
        let reuse =
            if self
                .db
                .key_may_exist_cf_opt(&self.node_cf, state.as_bytes(), &self.read_options)
            {
                CandidateReuse::MayExist
            } else {
                CandidateReuse::DefinitelyNew
            };
        self.record_candidate_reuse(reuse);
        Ok(CandidateState::Viable(reuse))
    }

    #[cfg(test)]
    fn known_dead(&mut self, state: StateKey) -> Result<bool> {
        Ok(self.classify_candidate(state)? == CandidateState::Dead)
    }

    fn consider_prepared_placement(
        &mut self,
        prepared: PreparedPlacement,
        best: &mut Option<Candidate>,
    ) -> Result<()> {
        self.metrics.candidates.examined += 1;
        ensure!(
            self.metrics.candidates.examined <= MAX_CANDIDATES_PER_BATCH as u64,
            "candidate scratch limit exceeded"
        );
        let (succ, score, orientation) = match prepared {
            PreparedPlacement::Lost => {
                self.metrics.candidates.rejected_lost += 1;
                return Ok(());
            }
            PreparedPlacement::RejectedHeight => {
                self.metrics.candidates.rejected_height += 1;
                return Ok(());
            }
            PreparedPlacement::RejectedHoles => {
                self.metrics.candidates.rejected_holes += 1;
                return Ok(());
            }
            PreparedPlacement::Admissible {
                succ,
                score,
                orientation,
            } => (succ, score, orientation),
        };
        let CandidateState::Viable(reuse) = self.classify_candidate(succ)? else {
            self.metrics.candidates.rejected_known_dead += 1;
            return Ok(());
        };
        self.metrics.candidates.viable += 1;
        let candidate = Candidate {
            reuse,
            score,
            orientation,
            succ,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate_order(&candidate, current).is_gt())
        {
            *best = Some(candidate);
        }
        Ok(())
    }

    fn choose_next_scalar(
        &mut self,
        pred: StateKey,
        piece: TetrisPiece,
    ) -> Result<Option<Candidate>> {
        let (board, next_bag) = choice_context(pred, piece)?;
        let mut best = None;
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let prepared = prepare_placement(board, next_bag, placement)?;
            self.consider_prepared_placement(prepared, &mut best)?;
        }
        Ok(best)
    }

    fn choose_next(&mut self, pred: StateKey, piece: TetrisPiece) -> Result<Option<Candidate>> {
        let key = ChoiceKey { pred, piece };
        let Ok(index) = self
            .scratch
            .prepared_choices
            .binary_search_by_key(&key, |choice| choice.key)
        else {
            self.metrics.candidates.unplanned_fallbacks += 1;
            return self.choose_next_scalar(pred, piece);
        };
        let (placement_start, placement_end) = {
            let choice = &mut self.scratch.prepared_choices[index];
            choice.used = true;
            (choice.placement_start, choice.placement_end)
        };
        let mut best = None;
        for placement_index in placement_start..placement_end {
            let prepared = self.scratch.prepared_placements[placement_index];
            self.consider_prepared_placement(prepared, &mut best)?;
        }
        Ok(best)
    }

    fn expand(&mut self, state: StateKey) -> Result<()> {
        let bag = state.bag()?;
        let mut record = self
            .load_node(state)?
            .context("Expand referenced an absent node")?;
        if record.is_dead(bag)? {
            self.metrics.commands_stale[COMMAND_EXPAND as usize] += 1;
            return Ok(());
        }

        let mut plan = ExpansionPlan::new();

        // Phase 1: verify that every unexpanded forced piece has a candidate
        // before publishing any new edge for this state.
        for piece in bag.iter_pieces() {
            if record.slot(piece)? != MoveSlot::Unexpanded {
                continue;
            }

            let Some(candidate) = self.choose_next(state, piece)? else {
                // One exhausted forced piece makes the whole AND-state dead.
                // The local plan has emitted no work, so dropping it avoids
                // attachments and their matching cleanup commands.
                record.set_slot(piece, MoveSlot::Exhausted)?;
                self.metrics.expansion.doomed += 1;
                self.metrics.expansion.planned_attachments_avoided += plan.len() as u64;
                return self.transition_to_dead(state, record);
            };

            ensure!(
                plan.try_push(PlannedMove { piece, candidate }),
                "expansion plan exceeds the canonical bag size"
            );
        }

        if plan.is_empty() {
            return Ok(());
        }

        // Phase 2a: make the local record complete before exposing commands.
        // Every newly pending slot receives exactly one AttachEdge below.
        for planned in plan.to_slice() {
            record.set_slot(
                planned.piece,
                MoveSlot::Pending(planned.candidate.orientation.index()),
            )?;
        }
        self.put_node(state, record)?;

        // Phase 2b: publish in deterministic bag order. commit() persists
        // these commands atomically with the pending slots above.
        for planned in plan.to_slice() {
            self.record_published_candidate(planned.candidate.reuse);
            self.output(Command::AttachEdge {
                succ: planned.candidate.succ,
                pred: state,
                piece: planned.piece,
                orientation: planned.candidate.orientation,
            })?;
            self.metrics.expansion.planned_attachments_published += 1;
        }
        Ok(())
    }

    fn attach_edge(
        &mut self,
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    ) -> Result<()> {
        let succ_bag = succ.bag()?;
        let record = match self.load_node(succ)? {
            Some(record) => record,
            None => {
                let record = NodeRecord::INIT;
                self.put_node(succ, record)?;
                self.metrics.nodes_created += 1;
                self.output(Command::Expand { state: succ })?;
                record
            }
        };
        let accepted = !record.is_dead(succ_bag)?;
        if accepted {
            self.metrics.attach_accepted += 1;
            let edge = ReverseEdgeKey::new(succ, pred, piece, orientation)?;
            self.set_redge(edge, true)?;
            self.metrics.reverse_edges.inserted += 1;
            self.estimated_bytes += REDGE_KEY_LEN;
        } else {
            self.metrics.attach_rejected += 1;
        }
        self.output(Command::AttachResult {
            accepted,
            succ,
            pred,
            piece,
            orientation,
        })
    }

    fn attach_result(
        &mut self,
        accepted: bool,
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    ) -> Result<()> {
        let bag = pred.bag()?;
        let mut record = self
            .load_node(pred)?
            .context("AttachResult referenced an absent predecessor")?;
        if record.is_dead(bag)?
            || record.slot(piece)? != MoveSlot::Pending(orientation.index())
            || successor(pred, piece, orientation)? != Some(succ)
        {
            self.metrics.commands_stale[COMMAND_ATTACH_RESULT as usize] += 1;
            return Ok(());
        }
        if accepted {
            record.set_slot(piece, MoveSlot::Active(orientation.index()))?;
            self.put_node(pred, record)
        } else {
            self.reroute(pred, piece, record)
        }
    }

    fn child_dead(
        &mut self,
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    ) -> Result<()> {
        let bag = pred.bag()?;
        let record = self
            .load_node(pred)?
            .context("ChildDead referenced an absent predecessor")?;
        if record.is_dead(bag)?
            || !matches!(
                record.slot(piece)?,
                MoveSlot::Pending(index) | MoveSlot::Active(index)
                    if index == orientation.index()
            )
            || successor(pred, piece, orientation)? != Some(succ)
        {
            self.metrics.commands_stale[COMMAND_CHILD_DEAD as usize] += 1;
            return Ok(());
        }
        self.reroute(pred, piece, record)
    }

    fn reroute(
        &mut self,
        pred: StateKey,
        piece: TetrisPiece,
        mut record: NodeRecord,
    ) -> Result<()> {
        self.metrics.reroutes += 1;
        match record.slot(piece)? {
            MoveSlot::Active(_) => self.metrics.reroutes_from_active += 1,
            MoveSlot::Pending(_) => self.metrics.reroutes_from_pending += 1,
            MoveSlot::Unexpanded | MoveSlot::Exhausted => {
                bail!("reroute requires an active or pending move");
            }
        }
        if let Some(candidate) = self.choose_next(pred, piece)? {
            record.set_slot(piece, MoveSlot::Pending(candidate.orientation.index()))?;
            self.put_node(pred, record)?;
            self.record_published_candidate(candidate.reuse);
            self.output(Command::AttachEdge {
                succ: candidate.succ,
                pred,
                piece,
                orientation: candidate.orientation,
            })
        } else {
            record.set_slot(piece, MoveSlot::Exhausted)?;
            self.transition_to_dead(pred, record)
        }
    }

    fn transition_to_dead(&mut self, state: StateKey, record: NodeRecord) -> Result<()> {
        let bag = state.bag()?;
        ensure!(
            record.is_dead(bag)?,
            "state-death transition lacks an exhausted forced piece"
        );
        self.put_node(state, record)?;
        self.metrics.dead_filter.bits_set += self.dead_filter.insert(state);
        self.metrics.dead_filter.inserts += 1;
        self.metrics.nodes_dead += 1;
        self.output(Command::DrainDead { succ: state })?;

        for piece in bag.iter_pieces() {
            let orientation_index = match record.slot(piece)? {
                MoveSlot::Active(index) | MoveSlot::Pending(index) => index,
                MoveSlot::Unexpanded | MoveSlot::Exhausted => continue,
            };
            let orientation = validate_orientation(piece, orientation_index)?;
            let succ = successor(state, piece, orientation)?
                .context("persisted move leaves the admissibility carrier")?;
            self.output(Command::DetachEdge {
                succ,
                pred: state,
                piece,
                orientation,
            })?;
        }
        if state == seed_key()? {
            self.seed_died = true;
        }
        Ok(())
    }

    fn detach_edge(
        &mut self,
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    ) -> Result<()> {
        let edge = ReverseEdgeKey::new(succ, pred, piece, orientation)?;
        self.set_redge(edge, false)?;
        self.metrics.reverse_edges.detached += 1;
        self.estimated_bytes += REDGE_KEY_LEN;
        Ok(())
    }

    fn set_redge(&mut self, edge: ReverseEdgeKey, present: bool) -> Result<()> {
        ensure!(
            self.scratch.redges.contains_key(&edge)
                || self.scratch.redges.len() < MAX_REDGE_MUTATIONS_PER_BATCH,
            "reverse-edge overlay exceeds fixed batch capacity"
        );
        self.scratch.redges.insert(edge, present);
        Ok(())
    }

    fn drain_dead(&mut self, succ: StateKey) -> Result<()> {
        let bag = succ.bag()?;
        let record = self
            .load_node(succ)?
            .context("DrainDead referenced an absent successor")?;
        ensure!(record.is_dead(bag)?, "DrainDead successor is not dead");

        let has_more = self.prepare_effective_reverse_edges(succ)?;
        let edge_count = self.scratch.drain_edges.len().min(MAX_REDGES_PER_DRAIN);
        for index in 0..edge_count {
            let edge = self.scratch.drain_edges[index];
            let (_, pred, piece, orientation) = edge.decode()?;
            self.set_redge(edge, false)?;
            self.metrics.reverse_edges.drained += 1;
            self.estimated_bytes += REDGE_KEY_LEN;
            self.output(Command::ChildDead {
                succ,
                pred,
                piece,
                orientation,
            })?;
        }
        if has_more {
            self.metrics.reverse_edges.drain_continuations += 1;
            self.output(Command::DrainDead { succ })?;
        }
        Ok(())
    }

    fn prepare_effective_reverse_edges(&mut self, succ: StateKey) -> Result<bool> {
        self.scratch.ordered_redges.clear();
        for (&edge, &present) in &self.scratch.redges {
            if edge.as_bytes().starts_with(succ.as_bytes()) && present {
                ensure!(
                    self.scratch.ordered_redges.len() < MAX_REDGE_MUTATIONS_PER_BATCH,
                    "ordered reverse-edge scratch exceeds fixed capacity"
                );
                self.scratch.ordered_redges.push(edge);
            }
        }
        self.scratch.ordered_redges.sort_unstable();
        self.scratch.ordered_redges.dedup();
        self.scratch.drain_edges.clear();

        let cf = cf(&self.db, CF_REDGE)?;
        let mut options = ReadOptions::default();
        options.fill_cache(false);
        options.set_prefix_same_as_start(true);
        let mut iterator = self.db.raw_iterator_cf_opt(&cf, options);
        iterator.seek(succ.as_bytes());
        let mut overlay_index = 0;

        while self.scratch.drain_edges.len() <= MAX_REDGES_PER_DRAIN {
            let db_edge = loop {
                let Some(key_bytes) = iterator.key() else {
                    iterator.status().context("scan reverse edges")?;
                    break None;
                };
                if !key_bytes.starts_with(succ.as_bytes()) {
                    break None;
                }
                let edge = ReverseEdgeKey::from_slice(key_bytes)?;
                if self.scratch.redges.get(&edge) == Some(&false) {
                    iterator.next();
                    continue;
                }
                break Some(edge);
            };
            let overlay_edge = self.scratch.ordered_redges.get(overlay_index).copied();

            let next = match (db_edge, overlay_edge) {
                (Some(db_edge), Some(overlay_edge)) => match db_edge.cmp(&overlay_edge) {
                    Ordering::Less => {
                        iterator.next();
                        db_edge
                    }
                    Ordering::Equal => {
                        iterator.next();
                        overlay_index += 1;
                        db_edge
                    }
                    Ordering::Greater => {
                        overlay_index += 1;
                        overlay_edge
                    }
                },
                (Some(db_edge), None) => {
                    iterator.next();
                    db_edge
                }
                (None, Some(overlay_edge)) => {
                    overlay_index += 1;
                    overlay_edge
                }
                (None, None) => break,
            };
            self.scratch.drain_edges.push(next);
        }

        Ok(self.scratch.drain_edges.len() > MAX_REDGES_PER_DRAIN)
    }

    fn closure_delta(&self) -> Result<ClosureGauges> {
        let mut delta = ClosureGauges::default();
        for state in self.scratch.dirty_nodes.to_slice() {
            let entry = self
                .scratch
                .nodes
                .get(state)
                .context("dirty node is absent from batch overlay")?;
            let bag = state.bag()?;
            let before = ClosureGauges::from_record(entry.original, bag)?;
            let after = ClosureGauges::from_record(entry.record, bag)?;
            delta.add(after.delta_from(before));
        }
        Ok(delta)
    }

    fn commit(
        &mut self,
        shared: &SharedState,
        write_options: &WriteOptions,
        claim_cursor: &WorkClaimCursor,
    ) -> Result<()> {
        let input_count = self.scratch.processed.len() as u64;
        let output_count = self.scratch.outputs.len() as u64;
        ensure!(
            CommandCounts::from_array(self.metrics.commands_processed).total() == input_count,
            "processed command metrics do not match batch inputs"
        );
        ensure!(
            CommandCounts::from_array(self.metrics.commands_produced).total() == output_count,
            "produced command metrics do not match batch outputs"
        );
        self.metrics.closure_delta = self.closure_delta()?;
        let mut outputs_by_owner = [0u64; VIRTUAL_SHARDS];
        for output in &self.scratch.outputs {
            outputs_by_owner[output.key.owner() as usize] += 1;
        }
        shared
            .stats
            .reserve_outputs(output_count, &outputs_by_owner)?;

        let node_cf = cf(&self.db, CF_NODE)?;
        let redge_cf = cf(&self.db, CF_REDGE)?;
        let work_cf = cf(&self.db, CF_WORK)?;
        let mut batch = WriteBatch::default();
        for state in self.scratch.dirty_nodes.to_slice() {
            let entry = self
                .scratch
                .nodes
                .get(state)
                .context("dirty node is absent from batch overlay")?;
            let record = entry.record.context("dirty node must contain a record")?;
            batch.put_cf(&node_cf, state.as_bytes(), record.as_bytes());
        }
        for (edge, present) in &self.scratch.redges {
            if *present {
                batch.put_cf(&redge_cf, edge.as_bytes(), []);
            } else {
                batch.delete_cf(&redge_cf, edge.as_bytes());
            }
        }
        let work_commit = claim_cursor.append_committed(
            &mut batch,
            &work_cf,
            self.scratch.processed.to_slice(),
        )?;
        self.metrics.work_heads_advanced = work_commit.heads_advanced;
        self.metrics.work_reclaim_ranges = work_commit.reclaim_ranges;
        for output in &self.scratch.outputs {
            batch.put_cf(&work_cf, output.key.as_bytes(), output.command.encode()?);
        }
        if output_count != 0 {
            batch.put(
                producer_sequence_meta_key(self.owner),
                (*self.next_sequence).to_be_bytes(),
            );
        }

        let commit_started = Instant::now();
        self.db
            .write_opt(batch, write_options)
            .context("commit owner batch")?;
        shared.stats.record_commit(
            self.owner as usize,
            &self.metrics,
            input_count,
            output_count,
            self.estimated_bytes as u64,
            duration_micros(commit_started.elapsed()),
        )?;
        if self.seed_died {
            shared.seed_dead.store(true, AtomicOrdering::SeqCst);
        }
        Ok(())
    }
}

struct SharedState {
    db: Arc<DB>,
    owners: OwnerMap,
    dead_filter: DeadStateBloom,
    stats: RuntimeStats,
    shutdown: Arc<ShutdownSignal>,
    converged: AtomicBool,
    seed_dead: AtomicBool,
    error: Mutex<Option<String>>,
}

#[derive(Default)]
struct ShutdownSignal {
    stop: AtomicBool,
    pause_requested: AtomicBool,
}

impl ShutdownSignal {
    fn request_pause(&self) {
        self.pause_requested.store(true, AtomicOrdering::SeqCst);
        self.stop.store(true, AtomicOrdering::SeqCst);
    }
}

impl SharedState {
    fn fail(&self, error: anyhow::Error) {
        if let Ok(mut slot) = self.error.lock() {
            if slot.is_none() {
                *slot = Some(format!("{error:#}"));
            }
        }
        self.shutdown.stop.store(true, AtomicOrdering::SeqCst);
    }
}

fn atomic_add_relaxed(counter: &AtomicU64, amount: u64) {
    if amount != 0 {
        counter.fetch_add(amount, AtomicOrdering::Relaxed);
    }
}

fn atomic_add_i64_relaxed(counter: &AtomicI64, amount: i64) {
    if amount != 0 {
        counter.fetch_add(amount, AtomicOrdering::Relaxed);
    }
}

fn nonnegative_atomic(counter: &AtomicI64) -> u64 {
    counter.load(AtomicOrdering::Relaxed).max(0) as u64
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn signed_delta(current: u64, previous: u64) -> i64 {
    if current >= previous {
        current.saturating_sub(previous).min(i64::MAX as u64) as i64
    } else {
        -(previous.saturating_sub(current).min(i64::MAX as u64) as i64)
    }
}

fn atomic_checked_add(counter: &AtomicU64, amount: u64) -> Result<()> {
    counter
        .fetch_update(AtomicOrdering::SeqCst, AtomicOrdering::SeqCst, |current| {
            current.checked_add(amount)
        })
        .map(|_| ())
        .map_err(|_| anyhow::anyhow!("in-flight counter overflow"))
}

fn atomic_checked_sub(counter: &AtomicU64, amount: u64) -> Result<()> {
    counter
        .fetch_update(AtomicOrdering::SeqCst, AtomicOrdering::SeqCst, |current| {
            current.checked_sub(amount)
        })
        .map(|_| ())
        .map_err(|_| anyhow::anyhow!("in-flight counter underflow"))
}

fn seed_key() -> Result<StateKey> {
    StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, TetrisPieceBagState::new())
}

fn claim_batch(
    db: &DB,
    owners: &OwnerMap,
    owner: u8,
    producer_start: usize,
    cursor: &WorkClaimCursor,
    claimed: &mut ClaimedBatch,
) -> Result<()> {
    claimed.clear();
    ensure!(
        producer_start < owners.writer_count,
        "producer start is outside the owner map"
    );
    let cf = cf(&db, CF_WORK)?;
    let mut options = ReadOptions::default();
    options.fill_cache(false);
    let mut iterator = db.raw_iterator_cf_opt(&cf, options);

    for priority in PRIORITY_DEATH..=PRIORITY_EXPAND {
        for (producer_begin, producer_end) in
            [(producer_start, owners.writer_count), (0, producer_start)]
        {
            if producer_begin == producer_end {
                continue;
            }
            let producer = producer_begin as u8;
            let seek_key =
                WorkKey::new(owner, priority, producer, cursor.next(priority, producer))?;
            iterator.seek(seek_key.as_bytes());
            while !claimed.is_full() {
                let Some((key_bytes, value)) = iterator.item() else {
                    iterator.status().context("scan work queue")?;
                    break;
                };
                let key = WorkKey::from_slice(key_bytes)?;
                let producer = key.producer() as usize;
                if key.owner() != owner
                    || key.priority() != priority
                    || producer < producer_begin
                    || producer >= producer_end
                {
                    break;
                }
                let lane_head = cursor.next(priority, key.producer());
                if key.sequence() < lane_head {
                    let seek_key = WorkKey::new(owner, priority, key.producer(), lane_head)?;
                    iterator.seek(seek_key.as_bytes());
                    continue;
                }
                let command = Command::decode(value)?;
                owners.validate_work(key, command)?;
                ensure!(
                    claimed.try_push(ClaimedCommand { key, command }),
                    "claimed work exceeds fixed batch capacity"
                );
                iterator.next();
            }
            if claimed.is_full() {
                break;
            }
        }
        if !claimed.is_empty() {
            claimed.as_mut_slice().sort_unstable_by_key(|item| item.key);
            return Ok(());
        }
    }
    Ok(())
}

fn writer_loop(
    shared: Arc<SharedState>,
    owner: u8,
    mut next_sequence: u64,
    mut claim_cursor: WorkClaimCursor,
) -> Result<()> {
    let mut producer_start = owner as usize;
    let mut claimed = ClaimedBatch::new();
    let mut scratch = WorkerScratch::new()?;
    let mut write_options = WriteOptions::default();
    write_options.disable_wal(true);

    while !shared.shutdown.stop.load(AtomicOrdering::SeqCst) {
        let iteration_started = Instant::now();
        if shared.seed_dead.load(AtomicOrdering::SeqCst) {
            shared.shutdown.stop.store(true, AtomicOrdering::SeqCst);
            break;
        }
        let claim_started = Instant::now();
        claim_batch(
            &shared.db,
            &shared.owners,
            owner,
            producer_start,
            &claim_cursor,
            &mut claimed,
        )?;
        shared
            .stats
            .record_claim(claimed.len() as u64, claim_started.elapsed());
        producer_start = (producer_start + 1) % shared.owners.writer_count;
        if claimed.is_empty() {
            if shared.stats.outstanding_work.load(AtomicOrdering::SeqCst) == 0 {
                shared.converged.store(true, AtomicOrdering::SeqCst);
                shared.shutdown.stop.store(true, AtomicOrdering::SeqCst);
                break;
            }
            thread::sleep(PARK_DURATION);
            shared
                .stats
                .record_idle(owner as usize, iteration_started.elapsed());
            continue;
        }

        let mut processor = BatchProcessor::new(
            &shared.db,
            &shared.owners,
            &shared.dead_filter,
            owner,
            &mut next_sequence,
            &mut scratch,
        )?;
        let process_started = Instant::now();
        processor.process_claimed(claimed.to_slice())?;
        shared.stats.record_process(process_started.elapsed());
        processor.commit(&shared, &write_options, &claim_cursor)?;
        claim_cursor.advance_committed(processor.scratch.processed.to_slice())?;
        shared
            .stats
            .record_busy(owner as usize, iteration_started.elapsed());
    }
    Ok(())
}

fn record_source_commit() -> String {
    let commit = process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|commit| commit.trim().to_owned())
        .filter(|commit| !commit.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    let dirty = process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    }
}

struct DbMemory {
    cache: Cache,
    write_buffer_manager: WriteBufferManager,
    configured_bytes: usize,
    application_reserve_bytes: usize,
    block_cache_capacity_bytes: usize,
    memtable_capacity_bytes: usize,
}

struct DbTelemetry {
    memory: DbMemory,
    statistics: Options,
}

struct OpenedDb {
    db: DB,
    telemetry: DbTelemetry,
}

fn application_memory_reserve(writer_count: usize) -> Result<usize> {
    let writers = writer_count
        .checked_mul(PER_WRITER_MEMORY_RESERVE_BYTES)
        .context("writer memory reserve overflow")?;
    writers
        .checked_add(NATIVE_OVERHEAD_BYTES)
        .and_then(|bytes| bytes.checked_add(DEAD_STATE_BLOOM_BYTES))
        .context("application RAM budget overflow")
}

const fn split_rocks_memory(rocks_bytes: usize) -> (usize, usize) {
    let cache_bytes = rocks_bytes / 3 * 2 + (rocks_bytes % 3) * 2 / 3;
    (cache_bytes, rocks_bytes - cache_bytes)
}

fn memory_budget(ram_mib: usize, writer_count: usize) -> Result<DbMemory> {
    let total = ram_mib
        .checked_mul(MIB_BYTES)
        .context("--ram-mib overflow")?;
    let reserved = application_memory_reserve(writer_count)?;
    let required = reserved
        .checked_add(32 * MIB_BYTES)
        .context("minimum RAM budget overflow")?;
    ensure!(
        total >= required,
        "--ram-mib {ram_mib} is too small for {writer_count} writers; need at least {} MiB",
        required.div_ceil(MIB_BYTES)
    );
    let rocks = total - reserved;
    let (cache_bytes, memtable_bytes) = split_rocks_memory(rocks);
    let cache = Cache::new_lru_cache(cache_bytes);
    let write_buffer_manager = WriteBufferManager::new_write_buffer_manager_with_cache(
        memtable_bytes,
        true,
        cache.clone(),
    );
    Ok(DbMemory {
        cache,
        write_buffer_manager,
        configured_bytes: total,
        application_reserve_bytes: reserved,
        block_cache_capacity_bytes: cache_bytes,
        memtable_capacity_bytes: memtable_bytes,
    })
}

fn bounded_block_options(memory: &DbMemory) -> BlockBasedOptions {
    let mut blocks = BlockBasedOptions::default();
    blocks.set_block_cache(&memory.cache);
    blocks.set_cache_index_and_filter_blocks(true);
    blocks.set_pin_l0_filter_and_index_blocks_in_cache(false);
    blocks.set_pin_top_level_index_and_filter(false);
    blocks.set_optimize_filters_for_memory(true);
    blocks.set_index_type(BlockBasedIndexType::TwoLevelIndexSearch);
    blocks.set_partition_filters(true);
    blocks
}

fn column_family_options(memory: &DbMemory) -> Vec<ColumnFamilyDescriptor> {
    let mut node = Options::default();
    node.set_write_buffer_manager(&memory.write_buffer_manager);
    node.set_compaction_style(DBCompactionStyle::Level);
    let mut node_blocks = bounded_block_options(memory);
    node_blocks.set_bloom_filter(10.0, false);
    node.set_block_based_table_factory(&node_blocks);

    let mut redge = Options::default();
    redge.set_write_buffer_manager(&memory.write_buffer_manager);
    redge.set_compaction_style(DBCompactionStyle::Level);
    redge.set_prefix_extractor(SliceTransform::create_fixed_prefix(STATE_KEY_LEN));
    let mut redge_blocks = bounded_block_options(memory);
    redge_blocks.set_bloom_filter(10.0, true);
    redge.set_block_based_table_factory(&redge_blocks);

    let mut work = Options::default();
    work.set_write_buffer_manager(&memory.write_buffer_manager);
    work.set_compaction_style(DBCompactionStyle::Universal);
    let work_blocks = bounded_block_options(memory);
    work.set_block_based_table_factory(&work_blocks);

    vec![
        ColumnFamilyDescriptor::new(CF_NODE, node),
        ColumnFamilyDescriptor::new(CF_REDGE, redge),
        ColumnFamilyDescriptor::new(CF_WORK, work),
    ]
}

fn open_db_with_telemetry(
    path: &PathBuf,
    ram_mib: usize,
    writer_count: usize,
    create: bool,
) -> Result<OpenedDb> {
    let memory = memory_budget(ram_mib, writer_count)?;
    let mut options = Options::default();
    options.create_if_missing(create);
    options.create_missing_column_families(create);
    options.set_max_open_files(MAX_OPEN_FILES);
    options.set_max_background_jobs(MAX_BACKGROUND_JOBS);
    options.set_max_subcompactions(MAX_SUBCOMPACTIONS);
    options.set_allow_concurrent_memtable_write(true);
    options.set_enable_write_thread_adaptive_yield(true);
    options.set_write_buffer_manager(&memory.write_buffer_manager);
    options.enable_statistics();
    options.set_statistics_level(StatsLevel::ExceptHistogramOrTimers);
    let db = DB::open_cf_descriptors(&options, path, column_family_options(&memory))
        .with_context(|| format!("open RocksDB {}", path.display()))?;
    Ok(OpenedDb {
        db,
        telemetry: DbTelemetry {
            memory,
            statistics: options,
        },
    })
}

fn open_db(path: &PathBuf, ram_mib: usize, writer_count: usize, create: bool) -> Result<DB> {
    Ok(open_db_with_telemetry(path, ram_mib, writer_count, create)?.db)
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the provided rusage when it returns success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    // SAFETY: the successful getrusage call initialized usage.
    let max_rss = unsafe { usage.assume_init() }.ru_maxrss.max(0) as u64;
    if cfg!(target_os = "macos") {
        max_rss
    } else {
        max_rss.saturating_mul(1024)
    }
}

#[cfg(not(unix))]
const fn peak_rss_bytes() -> u64 {
    0
}

fn cf_property(db: &DB, cf_name: &str, property: &str) -> Result<u64> {
    Ok(db
        .property_int_value_cf(&cf(db, cf_name)?, property)
        .with_context(|| format!("read RocksDB property {property} for {cf_name}"))?
        .unwrap_or(0))
}

fn sum_cf_property(db: &DB, property: &str) -> Result<u64> {
    let mut total = 0u64;
    for cf_name in [CF_NODE, CF_REDGE, CF_WORK] {
        total = total.saturating_add(cf_property(db, cf_name, property)?);
    }
    Ok(total)
}

fn max_cf_property(db: &DB, property: &str) -> Result<u64> {
    let mut maximum = 0;
    for cf_name in [CF_NODE, CF_REDGE, CF_WORK] {
        maximum = maximum.max(cf_property(db, cf_name, property)?);
    }
    Ok(maximum)
}

fn statistics_ticker(statistics: &str, ticker: &str) -> u64 {
    statistics
        .lines()
        .find_map(|line| {
            let suffix = line.strip_prefix(ticker)?.trim_start();
            let count = suffix.strip_prefix("COUNT :")?.trim_start();
            count.split_whitespace().next()?.parse().ok()
        })
        .unwrap_or(0)
}

fn sample_storage(db: &DB, telemetry: &DbTelemetry) -> Result<StorageStatsSnapshot> {
    let statistics = telemetry.statistics.get_statistics().unwrap_or_default();
    let block_cache_usage_bytes = telemetry.memory.cache.get_usage() as u64;
    let memtable_usage_bytes = telemetry.memory.write_buffer_manager.get_usage() as u64;
    let node_table_reader_bytes = cf_property(db, CF_NODE, "rocksdb.estimate-table-readers-mem")?;
    let reverse_edge_table_reader_bytes =
        cf_property(db, CF_REDGE, "rocksdb.estimate-table-readers-mem")?;
    let work_table_reader_bytes = cf_property(db, CF_WORK, "rocksdb.estimate-table-readers-mem")?;
    Ok(StorageStatsSnapshot {
        peak_rss_bytes: peak_rss_bytes(),
        block_cache_usage_bytes,
        block_cache_pinned_bytes: telemetry.memory.cache.get_pinned_usage() as u64,
        memtable_usage_bytes,
        pending_compaction_bytes: sum_cf_property(db, "rocksdb.estimate-pending-compaction-bytes")?,
        live_data_bytes: sum_cf_property(db, "rocksdb.estimate-live-data-size")?,
        live_sst_bytes: sum_cf_property(db, "rocksdb.live-sst-files-size")?,
        node_keys_estimate: cf_property(db, CF_NODE, "rocksdb.estimate-num-keys")?,
        reverse_edge_keys_estimate: cf_property(db, CF_REDGE, "rocksdb.estimate-num-keys")?,
        work_keys_estimate: cf_property(db, CF_WORK, "rocksdb.estimate-num-keys")?,
        node_table_reader_bytes,
        reverse_edge_table_reader_bytes,
        work_table_reader_bytes,
        level_zero_files: sum_cf_property(db, "rocksdb.num-files-at-level0")?,
        running_compactions: sum_cf_property(db, "rocksdb.num-running-compactions")?,
        running_flushes: sum_cf_property(db, "rocksdb.num-running-flushes")?,
        background_errors: sum_cf_property(db, "rocksdb.background-errors")?,
        delayed_write_rate: max_cf_property(db, "rocksdb.actual-delayed-write-rate")?,
        write_stopped: max_cf_property(db, "rocksdb.is-write-stopped")?,
        block_cache_hits: statistics_ticker(&statistics, "rocksdb.block.cache.hit"),
        block_cache_misses: statistics_ticker(&statistics, "rocksdb.block.cache.miss"),
        block_cache_data_hits: statistics_ticker(&statistics, "rocksdb.block.cache.data.hit"),
        block_cache_data_misses: statistics_ticker(&statistics, "rocksdb.block.cache.data.miss"),
        block_cache_index_hits: statistics_ticker(&statistics, "rocksdb.block.cache.index.hit"),
        block_cache_index_misses: statistics_ticker(&statistics, "rocksdb.block.cache.index.miss"),
        block_cache_filter_hits: statistics_ticker(&statistics, "rocksdb.block.cache.filter.hit"),
        block_cache_filter_misses: statistics_ticker(
            &statistics,
            "rocksdb.block.cache.filter.miss",
        ),
        bloom_filter_useful: statistics_ticker(&statistics, "rocksdb.bloom.filter.useful"),
        bloom_filter_full_positive: statistics_ticker(
            &statistics,
            "rocksdb.bloom.filter.full.positive",
        ),
        bloom_filter_full_true_positive: statistics_ticker(
            &statistics,
            "rocksdb.bloom.filter.full.true.positive",
        ),
        memtable_hits: statistics_ticker(&statistics, "rocksdb.memtable.hit"),
        memtable_misses: statistics_ticker(&statistics, "rocksdb.memtable.miss"),
        get_hits_l0: statistics_ticker(&statistics, "rocksdb.l0.hit"),
        get_hits_l1: statistics_ticker(&statistics, "rocksdb.l1.hit"),
        get_hits_l2_and_up: statistics_ticker(&statistics, "rocksdb.l2andup.hit"),
        db_seeks: statistics_ticker(&statistics, "rocksdb.number.db.seek"),
        db_nexts: statistics_ticker(&statistics, "rocksdb.number.db.next"),
        iterator_bytes_read: statistics_ticker(&statistics, "rocksdb.db.iter.bytes.read"),
        keys_read: statistics_ticker(&statistics, "rocksdb.number.keys.read"),
        bytes_read: statistics_ticker(&statistics, "rocksdb.bytes.read"),
        stall_micros: statistics_ticker(&statistics, "rocksdb.stall.micros"),
        bytes_written: statistics_ticker(&statistics, "rocksdb.bytes.written"),
        compaction_read_bytes: statistics_ticker(&statistics, "rocksdb.compact.read.bytes"),
        compaction_write_bytes: statistics_ticker(&statistics, "rocksdb.compact.write.bytes"),
        flush_write_bytes: statistics_ticker(&statistics, "rocksdb.flush.write.bytes"),
    })
}

fn is_fresh_database_path(path: &PathBuf) -> Result<bool> {
    if !path.exists() {
        return Ok(true);
    }
    ensure!(path.is_dir(), "{} is not a directory", path.display());
    Ok(fs::read_dir(path)
        .with_context(|| format!("read {}", path.display()))?
        .next()
        .is_none())
}

fn append_work_heads(
    batch: &mut WriteBatch,
    owners: &OwnerMap,
    work_heads: &[WorkClaimCursor],
) -> Result<()> {
    ensure!(
        work_heads.len() == owners.writer_count,
        "work-head owner count does not match writer count"
    );
    for (owner, heads) in work_heads.iter().enumerate() {
        for priority in PRIORITY_DEATH..=PRIORITY_EXPAND {
            for producer in 0..owners.writer_count {
                batch.put(
                    work_head_meta_key(owner as u8, priority, producer as u8),
                    heads.next(priority, producer as u8).to_be_bytes(),
                );
            }
        }
    }
    Ok(())
}

fn read_work_heads(db: &DB, owners: &OwnerMap) -> Result<Option<Vec<WorkClaimCursor>>> {
    let mut work_heads = vec![WorkClaimCursor::new(); owners.writer_count];
    let mut present = 0usize;
    for (owner, heads) in work_heads.iter_mut().enumerate() {
        for priority in PRIORITY_DEATH..=PRIORITY_EXPAND {
            for producer in 0..owners.writer_count {
                let key = work_head_meta_key(owner as u8, priority, producer as u8);
                if let Some(value) = db.get(key).context("read work-head metadata")? {
                    heads.set(
                        priority,
                        producer as u8,
                        decode_u64_metadata(&value, "work head")?,
                    );
                    present += 1;
                }
            }
        }
    }
    let expected = owners.writer_count * WORK_PRIORITY_COUNT * owners.writer_count;
    ensure!(
        present == 0 || present == expected,
        "work-head metadata is partially populated: found {present} of {expected} lanes"
    );
    Ok((present == expected).then_some(work_heads))
}

fn initialize_database(db: &DB, owners: &OwnerMap, source_commit: &str) -> Result<(StateKey, u8)> {
    let seed = seed_key()?;
    let seed_owner = owners.owner(seed);
    let command = Command::Expand { state: seed };
    let work_key = WorkKey::new(seed_owner, command.priority(), seed_owner, 0)?;
    owners.validate_work(work_key, command)?;
    let node_cf = cf(&db, CF_NODE)?;
    let work_cf = cf(&db, CF_WORK)?;
    let mut batch = WriteBatch::default();
    batch.put(META_SCHEMA_VERSION, SCHEMA_VERSION);
    batch.put(META_STATUS, STATUS_BUILDING);
    batch.put(META_OWNER_MAP, owners.as_bytes());
    batch.put(META_CARRIER, carrier_description().as_bytes());
    batch.put(META_SCORER, b"height_mse_board_score");
    batch.put(META_SOURCE_COMMIT, source_commit.as_bytes());
    for producer in 0..owners.writer_count {
        let next_sequence = u64::from(producer as u8 == seed_owner);
        batch.put(
            producer_sequence_meta_key(producer as u8),
            next_sequence.to_be_bytes(),
        );
    }
    append_work_heads(
        &mut batch,
        owners,
        &vec![WorkClaimCursor::new(); owners.writer_count],
    )?;
    batch.put_cf(&node_cf, seed.as_bytes(), NodeRecord::INIT.as_bytes());
    batch.put_cf(&work_cf, work_key.as_bytes(), command.encode()?);
    let mut options = WriteOptions::default();
    options.disable_wal(true);
    db.write_opt(batch, &options)
        .context("initialize atlas database")?;
    Ok((seed, seed_owner))
}

struct ExploreMetadata {
    owners: OwnerMap,
    previous_status: Vec<u8>,
    created_source_commit: String,
}

struct RecoverySnapshot {
    outstanding_work: u64,
    owner_outstanding: Vec<u64>,
    commands_carried_in: [u64; COMMAND_KIND_COUNT],
    nodes_created: u64,
    nodes_dead: u64,
    dead_filter_bits_set: u64,
    owner_closure: Vec<ClosureGauges>,
    next_sequences: Vec<u64>,
    work_heads: Vec<WorkClaimCursor>,
}

fn read_database_metadata(path: &PathBuf, ram_mib: usize) -> Result<ExploreMetadata> {
    ensure!(path.is_dir(), "database directory does not exist");
    let db = open_db(path, ram_mib, 1, false)?;
    ensure!(
        read_required_meta(&db, META_SCHEMA_VERSION)? == SCHEMA_VERSION,
        "unsupported atlas schema"
    );
    ensure!(
        read_required_meta(&db, META_CARRIER)? == carrier_description().as_bytes(),
        "atlas carrier does not match this binary"
    );
    ensure!(
        read_required_meta(&db, META_SCORER)? == b"height_mse_board_score",
        "atlas scorer does not match this binary"
    );
    let previous_status = read_required_meta(&db, META_STATUS)?;
    let owners = OwnerMap::from_bytes(&read_required_meta(&db, META_OWNER_MAP)?)?;
    let created_source_commit = String::from_utf8(read_required_meta(&db, META_SOURCE_COMMIT)?)
        .context("source-commit metadata is not UTF-8")?;
    Ok(ExploreMetadata {
        owners,
        previous_status,
        created_source_commit,
    })
}

fn recover_runtime(
    db: &DB,
    owners: &OwnerMap,
    dead_filter: &DeadStateBloom,
) -> Result<RecoverySnapshot> {
    let work_heads = read_work_heads(db, owners)?.context("missing work-head metadata")?;
    let mut recovery = RecoverySnapshot {
        outstanding_work: 0,
        owner_outstanding: vec![0; owners.writer_count],
        commands_carried_in: [0; COMMAND_KIND_COUNT],
        nodes_created: 0,
        nodes_dead: 0,
        dead_filter_bits_set: 0,
        owner_closure: vec![ClosureGauges::default(); owners.writer_count],
        next_sequences: vec![0; owners.writer_count],
        work_heads,
    };

    let work_cf = cf(&db, CF_WORK)?;
    let mut read_options = ReadOptions::default();
    read_options.fill_cache(false);
    for item in db.iterator_cf_opt(&work_cf, read_options, IteratorMode::Start) {
        let (key_bytes, value) = item.context("scan work queue for recovery")?;
        let key = WorkKey::from_slice(&key_bytes)?;
        ensure!(
            (key.owner() as usize) < owners.writer_count,
            "work key owner is outside the owner map"
        );
        ensure!(
            (key.producer() as usize) < owners.writer_count,
            "work key producer is outside the owner map"
        );
        let next_sequence = key
            .sequence()
            .checked_add(1)
            .context("recovered producer sequence overflow")?;
        recovery.next_sequences[key.producer() as usize] =
            recovery.next_sequences[key.producer() as usize].max(next_sequence);
        let lane_head =
            recovery.work_heads[key.owner() as usize].next(key.priority(), key.producer());
        if key.sequence() < lane_head {
            continue;
        }
        let command = Command::decode(&value)?;
        owners.validate_work(key, command)?;
        recovery.outstanding_work = recovery
            .outstanding_work
            .checked_add(1)
            .context("recovered work count overflow")?;
        let owner_outstanding = &mut recovery.owner_outstanding[key.owner() as usize];
        *owner_outstanding = owner_outstanding
            .checked_add(1)
            .context("recovered owner work count overflow")?;
        let command_count = &mut recovery.commands_carried_in[command.kind_index()];
        *command_count = command_count
            .checked_add(1)
            .context("recovered command count overflow")?;
    }

    for producer in 0..owners.writer_count {
        if let Some(value) = db
            .get(producer_sequence_meta_key(producer as u8))
            .context("read producer sequence metadata")?
        {
            recovery.next_sequences[producer] = recovery.next_sequences[producer]
                .max(decode_u64_metadata(&value, "producer sequence")?);
        }
    }
    for heads in &recovery.work_heads {
        for priority in PRIORITY_DEATH..=PRIORITY_EXPAND {
            for producer in 0..owners.writer_count {
                ensure!(
                    heads.next(priority, producer as u8) <= recovery.next_sequences[producer],
                    "work head exceeds producer sequence"
                );
            }
        }
    }

    let node_cf = cf(&db, CF_NODE)?;
    let mut read_options = ReadOptions::default();
    read_options.fill_cache(false);
    for item in db.iterator_cf_opt(&node_cf, read_options, IteratorMode::Start) {
        let (key_bytes, value) = item.context("scan node table for recovery")?;
        let state = StateKey::from_slice(&key_bytes)?;
        let (_, bag) = state.decode()?;
        let record = NodeRecord::from_slice(&value, bag)?;
        recovery.nodes_created = recovery
            .nodes_created
            .checked_add(1)
            .context("recovered node count overflow")?;
        if record.is_dead(bag)? {
            recovery.nodes_dead = recovery
                .nodes_dead
                .checked_add(1)
                .context("recovered dead-node count overflow")?;
            recovery.dead_filter_bits_set = recovery
                .dead_filter_bits_set
                .checked_add(dead_filter.insert(state))
                .context("recovered dead-state Bloom bit count overflow")?;
        } else {
            recovery.owner_closure[owners.owner(state) as usize]
                .checked_add(ClosureGauges::from_record(Some(record), bag)?)?;
        }
    }

    ensure!(recovery.nodes_created != 0, "node table is empty");
    ensure!(
        recovery.owner_outstanding.iter().copied().sum::<u64>() == recovery.outstanding_work,
        "recovered owner work counts do not match total work"
    );
    ensure!(
        CommandCounts::from_array(recovery.commands_carried_in).total()
            == recovery.outstanding_work,
        "recovered command counts do not match total work"
    );
    Ok(recovery)
}

fn mark_exploring(
    db: &DB,
    owners: &OwnerMap,
    next_sequences: &[u64],
    work_heads: &[WorkClaimCursor],
) -> Result<()> {
    let mut batch = WriteBatch::default();
    batch.put(META_SCHEMA_VERSION, SCHEMA_VERSION);
    batch.put(META_STATUS, STATUS_BUILDING);
    for (producer, &next_sequence) in next_sequences.iter().enumerate() {
        batch.put(
            producer_sequence_meta_key(producer as u8),
            next_sequence.to_be_bytes(),
        );
    }
    append_work_heads(&mut batch, owners, work_heads)?;
    let mut options = WriteOptions::default();
    options.disable_wal(true);
    db.write_opt(batch, &options)
        .context("mark atlas database as building")?;
    db.flush().context("flush building status")
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ValidationSummary {
    total_states: u64,
    dead_states: u64,
    surviving_states: u64,
    active_moves: u64,
}

#[derive(Serialize)]
struct RunStartLog<'a> {
    event: &'static str,
    mode: &'static str,
    allocator: &'static str,
    db_path: &'a str,
    writers: usize,
    ram_mib: usize,
    configured_memory_bytes: usize,
    application_memory_reserve_bytes: usize,
    block_cache_capacity_bytes: usize,
    memtable_capacity_bytes: usize,
    per_writer_memory_reserve_bytes: usize,
    writer_stack_bytes: usize,
    max_open_files: i32,
    max_background_jobs: i32,
    max_subcompactions: u32,
    cache_index_and_filter_blocks: bool,
    partitioned_index_and_filters: bool,
    optimize_filters_for_memory: bool,
    dead_state_bloom_bytes: usize,
    dead_state_bloom_hashes: usize,
    dead_state_bloom_capacity_bits: u64,
    dead_state_bloom_set_bits: u64,
    work_head_lanes: usize,
    work_reclaim_sequence_quantum: u64,
    carrier: &'a str,
    schema: &'static str,
    source_commit: &'a str,
    created_source_commit: &'a str,
    previous_status: Option<&'a str>,
    hostname: &'a str,
    os: &'static str,
    arch: &'static str,
    command_line: &'a [String],
}

#[derive(Serialize)]
struct ProgressLog<'a> {
    event: &'static str,
    elapsed_seconds: f64,
    interval_seconds: f64,
    interval_commands: u64,
    commands_per_second: f64,
    interval_candidates: u64,
    candidates_per_second: f64,
    interval_nodes_created: u64,
    nodes_per_second: f64,
    closure_trend: ClosureTrendSnapshot,
    stats: &'a RuntimeStatsSnapshot,
}

#[derive(Clone, Debug, Serialize)]
struct ClosureTrendSnapshot {
    interval_expand_produced: u64,
    interval_expand_processed: u64,
    expansion_reproduction: Option<f64>,
    frontier_delta: i64,
    frontier_growth_per_second: f64,
    frontier_drain_eta_seconds: Option<f64>,
    interval_new_nodes: u64,
    interval_existing_node_hits: u64,
    novelty_rate: Option<f64>,
    realized_successor_reuse_rate: Option<f64>,
    interval_reuse_probes: u64,
    reuse_probes_per_second: f64,
    interval_chosen_likely_existing: u64,
    interval_chosen_definitely_new: u64,
    chosen_reuse_hint_rate: Option<f64>,
    interval_nodes_dead: u64,
    deaths_per_second: f64,
    interval_reroutes: u64,
    reroutes_per_second: f64,
    death_cascade_factor: Option<f64>,
    locally_resolved_delta: i64,
    locally_resolved_per_second: f64,
    active_slots_delta: i64,
    active_slots_per_second: f64,
}

impl ClosureTrendSnapshot {
    fn between(
        previous: &RuntimeStatsSnapshot,
        current: &RuntimeStatsSnapshot,
        interval_seconds: f64,
    ) -> Self {
        let interval_expand_produced = current
            .commands
            .produced
            .expand
            .saturating_sub(previous.commands.produced.expand);
        let interval_expand_processed = current
            .commands
            .processed
            .expand
            .saturating_sub(previous.commands.processed.expand);
        let frontier_delta = signed_delta(
            current.closure.expand_backlog,
            previous.closure.expand_backlog,
        );
        let frontier_growth_per_second = frontier_delta as f64 / interval_seconds;
        let frontier_drain_eta_seconds = (frontier_growth_per_second < 0.0)
            .then(|| current.closure.expand_backlog as f64 / -frontier_growth_per_second);
        let interval_new_nodes = current.nodes_created.saturating_sub(previous.nodes_created);
        let interval_attach_edges = current
            .commands
            .processed
            .attach_edge
            .saturating_sub(previous.commands.processed.attach_edge);
        let interval_existing_node_hits = interval_attach_edges.saturating_sub(interval_new_nodes);
        let interval_reuse_probes = current
            .candidates
            .reuse_probes
            .saturating_sub(previous.candidates.reuse_probes);
        let interval_chosen_likely_existing = current
            .candidates
            .chosen_likely_existing
            .saturating_sub(previous.candidates.chosen_likely_existing);
        let interval_chosen_definitely_new = current
            .candidates
            .chosen_definitely_new
            .saturating_sub(previous.candidates.chosen_definitely_new);
        let interval_nodes_dead = current.nodes_dead.saturating_sub(previous.nodes_dead);
        let interval_reroutes = current
            .closure
            .reroutes
            .saturating_sub(previous.closure.reroutes);
        let interval_death_notifications = current
            .commands
            .produced
            .child_dead
            .saturating_sub(previous.commands.produced.child_dead);
        let locally_resolved_delta = signed_delta(
            current.closure.locally_resolved_states,
            previous.closure.locally_resolved_states,
        );
        let active_slots_delta = signed_delta(
            current.closure.forced_slots_active,
            previous.closure.forced_slots_active,
        );
        Self {
            interval_expand_produced,
            interval_expand_processed,
            expansion_reproduction: ratio(interval_expand_produced, interval_expand_processed),
            frontier_delta,
            frontier_growth_per_second,
            frontier_drain_eta_seconds,
            interval_new_nodes,
            interval_existing_node_hits,
            novelty_rate: ratio(interval_new_nodes, interval_attach_edges),
            realized_successor_reuse_rate: ratio(
                interval_existing_node_hits,
                interval_attach_edges,
            ),
            interval_reuse_probes,
            reuse_probes_per_second: interval_reuse_probes as f64 / interval_seconds,
            interval_chosen_likely_existing,
            interval_chosen_definitely_new,
            chosen_reuse_hint_rate: ratio(
                interval_chosen_likely_existing,
                interval_chosen_likely_existing.saturating_add(interval_chosen_definitely_new),
            ),
            interval_nodes_dead,
            deaths_per_second: interval_nodes_dead as f64 / interval_seconds,
            interval_reroutes,
            reroutes_per_second: interval_reroutes as f64 / interval_seconds,
            death_cascade_factor: ratio(interval_death_notifications, interval_nodes_dead),
            locally_resolved_delta,
            locally_resolved_per_second: locally_resolved_delta as f64 / interval_seconds,
            active_slots_delta,
            active_slots_per_second: active_slots_delta as f64 / interval_seconds,
        }
    }
}

#[derive(Serialize)]
struct StorageLog {
    event: &'static str,
    elapsed_seconds: f64,
    storage: StorageStatsSnapshot,
}

#[derive(Serialize)]
struct TerminalLog<'a> {
    event: &'static str,
    status: &'a str,
    elapsed_seconds: f64,
    source_commit: &'a str,
    stats: &'a RuntimeStatsSnapshot,
    validation: Option<ValidationSummary>,
    error: Option<&'a str>,
}

fn emit_json<T: Serialize>(value: &T) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(value).context("encode metrics JSON")?
    );
    Ok(())
}

fn refresh_storage(db: &DB, telemetry: &DbTelemetry, stats: &RuntimeStats) -> StorageStatsSnapshot {
    match sample_storage(db, telemetry) {
        Ok(storage) => {
            stats.storage.store(&storage);
            storage
        }
        Err(error) => {
            eprintln!("storage-metrics-error: {error:#}");
            stats.storage.snapshot()
        }
    }
}

fn status_text(status: &[u8]) -> &str {
    std::str::from_utf8(status).unwrap_or("invalid-status")
}

fn validate_closure(db: &DB, owners: &OwnerMap) -> Result<ValidationSummary> {
    let node_cf = cf(&db, CF_NODE)?;
    let redge_cf = cf(&db, CF_REDGE)?;
    let work_heads = read_work_heads(db, owners)?.context("missing work-head metadata")?;
    for (owner, cursor) in work_heads.iter().enumerate() {
        let mut claimed = ClaimedBatch::new();
        claim_batch(db, owners, owner as u8, owner, cursor, &mut claimed)?;
        ensure!(claimed.is_empty(), "closure validation found pending work");
    }

    let mut summary = ValidationSummary {
        total_states: 0,
        dead_states: 0,
        surviving_states: 0,
        active_moves: 0,
    };
    for item in db.iterator_cf(&node_cf, IteratorMode::Start) {
        let (key_bytes, value) = item.context("scan node table")?;
        let state = StateKey::from_slice(&key_bytes)?;
        let (_, bag) = state.decode()?;
        let record = NodeRecord::from_slice(&value, bag)?;
        summary.total_states += 1;
        if record.is_dead(bag)? {
            summary.dead_states += 1;
            continue;
        }
        ensure!(
            record.is_resolved(bag)? && !record.has_pending(bag)?,
            "surviving state has unresolved forced slots"
        );
        summary.surviving_states += 1;
        for piece in bag.iter_pieces() {
            let MoveSlot::Active(orientation_index) = record.slot(piece)? else {
                bail!("surviving state lacks an active move for {piece:?}");
            };
            let orientation = validate_orientation(piece, orientation_index)?;
            let succ = successor(state, piece, orientation)?
                .context("active move leaves the admissibility carrier")?;
            let (_, succ_bag) = succ.decode()?;
            let succ_value = db
                .get_pinned_cf(&node_cf, succ.as_bytes())
                .context("read closure successor")?
                .context("active move references an absent successor")?;
            let succ_record = NodeRecord::from_slice(succ_value.as_ref(), succ_bag)?;
            ensure!(
                !succ_record.is_dead(succ_bag)?,
                "active move references a dead successor"
            );
            let edge = ReverseEdgeKey::new(succ, state, piece, orientation)?;
            ensure!(
                db.get_pinned_cf(&redge_cf, edge.as_bytes())
                    .context("read closure reverse edge")?
                    .is_some(),
                "active move lacks its reverse edge"
            );
            summary.active_moves += 1;
        }
    }

    let seed = seed_key()?;
    let (_, seed_bag) = seed.decode()?;
    let seed_value = db
        .get_pinned_cf(&node_cf, seed.as_bytes())
        .context("read seed during closure validation")?
        .context("node table lacks the seed")?;
    let seed_record = NodeRecord::from_slice(seed_value.as_ref(), seed_bag)?;
    ensure!(
        !seed_record.is_dead(seed_bag)? && seed_record.is_resolved(seed_bag)?,
        "seed is not a resolved surviving state"
    );
    Ok(summary)
}

fn flush_all(db: &DB) -> Result<()> {
    for name in [CF_NODE, CF_REDGE, CF_WORK] {
        db.flush_cf(&cf(db, name)?)
            .with_context(|| format!("flush {name} column family"))?;
    }
    db.flush().context("flush metadata")
}

fn store_final_status(db: &DB, status: &[u8], validation: Option<ValidationSummary>) -> Result<()> {
    let mut batch = WriteBatch::default();
    batch.put(META_STATUS, status);
    if let Some(validation) = validation {
        let summary = format!(
            "total_states={};dead_states={};surviving_states={};active_moves={}",
            validation.total_states,
            validation.dead_states,
            validation.surviving_states,
            validation.active_moves
        );
        batch.put(META_SUMMARY, summary.as_bytes());
    }
    let mut options = WriteOptions::default();
    options.disable_wal(true);
    db.write_opt(batch, &options)
        .context("write final database status")?;
    flush_all(db)
}

fn record_terminal_status(
    shared: &SharedState,
    telemetry: &DbTelemetry,
    status: &[u8],
    started: Instant,
    source_commit: &str,
    validation: Option<ValidationSummary>,
    error: Option<&str>,
) -> Result<()> {
    refresh_storage(&shared.db, telemetry, &shared.stats);
    let snapshot = shared.stats.snapshot();
    let result = store_final_status(&shared.db, status, validation);
    shared.db.cancel_all_background_work(true);
    result?;
    emit_json(&TerminalLog {
        event: "terminal",
        status: status_text(status),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        source_commit,
        stats: &snapshot,
        validation,
        error,
    })
}

fn pause_handler(shutdown: Arc<ShutdownSignal>) -> impl FnMut() + Send + 'static {
    move || shutdown.request_pause()
}

struct WorkerGroup {
    shared: Arc<SharedState>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl WorkerGroup {
    fn with_capacity(shared: Arc<SharedState>, capacity: usize) -> Self {
        Self {
            shared,
            handles: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, handle: thread::JoinHandle<()>) {
        self.handles.push(handle);
    }

    fn join_all(&mut self) {
        for handle in self.handles.drain(..) {
            if handle.join().is_err() {
                self.shared
                    .fail(anyhow::anyhow!("atlas owner thread panicked"));
            }
        }
    }
}

impl Drop for WorkerGroup {
    fn drop(&mut self) {
        self.shared
            .shutdown
            .stop
            .store(true, AtomicOrdering::SeqCst);
        self.join_all();
        self.shared.db.cancel_all_background_work(true);
    }
}

struct BuilderRun {
    path: PathBuf,
    ram_mib: usize,
    mode: &'static str,
    source_commit: String,
    created_source_commit: String,
    previous_status: Option<Vec<u8>>,
    opened_db: OpenedDb,
    owners: OwnerMap,
    dead_filter: DeadStateBloom,
    stats: RuntimeStats,
    next_sequences: Vec<u64>,
    work_heads: Vec<WorkClaimCursor>,
    mark_building: bool,
}

fn run_builder(run: BuilderRun) -> Result<()> {
    let BuilderRun {
        path,
        ram_mib,
        mode,
        source_commit,
        created_source_commit,
        previous_status,
        opened_db: OpenedDb { db, telemetry },
        owners,
        dead_filter,
        stats,
        next_sequences,
        work_heads,
        mark_building,
    } = run;
    let writer_count = owners.writer_count;
    ensure!(
        next_sequences.len() == writer_count,
        "producer sequence count does not match writer count"
    );
    ensure!(
        work_heads.len() == writer_count,
        "work-head owner count does not match writer count"
    );
    let shutdown = Arc::new(ShutdownSignal::default());
    let shared = Arc::new(SharedState {
        db: Arc::new(db),
        owners,
        dead_filter,
        stats,
        shutdown: Arc::clone(&shutdown),
        converged: AtomicBool::new(false),
        seed_dead: AtomicBool::new(false),
        error: Mutex::new(None),
    });
    let db_path = path.display().to_string();
    let hostname = gethostname::gethostname().to_string_lossy().into_owned();
    let command_line: Vec<_> = std::env::args().collect();
    let carrier = carrier_description();
    emit_json(&RunStartLog {
        event: "run_start",
        mode,
        allocator: "mimalloc",
        db_path: &db_path,
        writers: writer_count,
        ram_mib,
        configured_memory_bytes: telemetry.memory.configured_bytes,
        application_memory_reserve_bytes: telemetry.memory.application_reserve_bytes,
        block_cache_capacity_bytes: telemetry.memory.block_cache_capacity_bytes,
        memtable_capacity_bytes: telemetry.memory.memtable_capacity_bytes,
        per_writer_memory_reserve_bytes: PER_WRITER_MEMORY_RESERVE_BYTES,
        writer_stack_bytes: WRITER_STACK_BYTES,
        max_open_files: MAX_OPEN_FILES,
        max_background_jobs: MAX_BACKGROUND_JOBS,
        max_subcompactions: MAX_SUBCOMPACTIONS,
        cache_index_and_filter_blocks: true,
        partitioned_index_and_filters: true,
        optimize_filters_for_memory: true,
        dead_state_bloom_bytes: shared.dead_filter.allocated_bytes(),
        dead_state_bloom_hashes: DEAD_STATE_BLOOM_HASHES,
        dead_state_bloom_capacity_bits: shared.dead_filter.capacity_bits(),
        dead_state_bloom_set_bits: shared.dead_filter.set_bits(),
        work_head_lanes: writer_count * WORK_PRIORITY_COUNT * writer_count,
        work_reclaim_sequence_quantum: WORK_RECLAIM_SEQUENCE_QUANTUM,
        carrier: &carrier,
        schema: status_text(SCHEMA_VERSION),
        source_commit: &source_commit,
        created_source_commit: &created_source_commit,
        previous_status: previous_status.as_deref().map(status_text),
        hostname: &hostname,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        command_line: &command_line,
    })?;
    let mut previous_snapshot = shared.stats.snapshot();
    ctrlc::set_handler(pause_handler(shutdown)).context("install shutdown signal handler")?;
    if mark_building {
        mark_exploring(&shared.db, &shared.owners, &next_sequences, &work_heads)?;
    }

    let mut workers = WorkerGroup::with_capacity(Arc::clone(&shared), writer_count);
    for (owner, (initial_sequence, claim_cursor)) in
        next_sequences.into_iter().zip(work_heads).enumerate()
    {
        let worker = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(format!("atlas-owner-{owner}"))
            .stack_size(WRITER_STACK_BYTES)
            .spawn(move || {
                if let Err(error) =
                    writer_loop(worker.clone(), owner as u8, initial_sequence, claim_cursor)
                {
                    worker.fail(error);
                }
            })
            .with_context(|| format!("spawn atlas owner {owner}"))?;
        workers.push(handle);
    }

    let started = Instant::now();
    let mut last_log = Instant::now();
    let mut last_storage_log = Instant::now();
    while !shared.shutdown.stop.load(AtomicOrdering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
        if last_log.elapsed() >= LOG_INTERVAL {
            let snapshot = shared.stats.snapshot();
            let processed = snapshot.commands.processed.total();
            let interval_seconds = last_log.elapsed().as_secs_f64().max(f64::EPSILON);
            let interval_commands =
                processed.saturating_sub(previous_snapshot.commands.processed.total());
            let interval_candidates = snapshot
                .candidates
                .examined
                .saturating_sub(previous_snapshot.candidates.examined);
            let interval_nodes_created = snapshot
                .nodes_created
                .saturating_sub(previous_snapshot.nodes_created);
            let closure_trend =
                ClosureTrendSnapshot::between(&previous_snapshot, &snapshot, interval_seconds);
            if let Err(error) = emit_json(&ProgressLog {
                event: "progress",
                elapsed_seconds: started.elapsed().as_secs_f64(),
                interval_seconds,
                interval_commands,
                commands_per_second: interval_commands as f64 / interval_seconds,
                interval_candidates,
                candidates_per_second: interval_candidates as f64 / interval_seconds,
                interval_nodes_created,
                nodes_per_second: interval_nodes_created as f64 / interval_seconds,
                closure_trend,
                stats: &snapshot,
            }) {
                shared.fail(error);
                break;
            }
            previous_snapshot = snapshot;
            last_log = Instant::now();
        }
        if last_storage_log.elapsed() >= STORAGE_LOG_INTERVAL {
            let storage = refresh_storage(&shared.db, &telemetry, &shared.stats);
            if let Err(error) = emit_json(&StorageLog {
                event: "storage",
                elapsed_seconds: started.elapsed().as_secs_f64(),
                storage,
            }) {
                shared.fail(error);
                break;
            }
            last_storage_log = Instant::now();
        }
    }
    workers.join_all();

    let error = shared
        .error
        .lock()
        .map_err(|_| anyhow::anyhow!("worker error lock was poisoned"))?
        .clone();
    if let Some(error) = error {
        record_terminal_status(
            &shared,
            &telemetry,
            STATUS_ABORTED,
            started,
            &source_commit,
            None,
            Some(&error),
        )?;
        bail!("atlas build aborted: {error}");
    }
    if shared.seed_dead.load(AtomicOrdering::SeqCst) {
        record_terminal_status(
            &shared,
            &telemetry,
            STATUS_SEED_DEAD,
            started,
            &source_commit,
            None,
            None,
        )?;
        return Ok(());
    }
    if !shared.converged.load(AtomicOrdering::SeqCst) {
        let status = if shared.shutdown.pause_requested.load(AtomicOrdering::SeqCst) {
            STATUS_PAUSED
        } else {
            STATUS_ABORTED
        };
        record_terminal_status(
            &shared,
            &telemetry,
            status,
            started,
            &source_commit,
            None,
            None,
        )?;
        return Ok(());
    }

    if shared.stats.outstanding_work.load(AtomicOrdering::SeqCst) != 0 {
        let error = "converged with nonzero outstanding work";
        record_terminal_status(
            &shared,
            &telemetry,
            STATUS_ABORTED,
            started,
            &source_commit,
            None,
            Some(error),
        )?;
        bail!("{error}");
    }
    let summary = match validate_closure(&shared.db, &shared.owners) {
        Ok(summary) => summary,
        Err(error) => {
            let message = format!("{error:#}");
            record_terminal_status(
                &shared,
                &telemetry,
                STATUS_ABORTED,
                started,
                &source_commit,
                None,
                Some(&message),
            )?;
            return Err(error.context("closure validation failed"));
        }
    };
    record_terminal_status(
        &shared,
        &telemetry,
        STATUS_COMPLETE,
        started,
        &source_commit,
        Some(summary),
        None,
    )?;
    Ok(())
}

fn create_fresh(path: PathBuf, writers: Option<usize>, ram_mib: usize) -> Result<()> {
    let available = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let writer_count = writers.unwrap_or(available.min(4));
    let owners = OwnerMap::balanced(writer_count)?;
    let source_commit = record_source_commit();
    let opened_db = open_db_with_telemetry(&path, ram_mib, writer_count, true)?;
    let (_, seed_owner) = initialize_database(&opened_db.db, &owners, &source_commit)?;
    let dead_filter = DeadStateBloom::new(DEAD_STATE_BLOOM_BYTES)?;
    let next_sequences = (0..writer_count)
        .map(|owner| u64::from(owner as u8 == seed_owner))
        .collect();
    run_builder(BuilderRun {
        path,
        ram_mib,
        mode: "create",
        source_commit: source_commit.clone(),
        created_source_commit: source_commit,
        previous_status: None,
        opened_db,
        owners,
        dead_filter,
        stats: RuntimeStats::new(writer_count, seed_owner),
        next_sequences,
        work_heads: vec![WorkClaimCursor::new(); writer_count],
        mark_building: false,
    })
}

fn resume_create(
    path: PathBuf,
    writers: Option<usize>,
    ram_mib: usize,
    metadata: ExploreMetadata,
) -> Result<()> {
    let writer_count = metadata.owners.writer_count;
    if let Some(requested) = writers {
        ensure!(
            requested == writer_count,
            "database was created with {writer_count} writers; resume cannot use {requested}"
        );
    }
    let opened_db = open_db_with_telemetry(&path, ram_mib, writer_count, false)?;
    let dead_filter = DeadStateBloom::new(DEAD_STATE_BLOOM_BYTES)?;
    let recovery = recover_runtime(&opened_db.db, &metadata.owners, &dead_filter)?;
    let seed = seed_key()?;
    let (_, seed_bag) = seed.decode()?;
    let node_cf = cf(&opened_db.db, CF_NODE)?;
    let seed_value = opened_db
        .db
        .get_pinned_cf(&node_cf, seed.as_bytes())
        .context("read seed during recovery")?
        .context("node table lacks the seed")?;
    let seed_record = NodeRecord::from_slice(seed_value.as_ref(), seed_bag)?;
    ensure!(
        !seed_record.is_dead(seed_bag)?,
        "cannot continue a database whose seed is dead"
    );
    drop(seed_value);
    drop(node_cf);
    let source_commit = record_source_commit();
    let stats = RuntimeStats::recovered(&recovery);
    run_builder(BuilderRun {
        path,
        ram_mib,
        mode: "create",
        source_commit,
        created_source_commit: metadata.created_source_commit,
        previous_status: Some(metadata.previous_status),
        opened_db,
        owners: metadata.owners,
        dead_filter,
        stats,
        next_sequences: recovery.next_sequences,
        work_heads: recovery.work_heads,
        mark_building: true,
    })
}

fn run_create(path: PathBuf, writers: Option<usize>, ram_mib: usize) -> Result<()> {
    if is_fresh_database_path(&path)? {
        return create_fresh(path, writers, ram_mib);
    }

    let metadata = read_database_metadata(&path, ram_mib)?;
    match metadata.previous_status.as_slice() {
        STATUS_PAUSED | STATUS_ABORTED => resume_create(path, writers, ram_mib, metadata),
        STATUS_COMPLETE => {
            bail!("atlas is already complete; use explore to inspect it")
        }
        STATUS_BUILDING => {
            bail!("database status is building; only a clean Ctrl-C pause is resumable")
        }
        STATUS_SEED_DEAD => bail!("database status is seed-dead and cannot be resumed"),
        status => bail!("unsupported database status {}", status_text(status)),
    }
}

fn read_required_meta(db: &DB, key: &[u8]) -> Result<Vec<u8>> {
    db.get(key)
        .context("read atlas metadata")?
        .with_context(|| format!("missing metadata key {}", String::from_utf8_lossy(key)))
}

fn piece_from_char(value: char) -> Result<TetrisPiece> {
    match value.to_ascii_uppercase() {
        'O' => Ok(TetrisPiece::O_PIECE),
        'I' => Ok(TetrisPiece::I_PIECE),
        'S' => Ok(TetrisPiece::S_PIECE),
        'Z' => Ok(TetrisPiece::Z_PIECE),
        'T' => Ok(TetrisPiece::T_PIECE),
        'L' => Ok(TetrisPiece::L_PIECE),
        'J' => Ok(TetrisPiece::J_PIECE),
        other => bail!("invalid piece {other:?}; expected one of OISZTLJ"),
    }
}

fn print_state_moves(db: &DB, state: StateKey) -> Result<()> {
    let node_cf = cf(&db, CF_NODE)?;
    let (board, bag) = state.decode()?;
    let value = db
        .get_pinned_cf(&node_cf, state.as_bytes())
        .context("read atlas state")?
        .context("state is not present in the atlas")?;
    let record = NodeRecord::from_slice(value.as_ref(), bag)?;
    ensure!(!record.is_dead(bag)?, "state is dead");
    println!(
        "board_height={} holes={} bag={:#09b}",
        board.height(),
        board.total_holes(),
        u8::from(bag)
    );
    for piece in bag.iter_pieces() {
        let MoveSlot::Active(index) = record.slot(piece)? else {
            bail!("completed atlas state has no active move for {piece:?}");
        };
        println!("{piece:?}: {}", validate_orientation(piece, index)?);
    }
    Ok(())
}

fn inspect_complete_atlas(path: PathBuf, ram_mib: usize, pieces: String) -> Result<()> {
    ensure!(path.is_dir(), "database directory does not exist");
    let db = open_db(&path, ram_mib, 1, false)?;
    ensure!(
        read_required_meta(&db, META_SCHEMA_VERSION)? == SCHEMA_VERSION,
        "unsupported atlas schema"
    );
    ensure!(
        read_required_meta(&db, META_STATUS)? == STATUS_COMPLETE,
        "atlas is not complete"
    );
    let owners = OwnerMap::from_bytes(&read_required_meta(&db, META_OWNER_MAP)?)?;
    ensure!(owners.writer_count > 0, "atlas owner map is empty");

    let node_cf = cf(&db, CF_NODE)?;
    let mut state = seed_key()?;
    if pieces.is_empty() {
        return print_state_moves(&db, state);
    }
    for (turn, value) in pieces.chars().enumerate() {
        let piece = piece_from_char(value)?;
        let (_, bag) = state.decode()?;
        ensure!(
            bag.contains(piece),
            "turn {} piece {piece:?} is not in bag {:#09b}",
            turn + 1,
            u8::from(bag)
        );
        let record_value = db
            .get_pinned_cf(&node_cf, state.as_bytes())
            .context("read replay state")?
            .context("replay state is absent from atlas")?;
        let record = NodeRecord::from_slice(record_value.as_ref(), bag)?;
        let MoveSlot::Active(index) = record.slot(piece)? else {
            bail!("turn {} has no active atlas move", turn + 1);
        };
        let orientation = validate_orientation(piece, index)?;
        println!("turn={} piece={piece:?} move={orientation}", turn + 1);
        state = successor(state, piece, orientation)?
            .context("atlas move leaves the admissibility carrier")?;
    }
    print_state_moves(&db, state)
}

fn run_explore(path: PathBuf, ram_mib: usize, pieces: String) -> Result<()> {
    let metadata = read_database_metadata(&path, ram_mib)?;
    match metadata.previous_status.as_slice() {
        STATUS_COMPLETE => inspect_complete_atlas(path, ram_mib, pieces),
        STATUS_PAUSED | STATUS_ABORTED => {
            bail!("atlas build is paused; use create to resume it")
        }
        STATUS_BUILDING => bail!("atlas build did not stop cleanly and cannot be explored"),
        STATUS_SEED_DEAD => bail!("database status is seed-dead and has no surviving atlas"),
        status => bail!("unsupported database status {}", status_text(status)),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Create { writers, ram_mib } => run_create(cli.db_path, writers, ram_mib),
        Mode::Explore { ram_mib, pieces } => run_explore(cli.db_path, ram_mib, pieces),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static TEST_DB_ID: AtomicU64 = AtomicU64::new(0);
    const TEST_DEAD_STATE_BLOOM_BYTES: usize = 4 * 1024;

    fn empty_key() -> StateKey {
        StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, TetrisPieceBagState::new())
            .expect("empty state must encode")
    }

    fn first_orientation(piece: TetrisPiece) -> TetrisPieceOrientation {
        TetrisPiecePlacement::all_from_piece(piece)[0].orientation
    }

    fn test_dead_filter() -> DeadStateBloom {
        DeadStateBloom::new(TEST_DEAD_STATE_BLOOM_BYTES)
            .expect("test dead-state Bloom filter must allocate")
    }

    #[test]
    fn production_dead_state_bloom_uses_one_gibibyte() {
        assert_eq!(DEAD_STATE_BLOOM_BYTES, 1024 * MIB_BYTES);
        assert!(DEAD_STATE_BLOOM_BYTES.is_power_of_two());
    }

    #[test]
    fn application_memory_reserve_includes_bloom_native_and_writer_memory() {
        let reserved =
            application_memory_reserve(8).expect("eight-writer reserve must be representable");

        assert_eq!(
            reserved,
            DEAD_STATE_BLOOM_BYTES + NATIVE_OVERHEAD_BYTES + 8 * PER_WRITER_MEMORY_RESERVE_BYTES
        );
        assert_eq!(reserved, 1472 * MIB_BYTES);
    }

    #[test]
    fn writer_memory_reserve_covers_fixed_scratch_batch_stack_and_transients() {
        let scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let hash_table_bytes = |capacity: usize, entry_bytes: usize| {
            let buckets = (capacity + 1).next_power_of_two();
            buckets * (entry_bytes + 1) + 16
        };
        let reusable_heap_bytes = hash_table_bytes(
            scratch.nodes.capacity(),
            std::mem::size_of::<(StateKey, NodeOverlayEntry)>(),
        ) + hash_table_bytes(
            scratch.redges.capacity(),
            std::mem::size_of::<(ReverseEdgeKey, bool)>(),
        ) + scratch.node_prefetch_plan.capacity()
            * std::mem::size_of::<StateKey>()
            + scratch.choice_requests.capacity() * std::mem::size_of::<ChoiceKey>()
            + scratch.prepared_choices.capacity() * std::mem::size_of::<PreparedChoice>()
            + scratch.prepared_placements.capacity() * std::mem::size_of::<PreparedPlacement>()
            + scratch.outputs.capacity() * std::mem::size_of::<OutputCommand>()
            + scratch.ordered_redges.capacity() * std::mem::size_of::<ReverseEdgeKey>()
            + scratch.drain_edges.capacity() * std::mem::size_of::<ReverseEdgeKey>();
        let fixed_stack_bytes = std::mem::size_of::<ClaimedBatch>()
            + std::mem::size_of::<WorkerScratch>()
            + std::mem::size_of::<ExpansionPlan>();
        let transient_headroom_bytes = MIB_BYTES;
        let covered_bytes = reusable_heap_bytes
            + fixed_stack_bytes
            + MAX_WRITE_BATCH_BYTES
            + WRITER_STACK_BYTES
            + transient_headroom_bytes;

        assert!(
            PER_WRITER_MEMORY_RESERVE_BYTES >= covered_bytes,
            "writer reserve {} MiB does not cover the estimated {:.2} MiB high-water mark",
            PER_WRITER_MEMORY_RESERVE_BYTES / MIB_BYTES,
            covered_bytes as f64 / MIB_BYTES as f64
        );
    }

    #[test]
    fn configured_writer_stack_holds_fixed_scratch_and_hot_path() {
        let (node_capacity, score) = thread::Builder::new()
            .stack_size(WRITER_STACK_BYTES)
            .spawn(|| {
                let claimed = ClaimedBatch::new();
                let expansion_plan = ExpansionPlan::new();
                let scratch = WorkerScratch::new().expect("worker scratch must allocate");
                let seed = empty_key();
                let next = successor(
                    seed,
                    TetrisPiece::O_PIECE,
                    first_orientation(TetrisPiece::O_PIECE),
                )
                .expect("seed successor must evaluate")
                .expect("first O placement must remain in the carrier");
                let (board, _) = next.decode().expect("successor must decode");
                let score = score_board(board, 0).expect("successor must score");
                std::hint::black_box(claimed);
                std::hint::black_box(expansion_plan);
                (scratch.nodes.capacity(), score)
            })
            .expect("configured writer stack must spawn")
            .join()
            .expect("configured writer stack must not overflow");

        assert!(node_capacity >= MAX_NODE_OVERLAY_ENTRIES);
        assert!(score.is_finite());
    }

    #[test]
    fn expansion_plan_is_small_and_covers_a_full_bag() {
        assert_eq!(ExpansionPlan::capacity(), TetrisPiece::NUM_PIECES);
        assert!(std::mem::size_of::<ExpansionPlan>() <= 512);
    }

    #[test]
    fn memory_budget_enforces_the_complete_application_reserve() {
        let error = memory_budget(1503, 8)
            .err()
            .expect("one MiB below the minimum must be rejected");
        assert!(
            error.to_string().contains("need at least 1504 MiB"),
            "unexpected error: {error:#}"
        );
        memory_budget(1504, 8).expect("the documented minimum must be accepted");
    }

    #[test]
    fn memory_budget_records_every_managed_capacity() {
        let memory = memory_budget(8192, 8).expect("production memory budget must be valid");
        let total = 8192 * MIB_BYTES;
        let reserve = 1472 * MIB_BYTES;
        let rocks = total - reserve;
        let (expected_cache, expected_memtable) = split_rocks_memory(rocks);

        assert_eq!(memory.configured_bytes, total);
        assert_eq!(memory.application_reserve_bytes, reserve);
        assert_eq!(memory.block_cache_capacity_bytes, expected_cache);
        assert_eq!(memory.memtable_capacity_bytes, expected_memtable);
    }

    #[test]
    fn rocks_memory_split_is_exact_and_overflow_safe() {
        for rocks_bytes in [0, 1, 2, 3, 4, 5, usize::MAX - 1, usize::MAX] {
            let (cache_bytes, memtable_bytes) = split_rocks_memory(rocks_bytes);

            assert_eq!(cache_bytes + memtable_bytes, rocks_bytes);
            assert_eq!(cache_bytes as u128, rocks_bytes as u128 * 2 / 3);
        }
    }

    #[test]
    fn storage_stats_keep_peak_rss_and_latest_gauges() {
        let stats = AtomicStorageStats::default();
        stats.store(&StorageStatsSnapshot {
            peak_rss_bytes: 10,
            block_cache_usage_bytes: 20,
            ..StorageStatsSnapshot::default()
        });
        stats.store(&StorageStatsSnapshot {
            peak_rss_bytes: 5,
            block_cache_usage_bytes: 30,
            ..StorageStatsSnapshot::default()
        });

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.peak_rss_bytes, 10);
        assert_eq!(snapshot.block_cache_usage_bytes, 30);
    }

    #[test]
    fn application_memory_reserve_rejects_writer_count_overflow() {
        let overflowing_writer_count = usize::MAX / PER_WRITER_MEMORY_RESERVE_BYTES + 1;
        assert!(application_memory_reserve(overflowing_writer_count).is_err());
    }

    #[test]
    fn command_memory_budget_defaults_to_two_gibibytes() {
        let cli = Cli::try_parse_from(["test", "--db-path", "atlas", "create"])
            .expect("create arguments must parse");
        let create_ram_mib = match cli.mode {
            Mode::Create { ram_mib, .. } => Some(ram_mib),
            Mode::Explore { .. } => None,
        };
        assert_eq!(create_ram_mib, Some(DEFAULT_RAM_MIB));

        let cli = Cli::try_parse_from(["test", "--db-path", "atlas", "explore"])
            .expect("explore arguments must parse");
        let explore_ram_mib = match cli.mode {
            Mode::Create { .. } => None,
            Mode::Explore { ram_mib, .. } => Some(ram_mib),
        };
        assert_eq!(explore_ram_mib, Some(DEFAULT_RAM_MIB));
    }

    #[test]
    fn rocksdb_options_bound_table_metadata_memory() {
        let (path, db, _) = test_db();
        drop(db);

        let options_path = fs::read_dir(&path)
            .expect("test database directory must be readable")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("OPTIONS-"))
            })
            .max()
            .expect("RocksDB must persist an OPTIONS file");
        let options = fs::read_to_string(options_path).expect("OPTIONS file must be readable");
        let db_options = options
            .split("[DBOptions]")
            .nth(1)
            .and_then(|text| text.split("\n[").next())
            .expect("DBOptions section must exist");
        assert!(
            db_options.contains(&format!("max_open_files={MAX_OPEN_FILES}")),
            "max_open_files must be finite: {db_options}"
        );
        assert!(
            db_options.contains(&format!("max_background_jobs={MAX_BACKGROUND_JOBS}")),
            "max_background_jobs must be fixed: {db_options}"
        );
        assert!(
            db_options.contains(&format!("max_subcompactions={MAX_SUBCOMPACTIONS}")),
            "max_subcompactions must be fixed: {db_options}"
        );

        for cf_name in [CF_NODE, CF_REDGE, CF_WORK] {
            let marker = format!("[TableOptions/BlockBasedTable \"{cf_name}\"]");
            let table_options = options
                .split(&marker)
                .nth(1)
                .and_then(|text| text.split("\n[").next())
                .expect("column-family table options must exist");
            for expected in [
                "cache_index_and_filter_blocks=true",
                "pin_l0_filter_and_index_blocks_in_cache=false",
                "pin_top_level_index_and_filter=false",
                "optimize_filters_for_memory=true",
                "partition_filters=true",
                "index_type=kTwoLevelIndexSearch",
            ] {
                assert!(
                    table_options.contains(expected),
                    "{cf_name} table options lack {expected}: {table_options}"
                );
            }
        }

        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn carrier_metadata_is_derived_from_enforced_limits() {
        assert_eq!(carrier_description(), "height<=4;holes<=2");

        let (path, db, owners) = test_db();
        initialize_database(&db, &owners, "test").expect("database must initialize");
        let persisted = db
            .get(META_CARRIER)
            .expect("carrier metadata read must succeed")
            .expect("carrier metadata must exist");
        assert_eq!(persisted, carrier_description().as_bytes());

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    fn test_db() -> (PathBuf, Arc<DB>, OwnerMap) {
        let id = TEST_DB_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("tetris-atlas-partitioned-{}-{id}", process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("old test database must be removable");
        }
        let owners = OwnerMap::balanced(1).expect("one writer is valid");
        let db =
            Arc::new(open_db(&path, DEFAULT_RAM_MIB, 1, true).expect("test database must open"));
        let mut batch = WriteBatch::default();
        append_work_heads(&mut batch, &owners, &[WorkClaimCursor::new()])
            .expect("test work heads must encode");
        db.write(batch).expect("test work heads must persist");
        (path, db, owners)
    }

    #[test]
    fn batch_processor_requires_the_node_column_family() {
        let id = TEST_DB_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tetris-atlas-partitioned-missing-node-cf-{}-{id}",
            process::id()
        ));
        if path.exists() {
            fs::remove_dir_all(&path).expect("old test database must be removable");
        }
        let db = DB::open_default(&path).expect("default-only test database must open");
        let owners = OwnerMap::balanced(1).expect("one writer is valid");
        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");

        let error = BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
            .err()
            .expect("missing node column family must reject the processor");
        assert!(error.to_string().contains("missing node column family"));

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    fn test_shared(db: Arc<DB>, owners: OwnerMap, in_flight: u64) -> Arc<SharedState> {
        let stats = RuntimeStats::new(1, 0);
        stats
            .outstanding_work
            .store(in_flight, AtomicOrdering::SeqCst);
        stats.owners[0]
            .outstanding
            .store(in_flight, AtomicOrdering::SeqCst);
        Arc::new(SharedState {
            db,
            owners,
            dead_filter: test_dead_filter(),
            stats,
            shutdown: Arc::new(ShutdownSignal::default()),
            converged: AtomicBool::new(false),
            seed_dead: AtomicBool::new(false),
            error: Mutex::new(None),
        })
    }

    fn persist_work_heads_for_test(db: &DB, owners: &OwnerMap, work_heads: &[WorkClaimCursor]) {
        let mut batch = WriteBatch::default();
        append_work_heads(&mut batch, owners, work_heads).expect("test work heads must encode");
        db.write(batch).expect("test work heads must persist");
    }

    fn delete_work_heads_for_test(db: &DB, owners: &OwnerMap) {
        let mut batch = WriteBatch::default();
        for owner in 0..owners.writer_count {
            for priority in PRIORITY_DEATH..=PRIORITY_EXPAND {
                for producer in 0..owners.writer_count {
                    batch.delete(work_head_meta_key(owner as u8, priority, producer as u8));
                }
            }
        }
        db.write(batch).expect("test work heads must be deleted");
    }

    #[test]
    fn pause_handler_does_not_retain_shared_database_state() {
        let (path, db, owners) = test_db();
        let shared = test_shared(Arc::clone(&db), owners, 0);
        let shutdown = Arc::clone(&shared.shutdown);
        let weak_shared = Arc::downgrade(&shared);
        let mut handler = pause_handler(Arc::clone(&shutdown));

        assert_eq!(Arc::strong_count(&shared), 1);
        drop(shared);
        assert!(weak_shared.upgrade().is_none());
        handler();
        assert!(shutdown.pause_requested.load(AtomicOrdering::SeqCst));
        assert!(shutdown.stop.load(AtomicOrdering::SeqCst));

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn worker_group_drop_stops_and_joins_workers() {
        let (path, db, owners) = test_db();
        let shared = test_shared(Arc::clone(&db), owners, 0);
        let shutdown = Arc::clone(&shared.shutdown);
        let finished = Arc::new(AtomicBool::new(false));
        {
            let mut workers = WorkerGroup::with_capacity(Arc::clone(&shared), 1);
            let worker_shutdown = Arc::clone(&shutdown);
            let worker_finished = Arc::clone(&finished);
            workers.push(thread::spawn(move || {
                while !worker_shutdown.stop.load(AtomicOrdering::SeqCst) {
                    thread::yield_now();
                }
                worker_finished.store(true, AtomicOrdering::SeqCst);
            }));
        }

        assert!(shutdown.stop.load(AtomicOrdering::SeqCst));
        assert!(finished.load(AtomicOrdering::SeqCst));
        drop(shared);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    struct TestBatchCommit {
        inputs: u64,
        outputs: u64,
    }

    fn process_next_batch(
        shared: &SharedState,
        sequence: &mut u64,
        claim_cursor: &mut WorkClaimCursor,
        scratch: &mut WorkerScratch,
    ) -> TestBatchCommit {
        let mut claimed = ClaimedBatch::new();
        claim_batch(&shared.db, &shared.owners, 0, 0, claim_cursor, &mut claimed)
            .expect("work claim must succeed");
        assert!(!claimed.is_empty());
        let mut processor = BatchProcessor::new(
            &shared.db,
            &shared.owners,
            &shared.dead_filter,
            0,
            sequence,
            scratch,
        )
        .expect("batch processor must initialize");
        processor
            .process_claimed(claimed.to_slice())
            .expect("command processing must succeed");
        let commit = TestBatchCommit {
            inputs: processor.scratch.processed.len() as u64,
            outputs: processor.scratch.outputs.len() as u64,
        };
        let mut options = WriteOptions::default();
        options.disable_wal(true);
        processor
            .commit(shared, &options, claim_cursor)
            .expect("command batch must commit");
        claim_cursor
            .advance_committed(processor.scratch.processed.to_slice())
            .expect("committed work must advance the cursor");
        commit
    }

    fn put_node_for_test(db: &DB, state: StateKey, record: NodeRecord) {
        let (_, bag) = state.decode().expect("test state must decode");
        record.validate(bag).expect("test record must be valid");
        let cf = db.cf_handle(CF_NODE).expect("node CF must exist");
        db.put_cf(&cf, state.as_bytes(), record.as_bytes())
            .expect("test node must be written");
    }

    fn first_admissible_orientation(state: StateKey, piece: TetrisPiece) -> TetrisPieceOrientation {
        TetrisPiecePlacement::all_from_piece(piece)
            .iter()
            .find_map(|placement| {
                successor(state, piece, placement.orientation)
                    .expect("test successor must evaluate")
                    .map(|_| placement.orientation)
            })
            .expect("test piece must have an admissible placement")
    }

    fn persist_piece_successors_dead(
        db: &DB,
        dead_filter: &DeadStateBloom,
        state: StateKey,
        piece: TetrisPiece,
    ) -> usize {
        let mut persisted = 0;
        for placement in TetrisPiecePlacement::all_from_piece(piece) {
            let Some(succ) = successor(state, piece, placement.orientation)
                .expect("test successor must evaluate")
            else {
                continue;
            };
            let succ_bag = succ.bag().expect("successor bag must decode");
            let witness = succ_bag
                .iter_pieces()
                .next()
                .expect("successor bag must be nonempty");
            let mut dead = NodeRecord::INIT;
            dead.set_slot(witness, MoveSlot::Exhausted)
                .expect("death witness must be valid");
            put_node_for_test(db, succ, dead);
            dead_filter.insert(succ);
            persisted += 1;
        }
        persisted
    }

    fn state_owned_by(owners: &OwnerMap, wanted: u8) -> StateKey {
        let mut found = None;
        for value in 0..10_000u32 {
            let mut limbs = [0u32; StandardTetris::COLS];
            limbs[0] = value;
            let state = StateKey::from_board_bag(
                TetrisBoard::from_limbs(limbs),
                TetrisPieceBagState::new(),
            )
            .expect("synthetic state must encode");
            if owners.owner(state) == wanted {
                found = Some(state);
                break;
            }
        }
        found.expect("test must find a state for each owner")
    }

    #[test]
    fn state_key_round_trips_and_rejects_invalid_data() {
        let mut board = TetrisBoard::EMPTY_BOARD;
        board.set_bit(0, 0);
        board.set_bit(9, 19);
        let bag = TetrisPieceBagState::from_pieces([TetrisPiece::I_PIECE, TetrisPiece::T_PIECE]);
        let key = StateKey::from_board_bag(board, bag).expect("state must encode");
        assert_eq!(key.decode().expect("state must decode"), (board, bag));
        assert_eq!(key.bag().expect("state bag must decode"), bag);

        assert!(StateKey::from_slice(&key.as_bytes()[..STATE_KEY_LEN - 1]).is_err());
        let mut empty_bag = key.0;
        empty_bag[STATE_KEY_LEN - 1] = 0;
        assert!(StateKey::from_slice(&empty_bag).is_err());
        assert!(StateKey(empty_bag).bag().is_err());
        let mut high_bit = key.0;
        high_bit[2] |= 0x10;
        assert!(StateKey::from_slice(&high_bit).is_err());
    }

    #[test]
    fn state_hash_preserves_the_persisted_owner_mapping() {
        let state = empty_key();
        assert_eq!(state_hash(state), 0x63a9799ad7b792ba);
        assert_eq!(state_shard(state), 186);
    }

    #[test]
    fn dead_state_bloom_has_no_false_negatives_after_insert() {
        let filter = test_dead_filter();
        let bag = TetrisPieceBagState::new();
        let mut states = Vec::new();
        for value in 0..512u32 {
            let mut limbs = [0u32; StandardTetris::COLS];
            limbs[0] = value;
            let state = StateKey::from_board_bag(TetrisBoard::from_limbs(limbs), bag)
                .expect("synthetic state must encode");
            states.push(state);
            filter.insert(state);
        }

        assert_eq!(filter.allocated_bytes(), TEST_DEAD_STATE_BLOOM_BYTES);
        assert_eq!(
            filter.capacity_bits(),
            (TEST_DEAD_STATE_BLOOM_BYTES as u64) * 8
        );
        assert!(filter.set_bits() > 0);
        assert!(states.iter().all(|&state| filter.might_contain(state)));
        assert_eq!(filter.insert(states[0]), 0);
    }

    #[test]
    fn dead_filter_negative_skips_db_and_false_positive_is_verified() {
        let (path, db, owners) = test_db();
        let live_state = empty_key();
        put_node_for_test(&db, live_state, NodeRecord::INIT);
        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");

        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        assert!(
            !processor
                .known_dead(live_state)
                .expect("filter negative must be accepted")
        );
        assert_eq!(processor.metrics.dead_filter.definite_negatives, 1);
        assert_eq!(processor.metrics.node_reads.candidate_db_gets, 0);
        assert!(processor.scratch.nodes.is_empty());
        drop(processor);

        dead_filter.insert(live_state);
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        assert!(
            !processor
                .known_dead(live_state)
                .expect("possible match must be verified")
        );
        assert_eq!(processor.metrics.dead_filter.maybe_positives, 1);
        assert_eq!(processor.metrics.dead_filter.false_positives, 1);
        assert_eq!(processor.metrics.node_reads.candidate_db_gets, 1);
        drop(processor);

        let mut limbs = [0u32; StandardTetris::COLS];
        limbs[0] = 1;
        let dead_state =
            StateKey::from_board_bag(TetrisBoard::from_limbs(limbs), TetrisPieceBagState::new())
                .expect("dead state must encode");
        let (_, bag) = dead_state.decode().expect("dead state must decode");
        let witness = bag.iter_pieces().next().expect("bag is nonempty");
        let mut dead_record = NodeRecord::INIT;
        dead_record
            .set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        put_node_for_test(&db, dead_state, dead_record);
        dead_filter.insert(dead_state);

        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        assert!(
            processor
                .known_dead(dead_state)
                .expect("possible dead state must be verified")
        );
        assert_eq!(processor.metrics.dead_filter.maybe_positives, 1);
        assert_eq!(processor.metrics.dead_filter.verified_dead, 1);
        assert_eq!(processor.metrics.dead_filter.false_positives, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn batch_dead_transition_populates_filter_and_overlay_wins() {
        let (path, db, owners) = test_db();
        let state = empty_key();
        let (_, bag) = state.decode().expect("state must decode");
        let witness = bag.iter_pieces().next().expect("bag is nonempty");
        let mut dead = NodeRecord::INIT;
        dead.set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");

        processor
            .transition_to_dead(state, dead)
            .expect("death transition must succeed");
        assert!(dead_filter.might_contain(state));
        assert_eq!(processor.metrics.dead_filter.inserts, 1);
        assert!(processor.metrics.dead_filter.bits_set > 0);
        assert!(
            processor
                .known_dead(state)
                .expect("batch overlay must expose the death transition")
        );
        assert_eq!(processor.metrics.node_reads.candidate_overlay_hits, 1);
        assert_eq!(processor.metrics.dead_filter.checks, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn recovery_rebuilds_dead_state_bloom_filter() {
        let (path, db, owners) = test_db();
        let dead_state = empty_key();
        let (_, bag) = dead_state.decode().expect("dead state must decode");
        let witness = bag.iter_pieces().next().expect("bag is nonempty");
        let mut dead_record = NodeRecord::INIT;
        dead_record
            .set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        put_node_for_test(&db, dead_state, dead_record);

        let filter = test_dead_filter();
        let recovery =
            recover_runtime(&db, &owners, &filter).expect("recovery must rebuild the filter");
        assert_eq!(recovery.nodes_created, 1);
        assert_eq!(recovery.nodes_dead, 1);
        assert!(recovery.dead_filter_bits_set > 0);
        assert_eq!(filter.set_bits(), recovery.dead_filter_bits_set);
        assert!(filter.might_contain(dead_state));

        let stats = RuntimeStats::recovered(&recovery).snapshot();
        assert_eq!(stats.dead_filter.inserts, 1);
        assert_eq!(stats.dead_filter.set_bits, recovery.dead_filter_bits_set);

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn node_record_derives_state_from_forced_slots() {
        let bag = TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE, TetrisPiece::I_PIECE]);
        let mut node = NodeRecord::INIT;
        assert!(!node.is_dead(bag).expect("valid node"));
        assert!(!node.is_resolved(bag).expect("valid node"));

        let o = first_orientation(TetrisPiece::O_PIECE).index();
        let i = first_orientation(TetrisPiece::I_PIECE).index();
        node.set_slot(TetrisPiece::O_PIECE, MoveSlot::Active(o))
            .expect("valid O orientation");
        node.set_slot(TetrisPiece::I_PIECE, MoveSlot::Pending(i))
            .expect("valid I orientation");
        assert!(node.has_pending(bag).expect("valid node"));
        assert!(!node.is_resolved(bag).expect("valid node"));

        node.set_slot(TetrisPiece::I_PIECE, MoveSlot::Active(i))
            .expect("valid I orientation");
        assert!(node.is_resolved(bag).expect("valid node"));

        node.set_slot(TetrisPiece::I_PIECE, MoveSlot::Exhausted)
            .expect("exhaustion is valid");
        assert!(node.is_dead(bag).expect("valid node"));
        assert!(NodeRecord::from_slice(node.as_bytes(), bag).is_ok());
    }

    #[test]
    fn closure_gauges_follow_live_node_resolution_and_death() {
        let bag = TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE, TetrisPiece::I_PIECE]);
        let initial =
            ClosureGauges::from_record(Some(NodeRecord::INIT), bag).expect("initial gauges");
        assert_eq!(
            initial,
            ClosureGauges {
                live_states: 1,
                states_with_unexpanded: 1,
                forced_slots_total: 2,
                forced_slots_unexpanded: 2,
                ..ClosureGauges::default()
            }
        );

        let mut pending = NodeRecord::INIT;
        pending
            .set_slot(
                TetrisPiece::O_PIECE,
                MoveSlot::Pending(first_orientation(TetrisPiece::O_PIECE).index()),
            )
            .expect("pending O move must be valid");
        let pending_gauges =
            ClosureGauges::from_record(Some(pending), bag).expect("pending gauges");
        assert_eq!(pending_gauges.states_with_unexpanded, 1);
        assert_eq!(pending_gauges.states_with_pending, 1);
        assert_eq!(pending_gauges.forced_slots_pending, 1);

        let mut resolved = pending;
        resolved
            .set_slot(
                TetrisPiece::O_PIECE,
                MoveSlot::Active(first_orientation(TetrisPiece::O_PIECE).index()),
            )
            .expect("active O move must be valid");
        resolved
            .set_slot(
                TetrisPiece::I_PIECE,
                MoveSlot::Active(first_orientation(TetrisPiece::I_PIECE).index()),
            )
            .expect("active I move must be valid");
        let resolved_gauges =
            ClosureGauges::from_record(Some(resolved), bag).expect("resolved gauges");
        assert_eq!(resolved_gauges.locally_resolved_states, 1);
        assert_eq!(resolved_gauges.forced_slots_active, 2);

        resolved
            .set_slot(TetrisPiece::I_PIECE, MoveSlot::Exhausted)
            .expect("exhaustion must be valid");
        assert_eq!(
            ClosureGauges::from_record(Some(resolved), bag).expect("dead gauges"),
            ClosureGauges::default()
        );
    }

    #[test]
    fn node_record_rejects_piece_mismatched_orientation() {
        let vertical_i = TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)
            .iter()
            .find(|placement| placement.orientation.index() >= 10)
            .expect("I has a vertical orientation")
            .orientation
            .index();
        assert!(
            MoveSlot::decode(TetrisPiece::O_PIECE, vertical_i).is_err(),
            "O must reject an I-only orientation"
        );
    }

    #[test]
    fn owner_map_is_balanced_and_round_trips() {
        for writers in [1, 2, 3, 4, 7, 255, 256] {
            let map = OwnerMap::balanced(writers).expect("valid writer count");
            let mut counts = vec![0usize; writers as usize];
            for owner in map.owners {
                counts[owner as usize] += 1;
            }
            let min = counts.iter().copied().min().expect("nonempty counts");
            let max = counts.iter().copied().max().expect("nonempty counts");
            assert!(max - min <= 1);
            assert_eq!(
                OwnerMap::from_bytes(&map.as_bytes())
                    .expect("owner map must decode")
                    .owners,
                map.owners
            );
        }
    }

    #[test]
    fn work_key_orders_sequence_in_big_endian_order() {
        let low = WorkKey::new(1, 2, 3, 255).expect("valid work key");
        let high = WorkKey::new(1, 2, 3, 256).expect("valid work key");
        assert!(low < high);
        assert_eq!(low.sequence(), 255);
        assert!(WorkKey::new(1, 2, 3, u64::MAX).is_err());
    }

    #[test]
    fn work_claim_cursor_is_fixed_size_and_advances_lanes_monotonically() {
        assert_eq!(
            std::mem::size_of::<WorkClaimCursor>(),
            WORK_PRIORITY_COUNT * VIRTUAL_SHARDS * std::mem::size_of::<u64>()
        );

        let mut cursor = WorkClaimCursor::new();
        let edge_ten = WorkKey::new(0, PRIORITY_EDGE, 2, 10).expect("work key must encode");
        let edge_five = WorkKey::new(0, PRIORITY_EDGE, 2, 5).expect("work key must encode");
        let expand_seven = WorkKey::new(0, PRIORITY_EXPAND, 3, 7).expect("work key must encode");
        cursor
            .advance_committed(&[edge_ten, edge_five, expand_seven])
            .expect("cursor advancement must succeed");

        assert_eq!(cursor.next(PRIORITY_EDGE, 2), 11);
        assert_eq!(cursor.next(PRIORITY_EXPAND, 3), 8);
        assert_eq!(cursor.next(PRIORITY_DEATH, 2), 0);
        assert_eq!(cursor.next(PRIORITY_EDGE, 3), 0);
    }

    #[test]
    fn work_head_metadata_must_be_all_present_or_all_absent() {
        let (path, db, owners) = test_db();
        db.delete(work_head_meta_key(0, PRIORITY_EDGE, 0))
            .expect("one work head must be deleted");

        let error = read_work_heads(&db, &owners).expect_err("partial work heads must be rejected");
        assert!(
            error
                .to_string()
                .contains("work-head metadata is partially populated")
        );

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn recovery_rejects_missing_work_heads() {
        let (path, db, owners) = test_db();
        delete_work_heads_for_test(&db, &owners);
        let error = recover_runtime(&db, &owners, &test_dead_filter())
            .err()
            .expect("recovery without persistent heads must fail");
        assert!(error.to_string().contains("missing work-head metadata"));

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn legacy_schema_is_rejected_without_migration() {
        let (path, db, owners) = test_db();
        initialize_database(&db, &owners, "test").expect("database must initialize");
        db.put(META_SCHEMA_VERSION, b"partitioned-greedy-v1")
            .expect("legacy schema must be written");
        drop(db);

        let error = read_database_metadata(&path, DEFAULT_RAM_MIB)
            .err()
            .expect("legacy schema must not be migrated");
        assert!(error.to_string().contains("unsupported atlas schema"));

        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn command_codec_round_trips_and_validates_work_routing() {
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");
        let command = Command::AttachResult {
            accepted: true,
            succ,
            pred,
            piece,
            orientation,
        };
        let encoded = command.encode().expect("command must encode");
        assert_eq!(
            Command::decode(encoded.as_ref()).expect("command must decode"),
            command
        );

        let map = OwnerMap::balanced(4).expect("valid owner map");
        let key = WorkKey::new(map.owner(pred), command.priority(), map.owner(succ), 42)
            .expect("valid work key");
        map.validate_work(key, command)
            .expect("command routing must validate");
        let wrong_priority =
            WorkKey::new(key.owner(), PRIORITY_EXPAND, key.producer(), 42).expect("valid work key");
        assert!(map.validate_work(wrong_priority, command).is_err());
    }

    #[test]
    fn command_node_read_key_matches_each_handler() {
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");

        let cases = [
            (Command::Expand { state: pred }, Some(pred)),
            (
                Command::AttachEdge {
                    succ,
                    pred,
                    piece,
                    orientation,
                },
                Some(succ),
            ),
            (
                Command::AttachResult {
                    accepted: true,
                    succ,
                    pred,
                    piece,
                    orientation,
                },
                Some(pred),
            ),
            (
                Command::ChildDead {
                    succ,
                    pred,
                    piece,
                    orientation,
                },
                Some(pred),
            ),
            (
                Command::DetachEdge {
                    succ,
                    pred,
                    piece,
                    orientation,
                },
                None,
            ),
            (Command::DrainDead { succ }, Some(succ)),
        ];
        for (command, expected) in cases {
            assert_eq!(command.node_read_key().copied(), expected);
        }
    }

    #[test]
    fn node_prefetch_deduplicates_and_loads_present_and_missing_records() {
        let (path, db, owners) = test_db();
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");
        put_node_for_test(&db, pred, NodeRecord::INIT);
        db.flush().expect("test node must flush");

        let commands = [
            Command::AttachResult {
                accepted: true,
                succ,
                pred,
                piece,
                orientation,
            },
            Command::AttachEdge {
                succ,
                pred,
                piece,
                orientation,
            },
            Command::AttachResult {
                accepted: true,
                succ,
                pred,
                piece,
                orientation,
            },
            Command::DetachEdge {
                succ,
                pred,
                piece,
                orientation,
            },
        ];
        let mut claimed = ClaimedBatch::new();
        for (sequence, command) in commands.into_iter().enumerate() {
            let key = WorkKey::new(0, command.priority(), 0, sequence as u64)
                .expect("work key must encode");
            assert!(claimed.try_push(ClaimedCommand { key, command }));
        }

        let dead_filter = test_dead_filter();
        let mut next_sequence = claimed.len() as u64;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        {
            let mut processor = BatchProcessor::new(
                &db,
                &owners,
                &dead_filter,
                0,
                &mut next_sequence,
                &mut scratch,
            )
            .expect("batch processor must initialize");
            processor
                .prepare_and_prefetch_batch(claimed.to_slice())
                .expect("node prefetch must succeed");

            let mut expected_keys = vec![pred, succ];
            expected_keys.sort_unstable();
            assert_eq!(
                processor.scratch.node_prefetch_plan.as_slice(),
                expected_keys
            );
            assert_eq!(processor.scratch.nodes.len(), 2);
            let pred_entry = processor
                .scratch
                .nodes
                .get(&pred)
                .expect("present predecessor must be cached");
            assert_eq!(pred_entry.original, Some(NodeRecord::INIT));
            assert_eq!(pred_entry.record, Some(NodeRecord::INIT));
            assert!(!pred_entry.dirty);
            let succ_entry = processor
                .scratch
                .nodes
                .get(&succ)
                .expect("missing successor must be negatively cached");
            assert_eq!(succ_entry.original, None);
            assert_eq!(succ_entry.record, None);
            assert!(!succ_entry.dirty);
            assert_eq!(processor.metrics.node_reads.prefetch_calls, 1);
            assert_eq!(processor.metrics.node_reads.prefetch_keys, 2);
            assert_eq!(processor.metrics.node_reads.db_gets, 2);
            assert_eq!(processor.metrics.node_reads.db_found, 1);
            assert_eq!(processor.metrics.node_reads.db_missing, 1);
            assert_eq!(processor.metrics.node_reads.lookups, 0);
            assert_eq!(processor.metrics.node_reads.overlay_hits, 0);

            processor
                .attach_edge(succ, pred, piece, orientation)
                .expect("missing successor must be created from its prefetched absence");
            let succ_entry = processor
                .scratch
                .nodes
                .get(&succ)
                .expect("created successor must remain cached");
            assert_eq!(succ_entry.original, None);
            assert_eq!(succ_entry.record, Some(NodeRecord::INIT));
            assert!(succ_entry.dirty);
            assert_eq!(processor.metrics.nodes_created, 1);
            assert_eq!(processor.metrics.node_reads.lookups, 1);
            assert_eq!(processor.metrics.node_reads.overlay_hits, 1);
        }
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn node_prefetch_skips_commands_without_node_reads() {
        let (path, db, owners) = test_db();
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");
        let command = Command::DetachEdge {
            succ,
            pred,
            piece,
            orientation,
        };
        let claimed = [ClaimedCommand {
            key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
            command,
        }];
        let dead_filter = test_dead_filter();
        let mut next_sequence = 1;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        {
            let mut processor = BatchProcessor::new(
                &db,
                &owners,
                &dead_filter,
                0,
                &mut next_sequence,
                &mut scratch,
            )
            .expect("batch processor must initialize");
            processor
                .prepare_and_prefetch_batch(&claimed)
                .expect("empty node prefetch must succeed");
            assert!(processor.scratch.node_prefetch_plan.is_empty());
            assert!(processor.scratch.nodes.is_empty());
            assert_eq!(processor.metrics.node_reads.prefetch_calls, 0);
            assert_eq!(processor.metrics.node_reads.prefetch_keys, 0);
            assert_eq!(processor.metrics.node_reads.db_gets, 0);
        }
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn process_claimed_preserves_order_after_node_prefetch() {
        let (path, db, owners) = test_db();
        let bag = TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE]);
        let pred = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("predecessor must encode");
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");
        let mut record = NodeRecord::INIT;
        record
            .set_slot(piece, MoveSlot::Pending(orientation.index()))
            .expect("pending move must encode");
        put_node_for_test(&db, pred, record);

        let command = Command::AttachResult {
            accepted: true,
            succ,
            pred,
            piece,
            orientation,
        };
        let claimed = [
            ClaimedCommand {
                key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
                command,
            },
            ClaimedCommand {
                key: WorkKey::new(0, command.priority(), 0, 1).expect("work key must encode"),
                command,
            },
        ];
        let dead_filter = test_dead_filter();
        let mut next_sequence = 2;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        {
            let mut processor = BatchProcessor::new(
                &db,
                &owners,
                &dead_filter,
                0,
                &mut next_sequence,
                &mut scratch,
            )
            .expect("batch processor must initialize");
            processor
                .process_claimed(&claimed)
                .expect("commands must process in their original order");

            let final_record = processor
                .scratch
                .nodes
                .get(&pred)
                .and_then(|entry| entry.record)
                .expect("predecessor must remain cached");
            assert_eq!(
                final_record.slot(piece).expect("move slot must decode"),
                MoveSlot::Active(orientation.index())
            );
            assert_eq!(
                processor.scratch.processed.to_slice(),
                &claimed.map(|item| item.key)
            );
            assert!(processor.scratch.outputs.is_empty());
            assert_eq!(processor.metrics.node_reads.prefetch_calls, 1);
            assert_eq!(processor.metrics.node_reads.prefetch_keys, 1);
            assert_eq!(processor.metrics.node_reads.db_gets, 1);
            assert_eq!(processor.metrics.node_reads.db_found, 1);
            assert_eq!(processor.metrics.node_reads.db_missing, 0);
            assert_eq!(processor.metrics.node_reads.lookups, 2);
            assert_eq!(processor.metrics.node_reads.overlay_hits, 2);
            assert_eq!(
                processor.metrics.commands_stale[COMMAND_ATTACH_RESULT as usize],
                1
            );
        }
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn runtime_stats_publish_batch_and_owner_metrics() {
        let stats = RuntimeStats::new(2, 1);
        let mut metrics = BatchMetrics::default();
        metrics.commands_processed[COMMAND_EXPAND as usize] = 1;
        metrics.commands_produced[COMMAND_CHILD_DEAD as usize] = 2;
        metrics.commands_stale[COMMAND_EXPAND as usize] = 1;
        metrics.nodes_created = 2;
        metrics.nodes_dead = 1;
        metrics.expansion.doomed = 2;
        metrics.expansion.planned_attachments_published = 3;
        metrics.expansion.planned_attachments_avoided = 4;
        metrics.candidates.prepared_choice_requests = 8;
        metrics.candidates.prepared_choice_unique = 7;
        metrics.candidates.prepared_placements = 117;
        metrics.candidates.prepared_choices_unused = 2;
        metrics.candidates.unplanned_fallbacks = 1;
        metrics.candidates.examined = 5;
        metrics.candidates.rejected_lost = 1;
        metrics.candidates.rejected_height = 1;
        metrics.candidates.rejected_holes = 1;
        metrics.candidates.rejected_known_dead = 1;
        metrics.candidates.viable = 1;
        metrics.candidates.reuse_probes = 4;
        metrics.candidates.existing_overlay = 1;
        metrics.candidates.existing_database = 1;
        metrics.candidates.may_exist = 2;
        metrics.candidates.definitely_new = 2;
        metrics.candidates.chosen_likely_existing = 1;
        metrics.candidates.chosen_definitely_new = 1;
        metrics.attach_accepted = 3;
        metrics.attach_rejected = 2;
        metrics.reverse_edges.inserted = 3;
        metrics.reverse_edges.detached = 1;
        metrics.reverse_edges.drained = 2;
        metrics.reverse_edges.drain_continuations = 1;
        metrics.closure_delta = ClosureGauges {
            states_with_unexpanded: -1,
            states_with_pending: 1,
            forced_slots_unexpanded: -(TetrisPiece::NUM_PIECES as i64),
            forced_slots_pending: TetrisPiece::NUM_PIECES as i64,
            ..ClosureGauges::default()
        };
        metrics.reroutes = 2;
        metrics.reroutes_from_active = 1;
        metrics.reroutes_from_pending = 1;
        metrics.node_reads.lookups = 8;
        metrics.node_reads.overlay_hits = 3;
        metrics.node_reads.db_gets = 5;
        metrics.node_reads.db_found = 4;
        metrics.node_reads.db_missing = 1;
        metrics.node_reads.prefetch_calls = 2;
        metrics.node_reads.prefetch_keys = 4;
        metrics.node_reads.candidate_lookups = 6;
        metrics.node_reads.candidate_overlay_hits = 2;
        metrics.node_reads.candidate_db_gets = 4;
        metrics.dead_filter.checks = 5;
        metrics.dead_filter.definite_negatives = 3;
        metrics.dead_filter.maybe_positives = 2;
        metrics.dead_filter.verified_dead = 1;
        metrics.dead_filter.false_positives = 1;
        metrics.dead_filter.inserts = 1;
        metrics.dead_filter.bits_set = 4;
        metrics.stop_reason = Some(BatchStopReason::Outputs);
        let mut outputs_by_owner = [0; VIRTUAL_SHARDS];
        outputs_by_owner[0] = 1;
        outputs_by_owner[1] = 1;

        stats
            .reserve_outputs(2, &outputs_by_owner)
            .expect("output reservation must succeed");
        stats.record_claim(3, Duration::from_micros(5));
        stats.record_process(Duration::from_micros(9));
        stats
            .record_commit(1, &metrics, 1, 2, 512, 7)
            .expect("metrics commit must succeed");
        stats.record_busy(1, Duration::from_micros(11));
        stats.record_idle(0, Duration::from_micros(13));
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.outstanding_work, 2);
        assert_eq!((snapshot.nodes_created, snapshot.nodes_dead), (3, 1));
        assert_eq!(snapshot.commands.processed.expand, 1);
        assert_eq!(snapshot.commands.produced.expand, 1);
        assert_eq!(snapshot.commands.produced.child_dead, 2);
        assert_eq!(snapshot.commands.stale.expand, 1);
        assert_eq!(snapshot.expansion.doomed, 2);
        assert_eq!(snapshot.expansion.planned_attachments_published, 3);
        assert_eq!(snapshot.expansion.planned_attachments_avoided, 4);
        assert_eq!(snapshot.candidates.prepared_choice_requests, 8);
        assert_eq!(snapshot.candidates.prepared_choice_unique, 7);
        assert_eq!(snapshot.candidates.prepared_placements, 117);
        assert_eq!(snapshot.candidates.prepared_choices_unused, 2);
        assert_eq!(snapshot.candidates.unplanned_fallbacks, 1);
        assert_eq!(snapshot.candidates.examined, 5);
        assert_eq!(snapshot.candidates.viable, 1);
        assert_eq!(snapshot.candidates.reuse_probes, 4);
        assert_eq!(snapshot.candidates.existing_overlay, 1);
        assert_eq!(snapshot.candidates.existing_database, 1);
        assert_eq!(snapshot.candidates.may_exist, 2);
        assert_eq!(snapshot.candidates.definitely_new, 2);
        assert_eq!(snapshot.candidates.chosen_likely_existing, 1);
        assert_eq!(snapshot.candidates.chosen_definitely_new, 1);
        assert_eq!(snapshot.node_reads.lookups, 8);
        assert_eq!(snapshot.node_reads.overlay_hits, 3);
        assert_eq!(snapshot.node_reads.db_gets, 5);
        assert_eq!(snapshot.node_reads.db_found, 4);
        assert_eq!(snapshot.node_reads.db_missing, 1);
        assert_eq!(snapshot.node_reads.prefetch_calls, 2);
        assert_eq!(snapshot.node_reads.prefetch_keys, 4);
        assert_eq!(snapshot.node_reads.candidate_lookups, 6);
        assert_eq!(snapshot.node_reads.candidate_overlay_hits, 2);
        assert_eq!(snapshot.node_reads.candidate_db_gets, 4);
        assert_eq!(snapshot.dead_filter.checks, 5);
        assert_eq!(snapshot.dead_filter.definite_negatives, 3);
        assert_eq!(snapshot.dead_filter.maybe_positives, 2);
        assert_eq!(snapshot.dead_filter.verified_dead, 1);
        assert_eq!(snapshot.dead_filter.false_positives, 1);
        assert_eq!(snapshot.dead_filter.inserts, 1);
        assert_eq!(snapshot.dead_filter.set_bits, 4);
        assert_eq!((snapshot.attach_accepted, snapshot.attach_rejected), (3, 2));
        assert_eq!(snapshot.reverse_edges.drained, 2);
        assert_eq!(snapshot.closure.states_with_unexpanded, 0);
        assert_eq!(snapshot.closure.states_with_pending, 1);
        assert_eq!(
            snapshot.closure.forced_slots_pending,
            TetrisPiece::NUM_PIECES as u64
        );
        assert_eq!(snapshot.closure.reroutes, 2);
        assert_eq!(snapshot.batches.bytes_max, 512);
        assert_eq!(snapshot.batches.commit_micros_max, 7);
        assert_eq!(snapshot.batches.claim_calls, 1);
        assert_eq!(snapshot.batches.claimed_commands_total, 3);
        assert_eq!(snapshot.batches.claimed_commands_max, 3);
        assert_eq!(snapshot.batches.claim_micros_total, 5);
        assert_eq!(snapshot.batches.claim_micros_max, 5);
        assert_eq!(snapshot.batches.process_micros_total, 9);
        assert_eq!(snapshot.batches.process_micros_max, 9);
        assert_eq!(snapshot.batches.stopped_by_outputs, 1);
        let json = serde_json::to_value(&snapshot).expect("runtime stats must serialize");
        assert_eq!(json["expansion"]["doomed"], 2);
        assert_eq!(json["expansion"]["planned_attachments_published"], 3);
        assert_eq!(json["expansion"]["planned_attachments_avoided"], 4);
        assert_eq!(json["candidates"]["prepared_choice_requests"], 8);
        assert_eq!(json["candidates"]["prepared_choice_unique"], 7);
        assert_eq!(json["candidates"]["prepared_placements"], 117);
        assert_eq!(json["candidates"]["prepared_choices_unused"], 2);
        assert_eq!(json["candidates"]["unplanned_fallbacks"], 1);
        assert_eq!((snapshot.queue_min, snapshot.queue_max), (1, 1));
        assert_eq!(snapshot.owners[0].idle_micros, 13);
        assert_eq!(snapshot.owners[1].busy_micros, 11);
    }

    #[test]
    fn rocksdb_statistics_ticker_parser_reads_counts() {
        let statistics = "\
rocksdb.block.cache.hit COUNT : 42\n\
rocksdb.block.cache.data.hit COUNT : 31\n\
rocksdb.block.cache.index.miss COUNT : 9\n\
rocksdb.bloom.filter.useful COUNT : 123\n\
rocksdb.stall.micros COUNT : 17\n";
        assert_eq!(statistics_ticker(statistics, "rocksdb.block.cache.hit"), 42);
        assert_eq!(
            statistics_ticker(statistics, "rocksdb.block.cache.data.hit"),
            31
        );
        assert_eq!(
            statistics_ticker(statistics, "rocksdb.block.cache.index.miss"),
            9
        );
        assert_eq!(
            statistics_ticker(statistics, "rocksdb.bloom.filter.useful"),
            123
        );
        assert_eq!(statistics_ticker(statistics, "rocksdb.stall.micros"), 17);
        assert_eq!(statistics_ticker(statistics, "missing"), 0);
    }

    #[test]
    fn terminal_status_stores_only_exact_closure_summary() {
        let (path, db, _) = test_db();
        store_final_status(&db, STATUS_ABORTED, None).expect("aborted status must store");
        assert!(
            db.get(META_SUMMARY)
                .expect("summary read must succeed")
                .is_none()
        );
        store_final_status(&db, STATUS_PAUSED, None).expect("paused status must store");
        assert_eq!(
            db.get(META_STATUS)
                .expect("status read must succeed")
                .expect("status must exist"),
            STATUS_PAUSED
        );
        assert!(
            db.get(META_SUMMARY)
                .expect("summary read must succeed")
                .is_none()
        );

        let validation = ValidationSummary {
            total_states: 7,
            dead_states: 2,
            surviving_states: 5,
            active_moves: 11,
        };
        store_final_status(&db, STATUS_COMPLETE, Some(validation))
            .expect("complete status must store");
        let summary = db
            .get(META_SUMMARY)
            .expect("summary read must succeed")
            .expect("closure summary must exist");
        assert_eq!(
            summary,
            b"total_states=7;dead_states=2;surviving_states=5;active_moves=11"
        );

        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn closure_validation_rejects_an_unresolved_graph_with_no_work() {
        let (path, db, owners) = test_db();
        let seed = empty_key();
        put_node_for_test(&db, seed, NodeRecord::INIT);
        let retained = WorkKey::new(0, PRIORITY_EXPAND, 0, 0).expect("work key must encode");
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        db.put_cf(
            &work_cf,
            retained.as_bytes(),
            Command::Expand { state: seed }
                .encode()
                .expect("command must encode"),
        )
        .expect("retained work must be written");
        let mut heads = WorkClaimCursor::new();
        heads.set(PRIORITY_EXPAND, 0, 1);
        persist_work_heads_for_test(&db, &owners, &[heads]);

        let error = validate_closure(&db, &owners).expect_err("unresolved graph must not validate");
        assert!(
            error
                .to_string()
                .contains("surviving state has unresolved forced slots")
        );

        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn paused_database_recovers_work_counters_and_producer_sequence() {
        let (path, db, owners) = test_db();
        let (_, seed_owner) =
            initialize_database(&db, &owners, "created-at-test").expect("database must initialize");
        let shared = Arc::new(SharedState {
            db: Arc::clone(&db),
            owners,
            dead_filter: test_dead_filter(),
            stats: RuntimeStats::new(1, seed_owner),
            shutdown: Arc::new(ShutdownSignal::default()),
            converged: AtomicBool::new(false),
            seed_dead: AtomicBool::new(false),
            error: Mutex::new(None),
        });
        let mut sequence = 1;
        let mut claim_cursor = WorkClaimCursor::new();
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        process_next_batch(&shared, &mut sequence, &mut claim_cursor, &mut scratch);
        let before_pause = shared.stats.snapshot();
        store_final_status(&db, STATUS_PAUSED, None).expect("database must pause");

        drop(scratch);
        drop(shared);
        drop(db);

        let metadata = read_database_metadata(&path, DEFAULT_RAM_MIB)
            .expect("paused metadata must be readable");
        assert_eq!(metadata.previous_status, STATUS_PAUSED);
        assert_eq!(metadata.created_source_commit, "created-at-test");
        let db = Arc::new(
            open_db(&path, DEFAULT_RAM_MIB, 1, false).expect("paused database must reopen"),
        );
        let recovery_dead_filter = test_dead_filter();
        let recovery = recover_runtime(&db, &metadata.owners, &recovery_dead_filter)
            .expect("runtime state must recover exactly");
        assert_eq!(recovery.outstanding_work, before_pause.outstanding_work);
        assert_eq!(
            CommandCounts::from_array(recovery.commands_carried_in).total(),
            before_pause.outstanding_work
        );
        assert_eq!(recovery.nodes_created, before_pause.nodes_created);
        assert_eq!(recovery.nodes_dead, before_pause.nodes_dead);
        assert_eq!(recovery.next_sequences, [sequence]);

        mark_exploring(
            &db,
            &metadata.owners,
            &recovery.next_sequences,
            &recovery.work_heads,
        )
        .expect("database must resume");
        assert_eq!(
            db.get(META_STATUS)
                .expect("status read must succeed")
                .expect("status must exist"),
            STATUS_BUILDING
        );
        let recovered_stats = RuntimeStats::recovered(&recovery);
        let recovered_snapshot = recovered_stats.snapshot();
        assert_eq!(
            recovered_snapshot.commands.carried_in.total(),
            recovered_snapshot.outstanding_work
        );
        assert_eq!(
            recovered_snapshot.closure.states_with_pending,
            before_pause.closure.states_with_pending
        );

        let resumed = Arc::new(SharedState {
            db: Arc::clone(&db),
            owners: metadata.owners,
            dead_filter: recovery_dead_filter,
            stats: recovered_stats,
            shutdown: Arc::new(ShutdownSignal::default()),
            converged: AtomicBool::new(false),
            seed_dead: AtomicBool::new(false),
            error: Mutex::new(None),
        });
        let mut resumed_sequence = recovery.next_sequences[0];
        let mut resumed_claim_cursor = recovery.work_heads[0].clone();
        let mut resumed_scratch = WorkerScratch::new().expect("worker scratch must allocate");
        process_next_batch(
            &resumed,
            &mut resumed_sequence,
            &mut resumed_claim_cursor,
            &mut resumed_scratch,
        );
        let persisted_sequence = db
            .get(producer_sequence_meta_key(0))
            .expect("producer sequence read must succeed")
            .expect("producer sequence must exist");
        assert_eq!(
            decode_u64_metadata(&persisted_sequence, "producer sequence")
                .expect("producer sequence must decode"),
            resumed_sequence
        );
        let after_resume_filter = test_dead_filter();
        let after_resume = recover_runtime(&db, &resumed.owners, &after_resume_filter)
            .expect("resumed queue must remain recoverable");
        assert_eq!(
            after_resume.outstanding_work,
            resumed.stats.snapshot().outstanding_work
        );

        drop(resumed_scratch);
        drop(resumed);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn claim_batch_starts_at_the_lane_cursor_sequence() {
        let (path, db, owners) = test_db();
        let state = empty_key();
        let command = Command::Expand { state };
        let keys = [
            WorkKey::new(0, PRIORITY_EXPAND, 0, 5).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EXPAND, 0, 10).expect("work key must encode"),
        ];
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        for key in keys {
            owners
                .validate_work(key, command)
                .expect("test command must be routed correctly");
            db.put_cf(
                &work_cf,
                key.as_bytes(),
                command.encode().expect("command must encode"),
            )
            .expect("test work must be written");
        }

        let mut cursor = WorkClaimCursor::new();
        cursor
            .advance_committed(&[
                WorkKey::new(0, PRIORITY_EXPAND, 0, 7).expect("work key must encode")
            ])
            .expect("cursor advancement must succeed");
        let mut claimed = ClaimedBatch::new();
        claim_batch(&db, &owners, 0, 0, &cursor, &mut claimed).expect("work claim must succeed");

        assert_eq!(
            claimed
                .to_slice()
                .iter()
                .map(|item| item.key)
                .collect::<Vec<_>>(),
            vec![keys[1]]
        );

        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn claim_batch_scans_rotated_producer_ranges_and_restores_key_order() {
        let (path, db, _) = test_db();
        let owners = OwnerMap::balanced(2).expect("two writers are valid");
        let owner_zero = state_owned_by(&owners, 0);
        let owner_one = state_owned_by(&owners, 1);
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let commands = [
            Command::AttachResult {
                accepted: true,
                succ: owner_one,
                pred: owner_zero,
                piece,
                orientation,
            },
            Command::AttachResult {
                accepted: true,
                succ: owner_zero,
                pred: owner_zero,
                piece,
                orientation,
            },
        ];
        let keys = [
            WorkKey::new(0, PRIORITY_EDGE, 1, 7).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 0, 11).expect("work key must encode"),
        ];
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        for (key, command) in keys.into_iter().zip(commands) {
            owners
                .validate_work(key, command)
                .expect("test command must be routed correctly");
            db.put_cf(
                &work_cf,
                key.as_bytes(),
                command.encode().expect("command must encode"),
            )
            .expect("test work must be written");
        }
        for sequence in 12..(12 + WORK_BATCH_COMMANDS as u64) {
            let key = WorkKey::new(0, PRIORITY_EDGE, 0, sequence).expect("work key must encode");
            let command = Command::AttachResult {
                accepted: true,
                succ: owner_zero,
                pred: owner_zero,
                piece,
                orientation,
            };
            owners
                .validate_work(key, command)
                .expect("test command must be routed correctly");
            db.put_cf(
                &work_cf,
                key.as_bytes(),
                command.encode().expect("command must encode"),
            )
            .expect("test work must be written");
        }

        let mut claimed = ClaimedBatch::new();
        claim_batch(&db, &owners, 0, 1, &WorkClaimCursor::new(), &mut claimed)
            .expect("work claim must succeed");
        assert_eq!(claimed.len(), WORK_BATCH_COMMANDS);
        assert_eq!(claimed[0].key.producer(), 0);
        assert_eq!(
            claimed
                .to_slice()
                .iter()
                .filter(|item| item.key.producer() == 1)
                .count(),
            1
        );
        assert_eq!(claimed[WORK_BATCH_COMMANDS - 1].key.producer(), 1);

        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn claim_batch_applies_the_persisted_head_to_every_producer_lane() {
        let (path, db, _) = test_db();
        let owners = OwnerMap::balanced(3).expect("three writers are valid");
        let owner_zero = state_owned_by(&owners, 0);
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        let mut cursor = WorkClaimCursor::new();
        let mut expected = Vec::new();

        for producer in 0..owners.writer_count {
            let producer_state = state_owned_by(&owners, producer as u8);
            let command = Command::DetachEdge {
                succ: owner_zero,
                pred: producer_state,
                piece: TetrisPiece::O_PIECE,
                orientation: first_orientation(TetrisPiece::O_PIECE),
            };
            let head = 10 * (producer as u64 + 1);
            cursor.set(PRIORITY_EDGE, producer as u8, head);
            for sequence in [head - 1, head] {
                let key = WorkKey::new(0, PRIORITY_EDGE, producer as u8, sequence)
                    .expect("work key must encode");
                owners
                    .validate_work(key, command)
                    .expect("test command must be routed correctly");
                db.put_cf(
                    &work_cf,
                    key.as_bytes(),
                    command.encode().expect("command must encode"),
                )
                .expect("test work must be written");
                if sequence == head {
                    expected.push(key);
                }
            }
        }

        let mut claimed = ClaimedBatch::new();
        claim_batch(&db, &owners, 0, 0, &cursor, &mut claimed)
            .expect("claim must honor every producer head");
        assert_eq!(
            claimed
                .to_slice()
                .iter()
                .map(|item| item.key)
                .collect::<Vec<_>>(),
            expected
        );

        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn claim_batch_preserves_strict_priority_across_rotated_ranges() {
        let (path, db, _) = test_db();
        let owners = OwnerMap::balanced(2).expect("two writers are valid");
        let owner_zero = state_owned_by(&owners, 0);
        let owner_one = state_owned_by(&owners, 1);
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let edge = Command::DetachEdge {
            succ: owner_zero,
            pred: owner_zero,
            piece,
            orientation,
        };
        let death = Command::ChildDead {
            succ: owner_one,
            pred: owner_zero,
            piece,
            orientation,
        };
        let work = [
            (
                WorkKey::new(0, PRIORITY_EDGE, 0, 0).expect("work key must encode"),
                edge,
            ),
            (
                WorkKey::new(0, PRIORITY_DEATH, 1, 0).expect("work key must encode"),
                death,
            ),
        ];
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        for (key, command) in work {
            owners
                .validate_work(key, command)
                .expect("test command must be routed correctly");
            db.put_cf(
                &work_cf,
                key.as_bytes(),
                command.encode().expect("command must encode"),
            )
            .expect("test work must be written");
        }

        let mut claimed = ClaimedBatch::new();
        claim_batch(&db, &owners, 0, 0, &WorkClaimCursor::new(), &mut claimed)
            .expect("work claim must succeed");
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].command, death);
        assert_eq!(claimed[0].key.priority(), PRIORITY_DEATH);

        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn partial_rotated_claim_commit_preserves_unprocessed_work_and_recovery() {
        let (path, db, _) = test_db();
        let owners = OwnerMap::balanced(2).expect("two writers are valid");
        persist_work_heads_for_test(
            &db,
            &owners,
            &[WorkClaimCursor::new(), WorkClaimCursor::new()],
        );
        let owner_zero = state_owned_by(&owners, 0);
        let owner_one = state_owned_by(&owners, 1);
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let producer_zero = Command::DetachEdge {
            succ: owner_zero,
            pred: owner_zero,
            piece,
            orientation,
        };
        let producer_one = Command::DetachEdge {
            succ: owner_zero,
            pred: owner_one,
            piece,
            orientation,
        };
        let keys = [
            WorkKey::new(0, PRIORITY_EDGE, 0, 0).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 0, 1).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 1, 0).expect("work key must encode"),
        ];
        let commands = [producer_zero, producer_zero, producer_one];
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        put_node_for_test(&db, empty_key(), NodeRecord::INIT);
        for (key, command) in keys.into_iter().zip(commands) {
            owners
                .validate_work(key, command)
                .expect("test command must be routed correctly");
            db.put_cf(
                &work_cf,
                key.as_bytes(),
                command.encode().expect("command must encode"),
            )
            .expect("test work must be written");
        }

        let stats = RuntimeStats::new(2, 0);
        stats.outstanding_work.store(3, AtomicOrdering::SeqCst);
        stats.owners[0].outstanding.store(3, AtomicOrdering::SeqCst);
        stats.owners[1].outstanding.store(0, AtomicOrdering::SeqCst);
        let shared = Arc::new(SharedState {
            db: Arc::clone(&db),
            owners,
            dead_filter: test_dead_filter(),
            stats,
            shutdown: Arc::new(ShutdownSignal::default()),
            converged: AtomicBool::new(false),
            seed_dead: AtomicBool::new(false),
            error: Mutex::new(None),
        });

        let mut claim_cursor = WorkClaimCursor::new();
        let mut claimed = ClaimedBatch::new();
        claim_batch(&db, &shared.owners, 0, 1, &claim_cursor, &mut claimed)
            .expect("rotated work claim must succeed");
        assert_eq!(
            claimed
                .to_slice()
                .iter()
                .map(|item| item.key)
                .collect::<Vec<_>>(),
            keys
        );

        let mut sequence = 2;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor = BatchProcessor::new(
            &shared.db,
            &shared.owners,
            &shared.dead_filter,
            0,
            &mut sequence,
            &mut scratch,
        )
        .expect("batch processor must initialize");
        processor.estimated_bytes = MAX_WRITE_BATCH_BYTES - MAX_COMMAND_WRITE_BYTES;
        processor
            .process_claimed(claimed.to_slice())
            .expect("partial command processing must succeed");
        assert_eq!(processor.scratch.processed.to_slice(), &keys[..1]);
        assert_eq!(claim_cursor.next(PRIORITY_EDGE, 0), 0);
        assert_eq!(claim_cursor.next(PRIORITY_EDGE, 1), 0);
        assert_eq!(
            processor.metrics.stop_reason,
            Some(BatchStopReason::WriteBytes)
        );
        let mut options = WriteOptions::default();
        options.disable_wal(true);
        processor
            .commit(&shared, &options, &claim_cursor)
            .expect("partial batch must commit");
        claim_cursor
            .advance_committed(processor.scratch.processed.to_slice())
            .expect("committed work must advance the cursor");
        assert_eq!(claim_cursor.next(PRIORITY_EDGE, 0), 1);
        assert_eq!(claim_cursor.next(PRIORITY_EDGE, 1), 0);
        assert_eq!(claim_cursor.next(PRIORITY_DEATH, 0), 0);
        drop(processor);

        assert!(
            db.get_cf(&work_cf, keys[0].as_bytes())
                .expect("processed work read must succeed")
                .is_some()
        );
        for key in &keys[1..] {
            assert!(
                db.get_cf(&work_cf, key.as_bytes())
                    .expect("unprocessed work read must succeed")
                    .is_some()
            );
        }
        assert_eq!(shared.stats.snapshot().outstanding_work, 2);
        let recovery_filter = test_dead_filter();
        let recovery = recover_runtime(&db, &shared.owners, &recovery_filter)
            .expect("remaining work must recover exactly");
        assert_eq!(recovery.outstanding_work, 2);
        assert_eq!(recovery.next_sequences, [2, 1]);
        assert_eq!(recovery.work_heads[0], claim_cursor);

        let mut warm_claimed = ClaimedBatch::new();
        claim_batch(&db, &shared.owners, 0, 1, &claim_cursor, &mut warm_claimed)
            .expect("warm cursor must claim remaining work");
        assert_eq!(
            warm_claimed
                .to_slice()
                .iter()
                .map(|item| item.key)
                .collect::<Vec<_>>(),
            keys[1..]
        );

        let restarted_cursor = read_work_heads(&db, &shared.owners)
            .expect("persisted work heads must read")
            .expect("persisted work heads must exist")
            .remove(0);
        let mut restarted_claimed = ClaimedBatch::new();
        claim_batch(
            &db,
            &shared.owners,
            0,
            1,
            &restarted_cursor,
            &mut restarted_claimed,
        )
        .expect("persisted restart cursor must claim remaining work");
        assert_eq!(
            restarted_claimed
                .to_slice()
                .iter()
                .map(|item| item.key)
                .collect::<Vec<_>>(),
            keys[1..]
        );

        drop(scratch);
        drop(shared);
        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn commit_persists_lane_heads_without_eagerly_deleting_consumed_work() {
        let (path, db, owners) = test_db();
        let processed = [
            WorkKey::new(0, PRIORITY_DEATH, 0, 1).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_DEATH, 0, 3).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 0, 2).expect("work key must encode"),
        ];
        let retained = [
            WorkKey::new(0, PRIORITY_DEATH, 0, 5).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 0, 4).expect("work key must encode"),
        ];
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        for key in processed.into_iter().chain(retained) {
            db.put_cf(&work_cf, key.as_bytes(), b"work")
                .expect("work key must be written");
        }

        let shared = test_shared(Arc::clone(&db), owners, processed.len() as u64);
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor = BatchProcessor::new(
            &shared.db,
            &shared.owners,
            &shared.dead_filter,
            0,
            &mut sequence,
            &mut scratch,
        )
        .expect("batch processor must initialize");
        for key in processed {
            assert!(processor.scratch.processed.try_push(key));
        }
        processor.metrics.commands_processed[COMMAND_EXPAND as usize] = processed.len() as u64;
        let mut options = WriteOptions::default();
        options.disable_wal(true);
        let claim_cursor = WorkClaimCursor::new();
        processor
            .commit(&shared, &options, &claim_cursor)
            .expect("batch must commit");

        for key in processed {
            assert!(
                db.get_cf(&work_cf, key.as_bytes())
                    .expect("work read must succeed")
                    .is_some()
            );
        }
        for key in retained {
            assert!(
                db.get_cf(&work_cf, key.as_bytes())
                    .expect("work read must succeed")
                    .is_some()
            );
        }
        let work_heads = read_work_heads(&db, &shared.owners)
            .expect("work heads must read")
            .expect("work heads must exist");
        assert_eq!(work_heads[0].next(PRIORITY_DEATH, 0), 4);
        assert_eq!(work_heads[0].next(PRIORITY_EDGE, 0), 3);
        assert_eq!(processor.metrics.work_heads_advanced, 2);
        assert_eq!(processor.metrics.work_reclaim_ranges, 0);

        drop(processor);
        drop(shared);
        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn commit_reclaims_only_complete_sequence_quanta() {
        let (path, db, owners) = test_db();
        let old_head = WORK_RECLAIM_SEQUENCE_QUANTUM - 2;
        let mut claim_cursor = WorkClaimCursor::new();
        claim_cursor.set(PRIORITY_EDGE, 0, old_head);
        persist_work_heads_for_test(&db, &owners, &[claim_cursor.clone()]);

        let state = empty_key();
        let command = Command::DetachEdge {
            succ: state,
            pred: state,
            piece: TetrisPiece::O_PIECE,
            orientation: first_orientation(TetrisPiece::O_PIECE),
        };
        let old_retained =
            WorkKey::new(0, PRIORITY_EDGE, 0, old_head - 2).expect("work key must encode");
        let processed = [
            WorkKey::new(0, PRIORITY_EDGE, 0, old_head).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 0, old_head + 1).expect("work key must encode"),
            WorkKey::new(0, PRIORITY_EDGE, 0, WORK_RECLAIM_SEQUENCE_QUANTUM)
                .expect("work key must encode"),
        ];
        let unprocessed = WorkKey::new(0, PRIORITY_EDGE, 0, WORK_RECLAIM_SEQUENCE_QUANTUM + 2)
            .expect("work key must encode");
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        for key in std::iter::once(old_retained)
            .chain(processed)
            .chain(std::iter::once(unprocessed))
        {
            owners
                .validate_work(key, command)
                .expect("test command must be routed correctly");
            db.put_cf(
                &work_cf,
                key.as_bytes(),
                command.encode().expect("command must encode"),
            )
            .expect("work key must be written");
        }

        let shared = test_shared(Arc::clone(&db), owners, processed.len() as u64);
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor = BatchProcessor::new(
            &shared.db,
            &shared.owners,
            &shared.dead_filter,
            0,
            &mut sequence,
            &mut scratch,
        )
        .expect("batch processor must initialize");
        for key in processed {
            assert!(processor.scratch.processed.try_push(key));
        }
        processor.metrics.commands_processed[COMMAND_DETACH_EDGE as usize] = processed.len() as u64;
        let mut options = WriteOptions::default();
        options.disable_wal(true);
        processor
            .commit(&shared, &options, &claim_cursor)
            .expect("boundary-crossing batch must commit");

        for key in std::iter::once(old_retained).chain(processed[..2].iter().copied()) {
            assert!(
                db.get_cf(&work_cf, key.as_bytes())
                    .expect("reclaimed work read must succeed")
                    .is_none()
            );
        }
        for key in [processed[2], unprocessed] {
            assert!(
                db.get_cf(&work_cf, key.as_bytes())
                    .expect("retained work read must succeed")
                    .is_some()
            );
        }
        let persisted = read_work_heads(&db, &shared.owners)
            .expect("work heads must read")
            .expect("work heads must exist");
        assert_eq!(
            persisted[0].next(PRIORITY_EDGE, 0),
            WORK_RECLAIM_SEQUENCE_QUANTUM + 1
        );
        assert_eq!(processor.metrics.work_heads_advanced, 1);
        assert_eq!(processor.metrics.work_reclaim_ranges, 1);
        let snapshot = shared.stats.snapshot();
        assert_eq!(snapshot.batches.work_heads_advanced, 1);
        assert_eq!(snapshot.batches.work_reclaim_ranges, 1);

        drop(processor);
        flush_all(&db).expect("test database must flush");
        drop(shared);
        drop(work_cf);
        drop(db);

        let owners = OwnerMap::balanced(1).expect("one writer is valid");
        let db =
            open_db(&path, DEFAULT_RAM_MIB, 1, false).expect("flushed test database must reopen");
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        assert!(
            db.get_cf(&work_cf, processed[2].as_bytes())
                .expect("retained processed work read must succeed")
                .is_some()
        );
        assert!(
            db.get_cf(&work_cf, unprocessed.as_bytes())
                .expect("unprocessed work read must succeed")
                .is_some()
        );
        let persisted = read_work_heads(&db, &owners)
            .expect("work heads must read after reopen")
            .expect("work heads must persist after reopen");
        assert_eq!(
            persisted[0].next(PRIORITY_EDGE, 0),
            WORK_RECLAIM_SEQUENCE_QUANTUM + 1
        );
        let mut claimed = ClaimedBatch::new();
        claim_batch(&db, &owners, 0, 0, &persisted[0], &mut claimed)
            .expect("reopened claim must honor the persisted head");
        assert_eq!(
            claimed
                .to_slice()
                .iter()
                .map(|item| item.key)
                .collect::<Vec<_>>(),
            vec![unprocessed]
        );

        drop(work_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn reverse_edge_codec_round_trips() {
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");
        let edge = ReverseEdgeKey::new(succ, pred, piece, orientation).expect("valid reverse edge");
        assert_eq!(
            ReverseEdgeKey::from_slice(edge.as_bytes())
                .expect("edge must decode")
                .decode()
                .expect("edge fields must decode"),
            (succ, pred, piece, orientation)
        );
    }

    #[test]
    fn candidate_order_prefers_reuse_then_score_then_low_orientation() {
        let key = empty_key();
        let low_orientation = first_orientation(TetrisPiece::O_PIECE);
        let high_orientation =
            TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[1].orientation;
        let existing_low_score = Candidate {
            reuse: CandidateReuse::Existing,
            score: -2.0,
            orientation: low_orientation,
            succ: key,
        };
        let new_high_score = Candidate {
            reuse: CandidateReuse::DefinitelyNew,
            score: -1.0,
            orientation: high_orientation,
            succ: key,
        };
        assert_eq!(
            candidate_order(&existing_low_score, &new_high_score),
            Ordering::Greater
        );

        let existing_high_score = Candidate {
            reuse: CandidateReuse::Existing,
            score: -1.0,
            orientation: high_orientation,
            succ: key,
        };
        assert_eq!(
            candidate_order(&existing_high_score, &existing_low_score),
            Ordering::Greater
        );

        let tied_low_orientation = Candidate {
            reuse: CandidateReuse::Existing,
            score: -1.0,
            orientation: low_orientation,
            succ: key,
        };
        assert_eq!(
            candidate_order(&tied_low_orientation, &existing_high_score),
            Ordering::Greater
        );
    }

    #[test]
    fn expand_publishes_a_complete_plan_and_duplicate_expand_is_a_no_op() {
        let (path, db, owners) = test_db();
        let bag = TetrisPieceBagState::from_pieces([
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::T_PIECE,
        ]);
        let state = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("test state must encode");
        put_node_for_test(&db, state, NodeRecord::INIT);

        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        let command = Command::Expand { state };
        let claimed = [ClaimedCommand {
            key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
            command,
        }];
        processor
            .prepare_and_prefetch_batch(&claimed)
            .expect("candidate-aware prefetch must succeed");
        processor.expand(state).expect("expansion must succeed");

        let expanded = processor
            .load_node(state)
            .expect("expanded state read must succeed")
            .expect("expanded state must exist");
        for piece in bag.iter_pieces() {
            assert!(matches!(
                expanded.slot(piece).expect("slot must decode"),
                MoveSlot::Pending(_)
            ));
            assert_eq!(
                processor
                    .scratch
                    .outputs
                    .iter()
                    .filter(|output| {
                        matches!(
                            output.command,
                            Command::AttachEdge { pred, piece: output_piece, .. }
                                if pred == state && output_piece == piece
                        )
                    })
                    .count(),
                1
            );
        }
        assert_eq!(processor.scratch.outputs.len(), 3);
        assert!(
            processor
                .scratch
                .outputs
                .iter()
                .all(|output| matches!(output.command, Command::AttachEdge { .. }))
        );
        assert_eq!(processor.metrics.expansion.doomed, 0);
        assert_eq!(processor.metrics.expansion.planned_attachments_published, 3);
        assert_eq!(processor.metrics.expansion.planned_attachments_avoided, 0);
        assert_eq!(processor.metrics.candidates.prepared_choice_requests, 3);
        assert_eq!(processor.metrics.candidates.prepared_choice_unique, 3);
        assert_eq!(processor.metrics.candidates.unplanned_fallbacks, 0);
        assert_eq!(processor.metrics.node_reads.candidate_db_gets, 0);
        assert_eq!(processor.metrics.candidates.reuse_probes, 0);
        assert_eq!(
            processor.metrics.candidates.chosen_likely_existing
                + processor.metrics.candidates.chosen_definitely_new,
            3
        );

        let outputs_before_duplicate = processor.scratch.outputs.len();
        let published_before_duplicate = processor.metrics.expansion.planned_attachments_published;
        processor
            .expand(state)
            .expect("duplicate expansion must be a no-op");
        assert_eq!(processor.scratch.outputs.len(), outputs_before_duplicate);
        assert_eq!(
            processor.metrics.expansion.planned_attachments_published,
            published_before_duplicate
        );

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn late_exhaustion_discards_the_complete_new_plan() {
        let (path, db, owners) = test_db();
        let bag = TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE, TetrisPiece::I_PIECE]);
        let state = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("test state must encode");
        put_node_for_test(&db, state, NodeRecord::INIT);
        let forced: Vec<_> = bag.iter_pieces().collect();
        assert_eq!(forced.len(), 2);

        let dead_filter = test_dead_filter();
        assert!(persist_piece_successors_dead(&db, &dead_filter, state, forced[1]) > 0);
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        let command = Command::Expand { state };
        let claimed = [ClaimedCommand {
            key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
            command,
        }];
        processor
            .prepare_and_prefetch_batch(&claimed)
            .expect("candidate-aware prefetch must succeed");
        processor
            .expand(state)
            .expect("doomed expansion must transition to dead");

        let dead = processor
            .load_node(state)
            .expect("dead state read must succeed")
            .expect("dead state must exist");
        assert!(dead.is_dead(bag).expect("dead state must validate"));
        assert_eq!(
            dead.slot(forced[0]).expect("first slot must decode"),
            MoveSlot::Unexpanded
        );
        assert_eq!(
            dead.slot(forced[1]).expect("doomed slot must decode"),
            MoveSlot::Exhausted
        );
        assert_eq!(processor.scratch.outputs.len(), 1);
        assert!(matches!(
            processor.scratch.outputs[0].command,
            Command::DrainDead { succ } if succ == state
        ));
        assert_eq!(processor.metrics.expansion.doomed, 1);
        assert_eq!(processor.metrics.expansion.planned_attachments_published, 0);
        assert_eq!(processor.metrics.expansion.planned_attachments_avoided, 1);
        assert_eq!(processor.metrics.candidates.unplanned_fallbacks, 0);
        assert_eq!(processor.metrics.node_reads.candidate_db_gets, 0);
        assert_eq!(processor.metrics.dead_filter.checks, 0);
        assert_eq!(processor.metrics.candidates.chosen_likely_existing, 0);
        assert_eq!(processor.metrics.candidates.chosen_definitely_new, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn doomed_partial_expansion_detaches_only_preexisting_edges() {
        let (path, db, owners) = test_db();
        let bag = TetrisPieceBagState::from_pieces([
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
        ]);
        let state = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("test state must encode");
        let forced: Vec<_> = bag.iter_pieces().collect();
        assert_eq!(forced.len(), 4);

        let mut record = NodeRecord::INIT;
        let active_orientation = first_admissible_orientation(state, forced[0]);
        let pending_orientation = first_admissible_orientation(state, forced[1]);
        record
            .set_slot(forced[0], MoveSlot::Active(active_orientation.index()))
            .expect("active slot must be valid");
        record
            .set_slot(forced[1], MoveSlot::Pending(pending_orientation.index()))
            .expect("pending slot must be valid");
        put_node_for_test(&db, state, record);

        let dead_filter = test_dead_filter();
        assert!(persist_piece_successors_dead(&db, &dead_filter, state, forced[3]) > 0);
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        let command = Command::Expand { state };
        let claimed = [ClaimedCommand {
            key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
            command,
        }];
        processor
            .prepare_and_prefetch_batch(&claimed)
            .expect("candidate-aware prefetch must succeed");
        processor
            .expand(state)
            .expect("partial expansion must transition to dead");

        let dead = processor
            .load_node(state)
            .expect("dead state read must succeed")
            .expect("dead state must exist");
        assert!(dead.is_dead(bag).expect("dead state must validate"));
        assert_eq!(
            dead.slot(forced[2]).expect("planned slot must decode"),
            MoveSlot::Unexpanded
        );
        assert_eq!(
            dead.slot(forced[3]).expect("doomed slot must decode"),
            MoveSlot::Exhausted
        );
        assert_eq!(
            processor
                .scratch
                .outputs
                .iter()
                .filter(|output| matches!(output.command, Command::DrainDead { .. }))
                .count(),
            1
        );
        assert_eq!(
            processor
                .scratch
                .outputs
                .iter()
                .filter(|output| matches!(output.command, Command::DetachEdge { .. }))
                .count(),
            2
        );
        assert!(
            processor
                .scratch
                .outputs
                .iter()
                .all(|output| !matches!(output.command, Command::AttachEdge { .. }))
        );
        assert_eq!(processor.metrics.expansion.doomed, 1);
        assert_eq!(processor.metrics.expansion.planned_attachments_published, 0);
        assert_eq!(processor.metrics.expansion.planned_attachments_avoided, 1);
        assert_eq!(processor.metrics.candidates.unplanned_fallbacks, 0);
        assert_eq!(
            processor
                .scratch
                .prepared_choices
                .iter()
                .filter(|choice| choice.used)
                .count(),
            2
        );
        assert_eq!(processor.metrics.candidates.chosen_likely_existing, 0);
        assert_eq!(processor.metrics.candidates.chosen_definitely_new, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn choose_next_prefers_a_persisted_successor_without_publishing_it() {
        let (path, db, owners) = test_db();
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = TetrisPiecePlacement::all_from_piece(piece)
            .last()
            .expect("O piece must have placements")
            .orientation;
        let existing = successor(pred, piece, orientation)
            .expect("successor calculation must succeed")
            .expect("edge O placement must be admissible");
        put_node_for_test(&db, existing, NodeRecord::INIT);
        db.flush().expect("test node must flush");

        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        let chosen = processor
            .choose_next(pred, piece)
            .expect("candidate selection must succeed")
            .expect("O piece must have a viable placement");

        assert_eq!(chosen.succ, existing);
        assert!(chosen.reuse.is_likely_existing());
        assert_eq!(processor.metrics.candidates.chosen_likely_existing, 0);
        assert_eq!(processor.metrics.candidates.chosen_definitely_new, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn prepared_choice_uses_exact_prefetched_existence_without_scalar_reads() {
        let (path, db, owners) = test_db();
        let piece = TetrisPiece::O_PIECE;
        let bag = TetrisPieceBagState::from_pieces([piece]);
        let pred = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("predecessor must encode");
        let orientation = TetrisPiecePlacement::all_from_piece(piece)
            .last()
            .expect("O piece must have placements")
            .orientation;
        let existing = successor(pred, piece, orientation)
            .expect("successor calculation must succeed")
            .expect("edge O placement must be admissible");
        put_node_for_test(&db, pred, NodeRecord::INIT);
        put_node_for_test(&db, existing, NodeRecord::INIT);

        let command = Command::Expand { state: pred };
        let claimed = [ClaimedCommand {
            key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
            command,
        }];
        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        processor
            .prepare_and_prefetch_batch(&claimed)
            .expect("candidate-aware prefetch must succeed");
        let chosen = processor
            .choose_next(pred, piece)
            .expect("prepared candidate selection must succeed")
            .expect("O piece must have a viable placement");

        assert_eq!(chosen.succ, existing);
        assert_eq!(chosen.reuse, CandidateReuse::Existing);
        assert_eq!(processor.metrics.candidates.prepared_choice_requests, 1);
        assert_eq!(processor.metrics.candidates.prepared_choice_unique, 1);
        assert_eq!(
            processor.metrics.candidates.prepared_placements,
            TetrisPiecePlacement::all_from_piece(piece).len() as u64
        );
        assert_eq!(processor.metrics.candidates.unplanned_fallbacks, 0);
        assert_eq!(processor.metrics.node_reads.candidate_db_gets, 0);
        assert_eq!(processor.metrics.candidates.reuse_probes, 0);
        assert_eq!(processor.metrics.dead_filter.checks, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn duplicate_choice_requests_share_one_prepared_placement_set() {
        let (path, db, owners) = test_db();
        let piece = TetrisPiece::O_PIECE;
        let bag = TetrisPieceBagState::from_pieces([piece]);
        let pred = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("predecessor must encode");
        put_node_for_test(&db, pred, NodeRecord::INIT);

        let command = Command::Expand { state: pred };
        let claimed = [
            ClaimedCommand {
                key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
                command,
            },
            ClaimedCommand {
                key: WorkKey::new(0, command.priority(), 0, 1).expect("work key must encode"),
                command,
            },
        ];
        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        processor
            .prepare_and_prefetch_batch(&claimed)
            .expect("candidate-aware prefetch must succeed");

        assert_eq!(processor.metrics.candidates.prepared_choice_requests, 2);
        assert_eq!(processor.metrics.candidates.prepared_choice_unique, 1);
        assert_eq!(processor.scratch.prepared_choices.len(), 1);
        assert_eq!(
            processor.scratch.prepared_placements.len(),
            TetrisPiecePlacement::all_from_piece(piece).len()
        );

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn prepared_choice_observes_same_batch_overlay_mutations() {
        let (path, db, owners) = test_db();
        let piece = TetrisPiece::O_PIECE;
        let bag = TetrisPieceBagState::from_pieces([piece]);
        let pred = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("predecessor must encode");
        let orientation = TetrisPiecePlacement::all_from_piece(piece)
            .last()
            .expect("O piece must have placements")
            .orientation;
        let candidate = successor(pred, piece, orientation)
            .expect("successor calculation must succeed")
            .expect("edge O placement must be admissible");
        put_node_for_test(&db, pred, NodeRecord::INIT);

        let command = Command::Expand { state: pred };
        let claimed = [ClaimedCommand {
            key: WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode"),
            command,
        }];
        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let owner = owners.owner(candidate);
        let mut processor = BatchProcessor::new(
            &db,
            &owners,
            &dead_filter,
            owner,
            &mut sequence,
            &mut scratch,
        )
        .expect("batch processor must initialize");
        processor
            .prepare_and_prefetch_batch(&claimed)
            .expect("candidate-aware prefetch must succeed");
        let prefetched = processor
            .scratch
            .nodes
            .get(&candidate)
            .expect("candidate must have an exact prefetched entry");
        assert_eq!(prefetched.original, None);
        assert_eq!(prefetched.record, None);

        processor
            .put_node(candidate, NodeRecord::INIT)
            .expect("same-batch creation must succeed");
        let chosen = processor
            .choose_next(pred, piece)
            .expect("prepared candidate selection must succeed")
            .expect("created candidate must be viable");
        assert_eq!(chosen.succ, candidate);
        assert_eq!(chosen.reuse, CandidateReuse::Existing);

        let candidate_bag = candidate.bag().expect("candidate bag must decode");
        let witness = candidate_bag
            .iter_pieces()
            .next()
            .expect("candidate bag must be nonempty");
        let mut dead = NodeRecord::INIT;
        dead.set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        processor
            .put_node(candidate, dead)
            .expect("same-batch death must succeed");
        let rerouted = processor
            .choose_next(pred, piece)
            .expect("prepared candidate reselection must succeed")
            .expect("another O placement must remain viable");
        assert_ne!(rerouted.succ, candidate);
        assert_eq!(processor.metrics.candidates.unplanned_fallbacks, 0);
        assert_eq!(processor.metrics.node_reads.candidate_db_gets, 0);

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn empty_seed_and_first_o_successor_are_admissible() {
        let seed = empty_key();
        let (board, _) = seed.decode().expect("seed must decode");
        assert!(is_admissible(board));
        let orientation = first_orientation(TetrisPiece::O_PIECE);
        let succ =
            successor(seed, TetrisPiece::O_PIECE, orientation).expect("successor calculation");
        assert!(succ.is_some());
        let succ_board = succ
            .expect("successor exists")
            .decode()
            .expect("successor must decode")
            .0;
        assert!(
            score_board(succ_board, 0)
                .expect("score must be valid")
                .is_finite()
        );
    }

    #[test]
    fn persisted_attachment_handshake_activates_seed_moves() {
        let (path, db, owners) = test_db();
        let bag = TetrisPieceBagState::from_pieces([TetrisPiece::O_PIECE]);
        let pred = StateKey::from_board_bag(TetrisBoard::EMPTY_BOARD, bag)
            .expect("predecessor must encode");
        put_node_for_test(&db, pred, NodeRecord::INIT);
        let command = Command::Expand { state: pred };
        let key = WorkKey::new(0, command.priority(), 0, 0).expect("work key must encode");
        let work_cf = db.cf_handle(CF_WORK).expect("work CF must exist");
        db.put_cf(
            &work_cf,
            key.as_bytes(),
            command.encode().expect("command must encode"),
        )
        .expect("work command must be written");
        drop(work_cf);

        let shared = test_shared(Arc::clone(&db), owners, 1);
        let mut sequence = 1;
        let mut claim_cursor = WorkClaimCursor::new();
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let capacities = (
            scratch.nodes.capacity(),
            scratch.node_prefetch_plan.capacity(),
            scratch.choice_requests.capacity(),
            scratch.prepared_choices.capacity(),
            scratch.prepared_placements.capacity(),
            scratch.redges.capacity(),
            scratch.outputs.capacity(),
            scratch.ordered_redges.capacity(),
            scratch.drain_edges.capacity(),
        );
        let pointers = (
            scratch.node_prefetch_plan.as_ptr(),
            scratch.choice_requests.as_ptr(),
            scratch.prepared_choices.as_ptr(),
            scratch.prepared_placements.as_ptr(),
            scratch.outputs.as_ptr(),
            scratch.ordered_redges.as_ptr(),
            scratch.drain_edges.as_ptr(),
        );

        let expand = process_next_batch(&shared, &mut sequence, &mut claim_cursor, &mut scratch);
        assert_eq!((expand.inputs, expand.outputs), (1, 1));
        let attaches = process_next_batch(&shared, &mut sequence, &mut claim_cursor, &mut scratch);
        assert_eq!((attaches.inputs, attaches.outputs), (1, 2));
        let results = process_next_batch(&shared, &mut sequence, &mut claim_cursor, &mut scratch);
        assert_eq!((results.inputs, results.outputs), (1, 0));
        assert_eq!(
            capacities,
            (
                scratch.nodes.capacity(),
                scratch.node_prefetch_plan.capacity(),
                scratch.choice_requests.capacity(),
                scratch.prepared_choices.capacity(),
                scratch.prepared_placements.capacity(),
                scratch.redges.capacity(),
                scratch.outputs.capacity(),
                scratch.ordered_redges.capacity(),
                scratch.drain_edges.capacity(),
            )
        );
        assert!(scratch.node_prefetch_plan.capacity() >= MAX_NODE_OVERLAY_ENTRIES);
        assert_eq!(
            pointers,
            (
                scratch.node_prefetch_plan.as_ptr(),
                scratch.choice_requests.as_ptr(),
                scratch.prepared_choices.as_ptr(),
                scratch.prepared_placements.as_ptr(),
                scratch.outputs.as_ptr(),
                scratch.ordered_redges.as_ptr(),
                scratch.drain_edges.as_ptr(),
            )
        );

        let cf = db.cf_handle(CF_NODE).expect("node CF must exist");
        let value = db
            .get_cf(&cf, pred.as_bytes())
            .expect("predecessor read must succeed")
            .expect("predecessor node must exist");
        let record = NodeRecord::from_slice(&value, bag).expect("predecessor node must decode");
        assert!(
            record
                .is_resolved(bag)
                .expect("predecessor node must be valid")
        );
        assert!(
            !record
                .has_pending(bag)
                .expect("predecessor node must be valid")
        );

        drop(cf);
        drop(shared);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn closure_metrics_follow_committed_seed_expansion() {
        let (path, db, owners) = test_db();
        let (_, seed_owner) =
            initialize_database(&db, &owners, "test").expect("database must initialize");
        let shared = Arc::new(SharedState {
            db: Arc::clone(&db),
            owners,
            dead_filter: test_dead_filter(),
            stats: RuntimeStats::new(1, seed_owner),
            shutdown: Arc::new(ShutdownSignal::default()),
            converged: AtomicBool::new(false),
            seed_dead: AtomicBool::new(false),
            error: Mutex::new(None),
        });
        let mut sequence = 1;
        let mut claim_cursor = WorkClaimCursor::new();
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let expansion = process_next_batch(&shared, &mut sequence, &mut claim_cursor, &mut scratch);
        assert_eq!(expansion.inputs, 1);
        assert_eq!(expansion.outputs, TetrisPiece::NUM_PIECES as u64);

        let snapshot = shared.stats.snapshot();
        assert_eq!(snapshot.expansion.doomed, 0);
        assert_eq!(
            snapshot.expansion.planned_attachments_published,
            TetrisPiece::NUM_PIECES as u64
        );
        assert_eq!(snapshot.expansion.planned_attachments_avoided, 0);
        assert_eq!(
            snapshot.candidates.prepared_choice_requests,
            TetrisPiece::NUM_PIECES as u64
        );
        assert_eq!(
            snapshot.candidates.prepared_choice_unique,
            TetrisPiece::NUM_PIECES as u64
        );
        assert_eq!(
            snapshot.candidates.prepared_placements,
            TetrisPiecePlacement::NUM_PLACEMENTS as u64
        );
        assert_eq!(snapshot.candidates.prepared_choices_unused, 0);
        assert_eq!(snapshot.candidates.unplanned_fallbacks, 0);
        assert_eq!(snapshot.node_reads.candidate_db_gets, 0);
        assert_eq!(snapshot.candidates.reuse_probes, 0);
        assert_eq!(snapshot.closure.nodes_not_dead, 1);
        assert_eq!(snapshot.closure.states_with_unexpanded, 0);
        assert_eq!(snapshot.closure.states_with_pending, 1);
        assert_eq!(snapshot.closure.locally_resolved_states, 0);
        assert_eq!(
            snapshot.closure.forced_slots_pending,
            TetrisPiece::NUM_PIECES as u64
        );
        assert_eq!(snapshot.closure.forced_slots_unexpanded, 0);
        assert_eq!(snapshot.closure.expand_backlog, 0);
        assert_eq!(
            snapshot.closure.coordination_backlog,
            TetrisPiece::NUM_PIECES as u64
        );

        drop(shared);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn death_before_attach_rejects_without_installing_edge() {
        let (path, db, owners) = test_db();
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");
        let (_, succ_bag) = succ.decode().expect("successor must decode");
        let witness = succ_bag
            .iter_pieces()
            .next()
            .expect("successor bag is nonempty");
        let mut dead = NodeRecord::INIT;
        dead.set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        put_node_for_test(&db, succ, dead);
        let dead_filter = test_dead_filter();
        dead_filter.insert(succ);

        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        processor
            .attach_edge(succ, pred, piece, orientation)
            .expect("attachment must be processed");
        assert!(processor.scratch.redges.is_empty());
        assert_eq!(
            processor.scratch.outputs.as_slice(),
            &[OutputCommand {
                key: WorkKey::new(0, PRIORITY_DEATH, 0, 0).expect("valid work key"),
                command: Command::AttachResult {
                    accepted: false,
                    succ,
                    pred,
                    piece,
                    orientation,
                },
            }]
        );

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn child_death_before_acceptance_makes_acceptance_stale() {
        let (path, db, owners) = test_db();
        let pred = empty_key();
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let succ = successor(pred, piece, orientation)
            .expect("successor calculation")
            .expect("first O placement is admissible");

        let (_, pred_bag) = pred.decode().expect("predecessor must decode");
        let mut pred_record = NodeRecord::INIT;
        pred_record
            .set_slot(piece, MoveSlot::Pending(orientation.index()))
            .expect("pending move must be valid");
        pred_record
            .validate(pred_bag)
            .expect("predecessor record must be valid");
        put_node_for_test(&db, pred, pred_record);

        let (_, succ_bag) = succ.decode().expect("successor must decode");
        let witness = succ_bag
            .iter_pieces()
            .next()
            .expect("successor bag is nonempty");
        let mut dead = NodeRecord::INIT;
        dead.set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        put_node_for_test(&db, succ, dead);
        let dead_filter = test_dead_filter();
        dead_filter.insert(succ);

        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        processor
            .child_dead(succ, pred, piece, orientation)
            .expect("child death must reroute");
        processor
            .attach_result(true, succ, pred, piece, orientation)
            .expect("late acceptance must be a no-op");
        let rerouted = processor
            .load_node(pred)
            .expect("predecessor read")
            .expect("predecessor exists");
        assert!(matches!(
            rerouted.slot(piece).expect("slot must decode"),
            MoveSlot::Pending(index) if index != orientation.index()
        ));
        assert_eq!(processor.metrics.reroutes, 1);
        assert_eq!(processor.metrics.reroutes_from_pending, 1);
        assert_eq!(processor.metrics.reroutes_from_active, 0);
        assert_eq!(
            processor.metrics.candidates.chosen_likely_existing
                + processor.metrics.candidates.chosen_definitely_new,
            1
        );

        drop(processor);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn reverse_edge_merge_applies_overlay_without_duplicates() {
        let (path, db, owners) = test_db();
        let succ = empty_key();
        let (_, bag) = succ.decode().expect("successor must decode");
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let mut edges = Vec::new();
        for value in 1..=3 {
            let mut limbs = [0u32; StandardTetris::COLS];
            limbs[0] = value;
            let pred = StateKey::from_board_bag(TetrisBoard::from_limbs(limbs), bag)
                .expect("synthetic predecessor must encode");
            edges.push(
                ReverseEdgeKey::new(succ, pred, piece, orientation).expect("edge must encode"),
            );
        }
        edges.sort_unstable();

        let redge_cf = db.cf_handle(CF_REDGE).expect("reverse-edge CF must exist");
        db.put_cf(&redge_cf, edges[0].as_bytes(), [])
            .expect("first reverse edge must be written");
        db.put_cf(&redge_cf, edges[2].as_bytes(), [])
            .expect("third reverse edge must be written");

        let dead_filter = test_dead_filter();
        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        processor
            .set_redge(edges[0], false)
            .expect("stored edge deletion must fit");
        processor
            .set_redge(edges[1], true)
            .expect("new edge insertion must fit");
        processor
            .set_redge(edges[2], true)
            .expect("duplicate edge insertion must fit");

        assert!(
            !processor
                .prepare_effective_reverse_edges(succ)
                .expect("reverse-edge merge must succeed")
        );
        assert_eq!(processor.scratch.drain_edges, edges[1..]);

        drop(processor);
        drop(redge_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }

    #[test]
    fn high_indegree_drain_is_chunked() {
        let (path, db, owners) = test_db();
        let succ = empty_key();
        let (_, bag) = succ.decode().expect("successor must decode");
        let witness = bag.iter_pieces().next().expect("successor bag is nonempty");
        let mut dead = NodeRecord::INIT;
        dead.set_slot(witness, MoveSlot::Exhausted)
            .expect("death witness must be valid");
        put_node_for_test(&db, succ, dead);
        let dead_filter = test_dead_filter();
        dead_filter.insert(succ);

        let redge_cf = db.cf_handle(CF_REDGE).expect("reverse-edge CF must exist");
        let piece = TetrisPiece::O_PIECE;
        let orientation = first_orientation(piece);
        let mut batch = WriteBatch::default();
        for index in 0..MAX_REDGES_PER_DRAIN + 3 {
            let mut limbs = [0u32; StandardTetris::COLS];
            limbs[0] = index as u32 + 1;
            let pred = StateKey::from_board_bag(TetrisBoard::from_limbs(limbs), bag)
                .expect("synthetic predecessor must encode");
            let edge =
                ReverseEdgeKey::new(succ, pred, piece, orientation).expect("edge must encode");
            batch.put_cf(&redge_cf, edge.as_bytes(), []);
        }
        db.write(batch).expect("reverse edges must be written");

        let mut sequence = 0;
        let mut scratch = WorkerScratch::new().expect("worker scratch must allocate");
        let mut processor =
            BatchProcessor::new(&db, &owners, &dead_filter, 0, &mut sequence, &mut scratch)
                .expect("batch processor must initialize");
        processor
            .drain_dead(succ)
            .expect("dead successor must drain");
        let child_deaths = processor
            .scratch
            .outputs
            .iter()
            .filter(|output| matches!(output.command, Command::ChildDead { .. }))
            .count();
        let followups = processor
            .scratch
            .outputs
            .iter()
            .filter(|output| matches!(output.command, Command::DrainDead { .. }))
            .count();
        assert_eq!(child_deaths, MAX_REDGES_PER_DRAIN);
        assert_eq!(followups, 1);
        assert_eq!(processor.scratch.redges.len(), MAX_REDGES_PER_DRAIN);
        assert!(processor.estimated_bytes <= MAX_WRITE_BATCH_BYTES);

        drop(processor);
        drop(redge_cf);
        drop(db);
        fs::remove_dir_all(path).expect("test database must be removable");
    }
}
