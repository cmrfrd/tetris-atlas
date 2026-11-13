use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use tetris_codec::BoardCodec;
use tetris_game::{StandardTetris, TetrisBoard, TetrisGameConfig};

// Change these compile-time parameters together to investigate another codec.
pub type Game = StandardTetris;
pub const ROWS: usize = 4;
pub const MAX_HOLES: usize = 2;
pub const COLS: usize = Game::COLS;
pub type Board = TetrisBoard<{ Game::COLS }, { Game::ROWS }>;
pub type Columns = [u32; COLS];
pub type Codec = BoardCodec<Game, ROWS, MAX_HOLES>;
pub const CODEC: Codec = Codec::new();
pub const FULL_BAG: u8 = 0x7f;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub seed: u64,
    pub parents: usize,
    pub episodes: usize,
    pub candidates: usize,
    pub horizon: usize,
    pub witnesses: usize,
    pub controls: bool,
    pub watch: usize,
}

impl Workload {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.parents <= 1_000_000 && self.candidates <= 1_000_000,
            "batch bounds exceeded"
        );
        ensure!(
            self.horizon > 0 && self.horizon <= 4096,
            "horizon must be in 1..=4096"
        );
        ensure!(
            self.episodes
                .checked_mul(self.horizon)
                .is_some_and(|n| n <= 10_000_000),
            "episode work bound exceeded"
        );
        ensure!(
            self.parents + self.episodes + self.candidates > 0,
            "empty workload"
        );
        ensure!(
            self.witnesses <= 1000,
            "witness limit must be <=1000 per mode per batch"
        );
        ensure!(
            COLS <= StandardTetris::COLS,
            "engine placement inventory supports at most standard width"
        );
        ensure!(self.watch <= 10000, "watchlist must be <=10000");
        ensure!(
            Game::PIECE_SET == tetris_game::TetrisPieceBagState::full(),
            "sampler models all seven standard pieces"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayTotals {
    pub episodes: u64,
    pub pieces: u64,
    pub lines: u64,
    pub codec_visits: u64,
    pub unique: u64,
    pub excursions: u64,
    pub returns: u64,
    pub stopped: u64,
    pub max_holes: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Totals {
    pub batches: u64,
    pub parent_samples: u64,
    pub parent_moves: u64,
    pub incoming_moves: u64,
    pub incoming_unique: u64,
    pub candidates: u64,
    pub candidate_incoming_hits: u64,
    pub candidate_witness_hits: u64,
    pub proved_impossible: u64,
    pub witnesses_saved: u64,
    pub strict: PlayTotals,
    pub relaxed: PlayTotals,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub workload: Workload,
    pub totals: Totals,
    pub rng: [Rng; 4],
    pub watch: Vec<u64>,
}

pub struct Bits {
    pub len: u64,
    pub words: Vec<AtomicU64>,
}

impl Bits {
    pub fn new(len: u64) -> Result<Self> {
        let n = usize::try_from(len.div_ceil(64))?;
        let mut words = Vec::new();
        words.try_reserve_exact(n)?;
        words.resize_with(n, || AtomicU64::new(0));
        Ok(Self { len, words })
    }

    pub fn contains(&self, id: u64) -> bool {
        assert!(id < self.len);
        self.words[usize::try_from(id / 64).expect("bit index fits the allocated word count")]
            .load(Ordering::Relaxed)
            & (1 << (id % 64))
            != 0
    }

    pub fn mark(&self, id: u64) -> bool {
        assert!(id < self.len);
        self.words[usize::try_from(id / 64).expect("bit index fits the allocated word count")]
            .fetch_or(1 << (id % 64), Ordering::Relaxed)
            & (1 << (id % 64))
            == 0
    }

    pub fn count(&self) -> u64 {
        self.words
            .iter()
            .map(|w| u64::from(w.load(Ordering::Relaxed).count_ones()))
            .sum()
    }

    pub fn validate_padding(&self) -> Result<()> {
        if !self.len.is_multiple_of(64) {
            let last = self.words.last().map_or(0, |w| w.load(Ordering::Relaxed));
            ensure!(last >> (self.len % 64) == 0, "nonzero bitmap padding");
        }

        Ok(())
    }
}

pub struct State {
    pub header: Header,

    // Incoming geometry hits, strict witnesses, relaxed witnesses, proved impossible.
    pub bits: [Bits; 4],
}

impl State {
    pub fn new(workload: Workload, count: u64) -> Result<Self> {
        let seed = workload.seed;
        let mut rng = [Rng(seed), Rng(seed ^ 1), Rng(seed ^ 2), Rng(seed ^ 3)];
        let mut watch = std::collections::BTreeSet::new();
        while watch.len() < workload.watch.min(usize::try_from(count)?) {
            watch.insert(rng[3].next() % count);
        }

        Ok(Self {
            header: Header {
                workload,
                totals: Totals::default(),
                rng,
                watch: watch.into_iter().collect(),
            },
            bits: [
                Bits::new(count)?,
                Bits::new(count)?,
                Bits::new(count)?,
                Bits::new(count)?,
            ],
        })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.header.watch.len()
                == self
                    .header
                    .workload
                    .watch
                    .min(usize::try_from(self.bits[0].len)?),
            "watchlist size mismatch"
        );
        ensure!(
            self.header.watch.windows(2).all(|p| p[0] < p[1])
                && self.header.watch.iter().all(|&id| id < self.bits[0].len),
            "invalid watchlist"
        );

        for bits in &self.bits {
            bits.validate_padding()?;
        }

        let t = &self.header.totals;
        ensure!(
            self.bits[0].count() == t.incoming_unique,
            "incoming count mismatch"
        );
        ensure!(
            self.bits[1].count() == t.strict.unique,
            "strict count mismatch"
        );
        ensure!(
            self.bits[2].count() == t.relaxed.unique,
            "relaxed count mismatch"
        );
        ensure!(
            self.bits[3].count() == t.proved_impossible,
            "proof count mismatch"
        );

        for i in 0..self.bits[0].words.len() {
            let proof = self.bits[3].words[i].load(Ordering::Relaxed);
            let observed = self.bits[0].words[i].load(Ordering::Relaxed)
                | self.bits[1].words[i].load(Ordering::Relaxed)
                | self.bits[2].words[i].load(Ordering::Relaxed);
            ensure!(
                proof & observed == 0,
                "a proved-impossible board has been observed"
            );
        }

        Ok(())
    }
}

pub fn holes(board: &Board) -> u32 {
    board.total_holes()
}

pub fn row(board: &Board, y: usize) -> u16 {
    board
        .as_limbs()
        .to_owned()
        .iter()
        .enumerate()
        .fold(0, |r, (x, c)| r | (((c >> y) & 1) as u16) << x)
}

pub fn nonempty_rows(board: &Board) -> bool {
    (0..board.height() as usize).all(|y| row(board, y) != 0)
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixtures use bounded board dimensions and case counts, or deliberately select low bits from deterministic random words."
)]
mod tests {
    use super::*;

    #[test]
    fn bitmap_has_exact_membership_and_deduplicates() {
        let b =
            Bits::new(130).expect("valid fixture in bitmap_has_exact_membership_and_deduplicates");
        for id in [0, 63, 64, 129] {
            assert!(b.mark(id));
            assert!(!b.mark(id));
        }

        assert_eq!(b.count(), 4);
        assert!(!b.contains(128));
        b.validate_padding()
            .expect("valid fixture in bitmap_has_exact_membership_and_deduplicates");
    }

    #[test]
    fn concurrent_marks_count_each_board_once() {
        use rayon::prelude::*;
        let bits = Bits::new(193).expect("valid fixture in concurrent_marks_count_each_board_once");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("valid fixture in concurrent_marks_count_each_board_once");
        let fresh: u64 = pool.install(|| {
            (0..100_000u64)
                .into_par_iter()
                .map(|i| u64::from(bits.mark(i % 193)))
                .sum()
        });

        assert_eq!(fresh, 193);
        assert_eq!(bits.count(), 193);
        bits.validate_padding()
            .expect("valid fixture in concurrent_marks_count_each_board_once");
    }

    #[test]
    fn padding_corruption_is_detected() {
        let b = Bits::new(65).expect("valid fixture in padding_corruption_is_detected");
        b.words[1].store(2, Ordering::Relaxed);
        assert!(b.validate_padding().is_err());
    }

    #[test]
    fn rng_zero_seed_is_not_absorbing_and_replays() {
        let mut a = Rng(0);
        let mut b = a;
        let values: Vec<_> = (0..100).map(|_| a.next()).collect();
        assert!(values.iter().any(|&v| v != 0));
        assert_eq!(values, (0..100).map(|_| b.next()).collect::<Vec<_>>());
    }
}
