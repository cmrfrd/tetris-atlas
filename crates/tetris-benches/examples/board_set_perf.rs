//! Bounded board-set kernels. Prepare oracle files before timing or profiling.
use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::{
    collections::HashSet,
    fs::File,
    hash::BuildHasher,
    hint::black_box,
    io::{BufReader, BufWriter, Read, Write},
    time::Instant,
};
use tetris_board_set::BoardSet;

type Key = [u32; 10];
const MAGIC: &[u8; 8] = b"BSMIC001";

#[allow(
    clippy::cast_possible_truncation,
    reason = "The fixed-seed generator intentionally returns the low 32 bits of its xorshift state."
)]
fn random(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as u32
}

fn mixed(mut key: Key, index: usize) -> Key {
    if index & 1 != 0 {
        key[index % 10] ^= 1 << (index % 32);
    }
    key
}

fn u64le(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn key(reader: &mut impl Read) -> Result<Key> {
    let mut key = [0; 10];
    for value in &mut key {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        *value = u32::from_le_bytes(bytes);
    }
    Ok(key)
}

fn prepare(source: &str, count: usize, destination: &str) -> Result<()> {
    ensure!(count > 0 && count <= u32::MAX as usize, "invalid count");
    let mut boards = Vec::with_capacity(count);
    if source == "random" || source == "shared" {
        let mut seed = 0xDEAD_BEEF_CAFE;
        for i in 0..count {
            let mut board = std::array::from_fn(|_| random(&mut seed));
            if source == "shared" {
                board[..8].fill(0x55);
                board[9] = u32::try_from(i)?;
            } else {
                board[0] = u32::try_from(i)?;
            }
            boards.push(board);
        }
    } else {
        let mut reader = BufReader::new(File::open(source)?);
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        ensure!(
            magic == *b"AT100B01" || magic == *b"ATREF001",
            "unknown source schema"
        );
        ensure!(
            u64::try_from(count)? <= u64le(&mut reader)?,
            "source too short"
        );
        for _ in 0..count {
            boards.push(key(&mut reader)?);
            if magic == *b"ATREF001" {
                let mut tail = [0; 11];
                reader.read_exact(&mut tail)?;
            }
        }
    }
    let mut oracle = HashSet::with_capacity(count);
    let fresh: Vec<_> = boards
        .iter()
        .map(|&key| u8::from(oracle.insert(key)))
        .collect();
    let mut writer = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(destination)?,
    );
    writer.write_all(MAGIC)?;
    writer.write_all(&(count as u64).to_le_bytes())?;
    for (i, (board, &fresh)) in boards.iter().zip(&fresh).enumerate() {
        for value in board {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&[fresh, u8::from(oracle.contains(&mixed(*board, i)))])?;
    }
    writer.flush()?;
    println!(
        "{}",
        json!({"source":source,"count":count,"unique":oracle.len(),"destination":destination,"closure_certified":false})
    );
    Ok(())
}

struct Input {
    boards: Vec<Key>,
    fresh: Vec<u8>,
    hits: Vec<Key>,
    queries: Vec<Key>,
    expected: Vec<u8>,
    fingerprint: u64,
    unique: usize,
}

fn load(path: &str) -> Result<Input> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    ensure!(magic == *MAGIC, "unknown prepared schema");
    let n = usize::try_from(u64le(&mut reader)?)?;
    ensure!(n > 0, "empty fixture");
    let mut input = Input {
        boards: Vec::with_capacity(n),
        fresh: Vec::with_capacity(n),
        hits: Vec::with_capacity(n),
        queries: Vec::with_capacity(n),
        expected: Vec::with_capacity(n),
        fingerprint: 0xcbf29ce484222325,
        unique: 0,
    };
    for i in 0..n {
        let board = key(&mut reader)?;
        let mut flags = [0; 2];
        reader.read_exact(&mut flags)?;
        ensure!(flags.iter().all(|&flag| flag <= 1), "invalid oracle flags");
        for byte in board
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .chain(flags)
        {
            input.fingerprint = (input.fingerprint ^ byte as u64).wrapping_mul(0x100000001b3);
        }
        input.boards.push(board);
        input.hits.push(board);
        input.queries.push(mixed(board, i));
        input.fresh.push(flags[0]);
        input.expected.push(flags[1]);
        input.unique += flags[0] as usize;
    }
    let mut tail = [0];
    ensure!(reader.read(&mut tail)? == 0, "trailing fixture data");
    let mut seed = 0xFEED_D00D_DEAD_BEEF;
    for i in (1..n).rev() {
        let j = random(&mut seed) as usize % (i + 1);
        input.hits.swap(i, j);
        input.queries.swap(i, j);
        input.expected.swap(i, j);
    }
    Ok(input)
}

trait Set {
    fn with_capacity(n: usize) -> Self;
    fn insert(&mut self, key: Key) -> bool;
    fn contains(&self, key: &Key) -> bool;
    fn clear(&mut self);
    fn len(&self) -> usize;
}

impl Set for BoardSet<10> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity(n)
    }
    #[inline]
    fn insert(&mut self, key: Key) -> bool {
        self.insert(key)
    }
    #[inline]
    fn contains(&self, key: &Key) -> bool {
        self.contains(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn len(&self) -> usize {
        self.len()
    }
}

impl<H: BuildHasher + Default> Set for HashSet<Key, H> {
    fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, H::default())
    }
    #[inline]
    fn insert(&mut self, key: Key) -> bool {
        self.insert(key)
    }
    #[inline]
    fn contains(&self, key: &Key) -> bool {
        self.contains(key)
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn len(&self) -> usize {
        self.len()
    }
}

#[inline(never)]
fn insert<S: Set>(set: &mut S, boards: &[Key], output: &mut [u8]) {
    for (board, result) in boards.iter().zip(output) {
        *result = u8::from(set.insert(black_box(*board)));
    }
}

#[inline(never)]
fn contains<S: Set>(set: &S, boards: &[Key], output: &mut [u8]) {
    for (board, result) in boards.iter().zip(output) {
        *result = u8::from(set.contains(black_box(board)));
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "Sample counts and elapsed nanoseconds are converted to approximate floating-point benchmark statistics only."
)]
fn stats(samples: &[f64]) -> serde_json::Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sd = (samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (samples.len() - 1).max(1) as f64)
        .sqrt();
    json!({"n":samples.len(),"mean_ns":mean,"median_ns":sorted[sorted.len()/2],"sd_ns":sd,"min_ns":sorted[0],"max_ns":sorted[sorted.len()-1],"samples_ns":samples})
}

#[allow(
    clippy::cast_precision_loss,
    reason = "Population counts and elapsed nanoseconds are converted to approximate floating-point benchmark statistics only."
)]
fn run<S: Set>(input: &Input, backend: &str, reserve: bool, rounds: usize) -> Result<()> {
    ensure!(rounds > 0, "zero rounds");
    let n = input.boards.len();
    let capacity = if reserve { input.unique } else { 0 };
    let mut output = vec![0; n];
    let mut insertion = Vec::with_capacity(rounds);
    let mut constructors = Vec::with_capacity(rounds);
    let mut saved = None;
    for _ in 0..rounds {
        drop(saved.take());
        let start = Instant::now();
        let mut set = S::with_capacity(capacity);
        constructors.push(start.elapsed().as_nanos() as f64);
        let start = Instant::now();
        insert(&mut set, &input.boards, &mut output);
        insertion.push(start.elapsed().as_nanos() as f64 / n as f64);
        ensure!(
            output == input.fresh && set.len() == input.unique,
            "insertion mismatch"
        );
        saved = Some(set); // Previous set is destroyed outside timing.
    }
    let mut set = saved.context("missing set")?;
    let repeats = (2_000_000 / n).clamp(1, 64);
    let mut phases = serde_json::Map::new();
    phases.insert("insert".into(), stats(&insertion));
    phases.insert("constructor".into(), stats(&constructors));
    for mode in ["hits", "mixed", "duplicates", "clear_reuse"] {
        let mut samples = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let loops = if mode == "clear_reuse" { 1 } else { repeats };
            let start = Instant::now();
            for _ in 0..loops {
                match mode {
                    "hits" => contains(&set, &input.hits, &mut output),
                    "mixed" => contains(&set, &input.queries, &mut output),
                    "duplicates" => insert(&mut set, &input.hits, &mut output),
                    _ => {
                        set.clear();
                        insert(&mut set, &input.boards, &mut output);
                    }
                }
            }
            samples.push(start.elapsed().as_nanos() as f64 / (n * loops) as f64);
            let correct = match mode {
                "hits" => output.iter().all(|&flag| flag == 1),
                "mixed" => output == input.expected,
                "duplicates" => output.iter().all(|&flag| flag == 0),
                _ => output == input.fresh,
            };
            ensure!(correct && set.len() == input.unique, "{mode} mismatch");
        }
        phases.insert(mode.into(), stats(&samples));
    }
    println!(
        "{}",
        json!({"backend":backend,"reserve":reserve,"records":n,"unique":input.unique,"fingerprint":format!("{:016x}",input.fingerprint),"query_repeats":repeats,"phases":phases})
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prepare") => {
            ensure!(args.len() == 5, "prepare SOURCE COUNT OUTPUT");
            prepare(&args[2], args[3].parse()?, &args[4])
        }
        Some("run") => {
            ensure!(
                args.len() == 6,
                "run FIXTURE trie|std|fast reserved|growing ROUNDS"
            );
            let input = load(&args[2])?;
            ensure!(
                args[4] == "reserved" || args[4] == "growing",
                "unknown policy"
            );
            let reserve = args[4] == "reserved";
            let rounds = args[5].parse()?;
            match args[3].as_str() {
                "trie" => run::<BoardSet<10>>(&input, "trie", reserve, rounds),
                "std" => run::<HashSet<Key>>(&input, "std", reserve, rounds),
                "fast" => run::<HashSet<Key, foldhash::fast::RandomState>>(
                    &input, "fast", reserve, rounds,
                ),
                _ => anyhow::bail!("unknown backend"),
            }
        }
        _ => anyhow::bail!("prepare SOURCE COUNT OUTPUT | run FIXTURE BACKEND POLICY ROUNDS"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn scratch() -> Result<PathBuf> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/benchmarks/board-set-micro/tests")
            .join(format!(
                "{}-{}-{}",
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    #[test]
    fn prepared_answers_match_independent_oracle_and_all_backends() -> Result<()> {
        let directory = scratch()?;
        let path = directory.join("random");
        prepare("random", 128, path.to_str().context("path")?)?;
        let input = load(path.to_str().context("path")?)?;
        let mut oracle = HashSet::new();
        for (key, &fresh) in input.boards.iter().zip(&input.fresh) {
            assert_eq!(u8::from(oracle.insert(*key)), fresh);
        }
        for (query, &expected) in input.queries.iter().zip(&input.expected) {
            assert_eq!(u8::from(oracle.contains(query)), expected);
        }
        for reserve in [false, true] {
            run::<BoardSet<10>>(&input, "trie", reserve, 1)?;
            run::<HashSet<Key>>(&input, "std", reserve, 1)?;
            run::<HashSet<Key, foldhash::fast::RandomState>>(&input, "fast", reserve, 1)?;
        }
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn duplicate_trace_and_corrupted_oracle_are_detected() -> Result<()> {
        let directory = scratch()?;
        let source = directory.join("source");
        let path = directory.join("prepared");
        let mut bytes = b"AT100B01".to_vec();
        bytes.extend_from_slice(&3u64.to_le_bytes());
        for key in [[0u32; 10], [1; 10], [0; 10]] {
            for value in key {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(&source, bytes)?;
        prepare(
            source.to_str().context("path")?,
            3,
            path.to_str().context("path")?,
        )?;
        let input = load(path.to_str().context("path")?)?;
        assert_eq!(input.fresh, [1, 1, 0]);
        run::<BoardSet<10>>(&input, "trie", false, 1)?;
        assert!(prepare("random", 3, path.to_str().context("path")?).is_err());
        let mut bytes = fs::read(&path)?;
        bytes[16 + 40] = 0;
        fs::write(&path, bytes)?;
        let bad = load(path.to_str().context("path")?)?;
        assert!(run::<BoardSet<10>>(&bad, "trie", false, 1).is_err());
        fs::write(&path, b"BSMIC001")?;
        assert!(load(path.to_str().context("path")?).is_err());
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
