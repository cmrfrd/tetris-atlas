use crate::{
    proof::{Finding, Prover},
    state::{COLS, Codec, Game, Header, MAX_HOLES, ROWS, State, Workload},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    time::{SystemTime, UNIX_EPOCH},
};
use tetris_game::TetrisGameConfig;
const MAGIC: &[u8; 8] = b"TCREACH1";
const VERSION: u32 = 1;
const HASH_SEED: u64 = 0xcbf29ce484222325;
fn hash(hash: &mut u64, data: &[u8]) {
    for &b in data {
        *hash = (*hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
}
fn fingerprint(parts: &[&[u8]]) -> u64 {
    let mut h = HASH_SEED;
    for p in parts {
        hash(&mut h, &(p.len() as u64).to_le_bytes());
        hash(&mut h, p);
    }
    h
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sources {
    engine: u64,
    codec: u64,
    sampler: u64,
}
impl Sources {
    pub fn current() -> Self {
        Self {
            engine: fingerprint(&[
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-game/src/tetris.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-game/src/config.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-utils/src/lib.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-macros/src/lib.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-macros/src/getters.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../rust-toolchain.toml"
                )),
            ]),
            codec: fingerprint(&[
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-codec/src/arithmetic.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-codec/src/decision.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-codec/src/lib.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-codec/src/board.rs"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../tetris-codec/src/id.rs"
                )),
            ]),
            sampler: fingerprint(&[
                include_bytes!("main.rs"),
                include_bytes!("state.rs"),
                include_bytes!("proof.rs"),
                include_bytes!("sampling.rs"),
                include_bytes!("storage.rs"),
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock")),
            ]),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    cols: usize,
    rows: usize,
    holes: usize,
    game_rows: usize,
    board_count: u64,
}
impl Model {
    fn current() -> Self {
        Self {
            cols: COLS,
            rows: ROWS,
            holes: MAX_HOLES,
            game_rows: Game::ROWS,
            board_count: Codec::BOARD_COUNT,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    model: Model,
    sources: Sources,
    workload: Workload,
    created_unix: u64,
    command: Vec<String>,
    git_head: Option<String>,
    git_dirty: Option<bool>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    model: Model,
    sources: Sources,
    header: Header,
    journal_bytes: u64,
}

pub struct Store {
    _lock: File,
    pub directory: PathBuf,
    journal: BufWriter<File>,
}
impl Store {
    pub fn open(directory: &Path, workload: Workload, resume: bool) -> Result<(Self, State)> {
        fs::create_dir_all(directory)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("writer.lock"))?;
        lock.try_lock().context("dataset already has a writer")?;
        if !resume {
            ensure!(
                fs::read_dir(directory)?.all(|e| e.is_ok_and(|e| e.file_name() == "writer.lock")),
                "output directory is not empty; use --resume or a new directory"
            );
        }
        let sources = Sources::current();
        let journal_path = directory.join("findings.jsonl");
        let state = if resume {
            let manifest: Manifest =
                serde_json::from_reader(File::open(directory.join("manifest.json"))?)?;
            ensure!(
                manifest.version == VERSION && manifest.model == Model::current(),
                "model/schema changed; start a new dataset"
            );
            ensure!(
                manifest.sources == sources && manifest.workload == workload,
                "source or workload changed; use the matching binary/options, or start a new dataset and verify the old findings"
            );
            let checkpoint = directory.join("checkpoint.bin");
            let (loaded, journal_bytes) = if checkpoint.exists() {
                load(&checkpoint, &workload, &sources, Codec::BOARD_COUNT)?
            } else {
                (State::new(workload.clone(), Codec::BOARD_COUNT)?, 0)
            };
            let mut journal = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&journal_path)?;
            recover_tail(&mut journal, journal_bytes, directory)?;
            loaded
        } else {
            let manifest = Manifest {
                version: VERSION,
                model: Model::current(),
                sources,
                workload: workload.clone(),
                created_unix: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                command: std::env::args().collect(),
                git_dirty: std::process::Command::new("git")
                    .args(["-C", env!("CARGO_MANIFEST_DIR"), "status", "--porcelain"])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| !o.stdout.is_empty()),
                git_head: std::process::Command::new("git")
                    .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()),
            };
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join("manifest.json"))
                .context("dataset exists; use --resume or a new output directory")?;
            serde_json::to_writer_pretty(&file, &manifest)?;
            file.sync_all()?;
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&journal_path)?
                .sync_all()?;
            State::new(workload, Codec::BOARD_COUNT)?
        };
        let journal = BufWriter::new(OpenOptions::new().append(true).open(journal_path)?);
        Ok((
            Self {
                _lock: lock,
                directory: directory.to_owned(),
                journal,
            },
            state,
        ))
    }
    pub fn append(&mut self, findings: &[Finding]) -> Result<()> {
        for f in findings {
            serde_json::to_writer(&mut self.journal, f)?;
            self.journal.write_all(b"\n")?;
        }
        self.journal.flush()?;
        Ok(())
    }
    pub fn checkpoint(&mut self, state: &State) -> Result<()> {
        self.journal.flush()?;
        self.journal.get_ref().sync_all()?;
        let journal_bytes = self.journal.get_ref().metadata()?.len();
        save(&self.directory.join("checkpoint.bin"), state, journal_bytes)?;
        let temporary = self.directory.join("candidates.tmp");
        let mut output = BufWriter::new(File::create(&temporary)?);
        writeln!(
            output,
            "batch,codec_id,incoming_hit,strict_witness,relaxed_witness,proved_impossible,columns"
        )?;
        for &id in &state.header.watch {
            let board = crate::state::CODEC.board_from_u64(id)?;
            let columns = board
                .as_limbs()
                .to_owned()
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(";");
            writeln!(
                output,
                "{},{id},{},{},{},{},{columns}",
                state.header.totals.batches,
                state.bits[0].contains(id),
                state.bits[1].contains(id),
                state.bits[2].contains(id),
                state.bits[3].contains(id)
            )?;
        }
        output.flush()?;
        fs::rename(temporary, self.directory.join("candidates.csv"))?;
        Ok(())
    }
}

fn recover_tail(journal: &mut File, committed: u64, directory: &Path) -> Result<()> {
    let length = journal.metadata()?.len();
    ensure!(
        length >= committed,
        "journal shorter than durable checkpoint"
    );
    if length > committed {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let backup = directory.join(format!("uncheckpointed-{stamp}.jsonl.part"));
        let mut backup = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(backup)?;
        journal.seek(SeekFrom::Start(committed))?;
        std::io::copy(&mut *journal, &mut backup)?;
        backup.sync_all()?;
        journal.set_len(committed)?;
        journal.sync_all()?;
    }
    Ok(())
}

fn write_hashed(w: &mut impl Write, hash_value: &mut u64, bytes: &[u8]) -> Result<()> {
    w.write_all(bytes)?;
    hash(hash_value, bytes);
    Ok(())
}
fn read_hashed(r: &mut impl Read, hash_value: &mut u64, bytes: &mut [u8]) -> Result<()> {
    r.read_exact(bytes)?;
    hash(hash_value, bytes);
    Ok(())
}
fn save(path: &Path, state: &State, journal_bytes: u64) -> Result<()> {
    let disk = Checkpoint {
        version: VERSION,
        model: Model {
            board_count: state.bits[0].len,
            ..Model::current()
        },
        sources: Sources::current(),
        header: state.header.clone(),
        journal_bytes,
    };
    let json = serde_json::to_vec(&disk)?;
    let temp = path.with_extension("tmp");
    let mut writer = BufWriter::with_capacity(256 * 1024, File::create(&temp)?);
    let mut digest = HASH_SEED;
    write_hashed(&mut writer, &mut digest, MAGIC)?;
    write_hashed(&mut writer, &mut digest, &(json.len() as u64).to_le_bytes())?;
    write_hashed(&mut writer, &mut digest, &json)?;
    for bits in &state.bits {
        for word in &bits.words {
            write_hashed(
                &mut writer,
                &mut digest,
                &word.load(Ordering::Relaxed).to_le_bytes(),
            )?;
        }
    }
    writer.write_all(&digest.to_le_bytes())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}
fn load(path: &Path, workload: &Workload, sources: &Sources, count: u64) -> Result<(State, u64)> {
    let mut reader = BufReader::with_capacity(256 * 1024, File::open(path)?);
    let mut digest = HASH_SEED;
    let mut magic = [0; 8];
    read_hashed(&mut reader, &mut digest, &mut magic)?;
    ensure!(&magic == MAGIC, "invalid checkpoint magic");
    let mut bytes = [0; 8];
    read_hashed(&mut reader, &mut digest, &mut bytes)?;
    let len = usize::try_from(u64::from_le_bytes(bytes))?;
    ensure!(len <= 1_048_576, "checkpoint header too large");
    let mut json = vec![0; len];
    read_hashed(&mut reader, &mut digest, &mut json)?;
    let disk: Checkpoint = serde_json::from_slice(&json)?;
    ensure!(
        disk.version == VERSION
            && disk.model
                == (Model {
                    board_count: count,
                    ..Model::current()
                }),
        "checkpoint model/schema mismatch"
    );
    ensure!(
        &disk.sources == sources && &disk.header.workload == workload,
        "checkpoint contract mismatch"
    );
    let mut state = State::new(workload.clone(), count)?;
    state.header = disk.header;
    for bits in &state.bits {
        for word in &bits.words {
            read_hashed(&mut reader, &mut digest, &mut bytes)?;
            word.store(u64::from_le_bytes(bytes), Ordering::Relaxed);
        }
    }
    reader.read_exact(&mut bytes)?;
    ensure!(
        u64::from_le_bytes(bytes) == digest,
        "checkpoint checksum mismatch"
    );
    ensure!(reader.read(&mut [0])? == 0, "trailing checkpoint bytes");
    state.validate()?;
    Ok((state, disk.journal_bytes))
}
pub fn verify(directory: &Path, prover: &Prover, recheck_engine: bool) -> Result<(u64, u64)> {
    let lock = File::open(directory.join("writer.lock"))?;
    lock.try_lock_shared()
        .context("stop the dataset writer before verification")?;
    let manifest: Manifest = serde_json::from_reader(File::open(directory.join("manifest.json"))?)?;
    let current = Model::current();
    ensure!(manifest.version == VERSION, "unsupported findings schema");
    ensure!(
        (
            manifest.model.cols,
            manifest.model.rows,
            manifest.model.holes,
            manifest.model.game_rows
        ) == (current.cols, current.rows, current.holes, current.game_rows),
        "proof model differs from this binary"
    );
    ensure!(
        recheck_engine || manifest.sources.engine == Sources::current().engine,
        "engine source changed; review the change and use --recheck-engine to replay facts against the current engine"
    );
    let mut proofs = 0;
    let mut witnesses = 0;
    for (index, line) in BufReader::new(File::open(directory.join("findings.jsonl"))?)
        .lines()
        .enumerate()
    {
        let finding: Finding = serde_json::from_str(&line?)
            .with_context(|| format!("finding {} is incomplete or invalid", index + 1))?;
        prover
            .verify(&finding)
            .with_context(|| format!("finding {} failed verification", index + 1))?;
        match finding {
            Finding::Impossible { .. } => proofs += 1,
            Finding::Witness { .. } => witnesses += 1,
        }
    }
    Ok((proofs, witnesses))
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
    fn fingerprints_are_framed_and_sensitive() {
        assert_ne!(fingerprint(&[b"ab", b"c"]), fingerprint(&[b"a", b"bc"]));
        assert_ne!(fingerprint(&[b"abc"]), fingerprint(&[b"abd"]));
    }
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "codec-reachability-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("valid fixture in new");
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            drop(fs::remove_dir_all(&self.0));
        }
    }
    fn work() -> Workload {
        Workload {
            seed: 1,
            parents: 1,
            episodes: 1,
            candidates: 1,
            horizon: 8,
            witnesses: 1,
            controls: false,
            watch: 7,
        }
    }
    #[test]
    fn checkpoint_roundtrip_corruption_truncation_and_contract_guards() {
        let dir = Directory::new();
        let path = dir.0.join("checkpoint.bin");
        let mut state = State::new(work(), 130).expect(
            "valid fixture in checkpoint_roundtrip_corruption_truncation_and_contract_guards",
        );
        state.bits[0].mark(64);
        state.header.totals.incoming_unique = 1;
        state.bits[1].mark(0);
        state.header.totals.strict.unique = 1;
        state.bits[3].mark(129);
        state.header.totals.proved_impossible = 1;
        state.header.totals.batches = 17;
        state.header.rng[0].next();
        save(&path, &state, 123).expect(
            "valid fixture in checkpoint_roundtrip_corruption_truncation_and_contract_guards",
        );
        let (loaded, offset) = load(&path, &work(), &Sources::current(), 130).expect(
            "valid fixture in checkpoint_roundtrip_corruption_truncation_and_contract_guards",
        );
        assert_eq!(offset, 123);
        assert_eq!(loaded.header, state.header);
        assert!(loaded.bits[0].contains(64));
        assert!(loaded.bits[3].contains(129));
        let mut changed = work();
        changed.seed = 99;
        assert!(load(&path, &changed, &Sources::current(), 130).is_err());
        let mut changed = Sources::current();
        changed.codec ^= 1;
        assert!(load(&path, &work(), &changed, 130).is_err());
        assert!(load(&path, &work(), &Sources::current(), 131).is_err());
        let bytes = fs::read(&path).expect(
            "valid fixture in checkpoint_roundtrip_corruption_truncation_and_contract_guards",
        );
        let mut corrupt = bytes.clone();
        let pos = corrupt.len() - 9;
        corrupt[pos] ^= 1;
        fs::write(&path, corrupt).expect(
            "valid fixture in checkpoint_roundtrip_corruption_truncation_and_contract_guards",
        );
        assert!(load(&path, &work(), &Sources::current(), 130).is_err());
        fs::write(&path, &bytes[..bytes.len() - 1]).expect(
            "valid fixture in checkpoint_roundtrip_corruption_truncation_and_contract_guards",
        );
        assert!(load(&path, &work(), &Sources::current(), 130).is_err());
    }
    #[test]
    fn journal_recovery_preserves_uncommitted_partial_record() {
        let dir = Directory::new();
        let path = dir.0.join("findings.jsonl");
        fs::write(&path, b"committed\npartial")
            .expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record");
        recover_tail(&mut file, 10, &dir.0)
            .expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record");
        assert_eq!(
            fs::read(&path)
                .expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record"),
            b"committed\n"
        );
        let backups: Vec<_> = fs::read_dir(&dir.0)
            .expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record")
            .map(|e| {
                e.expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record")
                    .path()
            })
            .filter(|p| p != &path)
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(
            fs::read(&backups[0])
                .expect("valid fixture in journal_recovery_preserves_uncommitted_partial_record"),
            b"partial"
        );
        assert!(recover_tail(&mut file, 11, &dir.0).is_err());
    }
    #[test]
    fn dataset_lock_excludes_another_writer() {
        let dir = Directory::new();
        let path = dir.0.join("writer.lock");
        let first =
            File::create(&path).expect("valid fixture in dataset_lock_excludes_another_writer");
        first
            .try_lock()
            .expect("valid fixture in dataset_lock_excludes_another_writer");
        let second = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("valid fixture in dataset_lock_excludes_another_writer");
        assert!(second.try_lock().is_err());
        drop(first);
        second
            .try_lock()
            .expect("valid fixture in dataset_lock_excludes_another_writer");
    }
    #[test]
    fn engine_source_recheck_is_explicit_and_keeps_model_guards() {
        let dir = Directory::new();
        File::create(dir.0.join("writer.lock"))
            .expect("valid fixture in engine_source_recheck_is_explicit_and_keeps_model_guards");
        File::create(dir.0.join("findings.jsonl"))
            .expect("valid fixture in engine_source_recheck_is_explicit_and_keeps_model_guards");
        let mut sources = Sources::current();
        sources.engine ^= 1;
        let mut manifest = Manifest {
            version: VERSION,
            model: Model::current(),
            sources,
            workload: work(),
            created_unix: 0,
            command: Vec::new(),
            git_head: None,
            git_dirty: None,
        };
        let write = |manifest: &Manifest| {
            fs::write(
                dir.0.join("manifest.json"),
                serde_json::to_vec(manifest).expect(
                    "valid fixture in engine_source_recheck_is_explicit_and_keeps_model_guards",
                ),
            )
            .expect("valid fixture in engine_source_recheck_is_explicit_and_keeps_model_guards")
        };
        write(&manifest);
        let prover = Prover::new()
            .expect("valid fixture in engine_source_recheck_is_explicit_and_keeps_model_guards");
        assert!(verify(&dir.0, &prover, false).is_err());
        assert_eq!(
            verify(&dir.0, &prover, true).expect(
                "valid fixture in engine_source_recheck_is_explicit_and_keeps_model_guards"
            ),
            (0, 0)
        );
        manifest.model.rows += 1;
        write(&manifest);
        assert!(verify(&dir.0, &prover, true).is_err());
    }
}
