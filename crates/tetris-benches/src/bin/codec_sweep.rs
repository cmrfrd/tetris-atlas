//! Bounded, independently compiled codec benchmarks and reproducible reports.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

const OPERATIONS: [&str; 4] = [
    "encode_rows",
    "encode_native",
    "decode_rows",
    "decode_native",
];
const BENCH_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Selects the fixed grid and the per-process measurement budgets.
#[derive(Parser, Debug, Serialize)]
#[command(about = "Sweep ten-column codecs at heights 2–6 and hole budgets 0–5")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, num_args = 1.., default_values_t = [2, 3, 4, 5, 6])]
    rows: Vec<usize>,
    #[arg(long, num_args = 1.., default_values_t = [0, 1, 2, 3, 4, 5])]
    holes: Vec<usize>,
    #[arg(long, default_value_t = 180.0)]
    compile_timeout: f64,
    #[arg(long, default_value_t = 60.0)]
    benchmark_timeout: f64,
    #[arg(long, default_value_t = 20)]
    sample_size: usize,
    #[arg(long, default_value_t = 0.2)]
    warm_up_time: f64,
    #[arg(long, default_value_t = 0.5)]
    measurement_time: f64,
    #[arg(long, default_value_t = 10.0)]
    cooldown_seconds: f64,
    /// Retry unfinished cases only when source, grid and timing identities match.
    #[arg(long)]
    resume: bool,
}

/// One member of the bounded parameter grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Config {
    rows: usize,
    holes: usize,
}

/// Separates unmeasured cases from genuine construction and validation failures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    #[default]
    Pending,
    Compiling,
    Ok,
    CompileTimeout,
    CountOverflow,
    WorkspaceLimit,
    NodeLimit,
    CompileError,
    ValidationError,
    BenchmarkTimeout,
    BenchmarkError,
    ResultError,
    Interrupted,
}

impl Status {
    /// Returns the stable label used in reports and progress messages.
    fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Compiling => "compiling",
            Self::Ok => "ok",
            Self::CompileTimeout => "compile_timeout",
            Self::CountOverflow => "count_overflow",
            Self::WorkspaceLimit => "workspace_limit",
            Self::NodeLimit => "node_limit",
            Self::CompileError => "compile_error",
            Self::ValidationError => "validation_error",
            Self::BenchmarkTimeout => "benchmark_timeout",
            Self::BenchmarkError => "benchmark_error",
            Self::ResultError => "result_error",
            Self::Interrupted => "interrupted",
        }
    }
}

/// Exact process invocation, wall time and termination result.
#[derive(Debug, Serialize, Deserialize)]
struct ProcessRun {
    command: Vec<String>,
    seconds: f64,
    returncode: Option<i32>,
    timed_out: bool,
    timeout_seconds: f64,
    log: PathBuf,
}

impl ProcessRun {
    /// Returns whether the process completed successfully within its budget.
    fn succeeded(&self) -> bool {
        !self.timed_out && self.returncode == Some(0)
    }
}

/// Owns a process group so errors cannot leave its compiler running.
struct ProcessGroup(Child);

impl ProcessGroup {
    /// Signals the owned group, accepting a group that has already exited.
    fn signal(&mut self, force: bool) -> Result<()> {
        #[cfg(unix)]
        {
            use nix::{
                errno::Errno,
                sys::signal::{Signal, killpg},
                unistd::Pid,
            };
            let pid = Pid::from_raw(i32::try_from(self.0.id())?);
            match killpg(
                pid,
                if force {
                    Signal::SIGKILL
                } else {
                    Signal::SIGTERM
                },
            ) {
                Ok(()) | Err(Errno::ESRCH) => Ok(()),
                Err(error) => Err(error.into()),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = force;
            self.0.kill().context("terminating child")
        }
    }

    /// Reaps promptly after TERM, escalating only while the parent is still alive.
    fn stop(&mut self, grace: Duration) -> Result<ExitStatus> {
        if let Some(status) = self.0.try_wait()? {
            return Ok(status);
        }
        self.signal(false)?;
        let deadline = Instant::now() + grace;
        loop {
            if let Some(status) = self.0.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                self.signal(true)?;
                return self.0.wait().context("reaping terminated compiler");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            // Drop cannot report errors; explicit termination paths return them to the caller.
            drop(self.signal(true));
            drop(self.0.wait());
        }
    }
}

/// Runs one bounded command with a shared stdout/stderr log and cancellation cleanup.
fn run_process(
    command: &mut Command,
    log: &Path,
    timeout: Duration,
    cancelled: &AtomicBool,
) -> Result<ProcessRun> {
    ensure!(!cancelled.load(Ordering::Relaxed), "sweep interrupted");
    let arguments = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    let stream = File::create(log)?;
    command
        .stdout(stream.try_clone()?)
        .stderr(Stdio::from(stream));
    #[cfg(unix)]
    command.process_group(0);
    let started = Instant::now();
    let mut process = ProcessGroup(command.spawn().context("spawning sweep subprocess")?);
    let (status, timed_out) = loop {
        if let Some(status) = process.0.try_wait()? {
            break (status, false);
        }
        if cancelled.load(Ordering::Relaxed) {
            process.stop(Duration::from_secs(2))?;
            bail!("sweep interrupted; completed cases are saved");
        }
        if started.elapsed() >= timeout {
            break (process.stop(Duration::from_secs(2))?, true);
        }
        thread::sleep(Duration::from_millis(20));
    };
    #[cfg(unix)]
    let returncode = status
        .code()
        .or_else(|| status.signal().map(|signal| -signal));
    #[cfg(not(unix))]
    let returncode = status.code();
    Ok(ProcessRun {
        command: arguments,
        seconds: started.elapsed().as_secs_f64(),
        returncode,
        timed_out,
        timeout_seconds: timeout.as_secs_f64(),
        log: log.to_owned(),
    })
}

/// Captures bounded local metadata without making optional host commands mandatory.
fn capture(root: &Path, program: &str, args: &[&str], cancelled: &AtomicBool) -> String {
    let result = (|| -> Result<String> {
        let file = tempfile::NamedTempFile::new()?;
        run_process(
            Command::new(program).args(args).current_dir(root),
            file.path(),
            Duration::from_secs(10),
            cancelled,
        )?;
        Ok(fs::read_to_string(file.path())?.trim().to_owned())
    })();
    result.unwrap_or_else(|error| format!("unavailable: {error}"))
}

/// Atomically replaces a complete JSON checkpoint.
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    file.sync_all()?;
    fs::rename(temporary, path).with_context(|| format!("saving {}", path.display()))
}

/// Reads a typed record, retaining its path in any parse error.
fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(File::open(path)?)
        .with_context(|| format!("reading {}", path.display()))
}

/// Computes the content identity used for compiler inputs and saved executables.
fn fingerprint(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

/// Collects Rust compiler inputs in deterministic order.
fn rust_sources(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            rust_sources(&path, paths)?;
        } else if path.extension().is_some_and(|s| s == "rs") {
            paths.push(path);
        }
    }
    Ok(())
}

/// Binds resume to the codec, benchmark definitions, launcher and workspace inputs.
fn source_identity(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut paths: Vec<_> = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
        .iter()
        .map(|name| root.join(name))
        .collect();
    for name in [
        "tetris-game",
        "tetris-utils",
        "tetris-macros",
        "tetris-codec",
    ] {
        let directory = root.join("crates").join(name);
        rust_sources(&directory.join("src"), &mut paths)?;
        paths.push(directory.join("Cargo.toml"));
    }
    let benches = root.join("crates/tetris-benches");
    for name in [
        "Cargo.toml",
        "src/bin/codec_sweep.rs",
        "src/board_codec/sweep.rs",
        "src/board_codec/runner.rs",
        "src/board_codec/fixtures.rs",
    ] {
        paths.push(benches.join(name));
    }
    if root.join(".cargo").exists() {
        for entry in fs::read_dir(root.join(".cargo"))? {
            let path = entry?.path();
            if path.extension().is_some_and(|s| s == "toml") {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            Ok((
                path.strip_prefix(root)?.display().to_string(),
                fingerprint(&path)?,
            ))
        })
        .collect()
}

/// Creates a separate consumer while inheriting the actual workspace release profile.
fn prepare_project(
    root: &Path,
    directory: &Path,
    name: &str,
    config: Option<Config>,
) -> Result<Command> {
    fs::create_dir_all(directory.join("src"))?;
    let workspace = fs::read_to_string(root.join("Cargo.toml"))?;
    ensure!(
        !workspace.contains("[profile.release."),
        "nested release profiles need explicit sweep support"
    );
    let profile = workspace
        .split_once("[profile.release]")
        .context("missing release profile")?
        .1
        .split("\n[")
        .next()
        .context("missing release settings")?;
    let quoted = |path: PathBuf| -> Result<String> {
        Ok(serde_json::to_string(
            path.to_str().context("non-UTF-8 compiler path")?,
        )?)
    };
    let manifest = format!(
        "[package]\nname = {name:?}\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[workspace]\n\n[dependencies]\ntetris-codec = {{ path = {} }}\ntetris-game = {{ path = {} }}\ncriterion = {{ version = \"=0.5.1\", features = [\"html_reports\"] }}\n\n[profile.release]\n{profile}",
        quoted(root.join("crates/tetris-codec"))?,
        quoted(root.join("crates/tetris-game"))?,
    );
    fs::write(directory.join("Cargo.toml"), manifest)?;
    fs::copy(root.join("Cargo.lock"), directory.join("Cargo.lock"))?;
    let source = if let Some(Config { rows, holes }) = config {
        let kernel = root.join("crates/tetris-benches/src/board_codec/sweep.rs");
        let kernel = kernel.to_str().context("non-UTF-8 kernel path")?;
        format!(
            "#![feature(generic_const_exprs)]\n#![allow(incomplete_features, long_running_const_eval, reason = \"Bounded codec construction uses nightly const evaluation.\")]\n#[path = {kernel:?}]\nmod sweep;\nuse criterion::{{Criterion, criterion_group, criterion_main}};\nfn benchmark(c: &mut Criterion) {{ sweep::run::<{rows}, {holes}>(c); }}\ncriterion_group!(benches, benchmark);\ncriterion_main!(benches);\n"
        )
    } else {
        "fn main() {}\n".to_owned()
    };
    fs::write(directory.join("src/main.rs"), source)?;
    let mut command = Command::new("cargo");
    command
        .current_dir(root)
        .args([
            "build",
            "--release",
            "--offline",
            "--message-format=json",
            "--manifest-path",
        ])
        .arg(directory.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(root.join("target"));
    Ok(command)
}

/// Classifies compiler limits without promoting a timeout to unsupported status.
fn compile_status(build: &ProcessRun) -> Result<Status> {
    if build.timed_out {
        return Ok(Status::CompileTimeout);
    }
    if build.succeeded() {
        return Ok(Status::Compiling);
    }
    let log = fs::read_to_string(&build.log)?;
    Ok(if log.contains("codec count exceeds u64") {
        Status::CountOverflow
    } else if log.contains("codec exceeds the symbolic workspace budget") {
        Status::WorkspaceLimit
    } else if log.contains("codec rule diagram exceeds node budget") {
        Status::NodeLimit
    } else {
        Status::CompileError
    })
}

/// Finds the consumer binary in Cargo's artifact stream.
fn executable(log: &Path) -> Result<PathBuf> {
    for line in fs::read_to_string(log)?.lines() {
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if message["reason"] == "compiler-artifact"
            && message["target"]["kind"] == json!(["bin"])
            && let Some(path) = message["executable"].as_str()
        {
            return Ok(PathBuf::from(path));
        }
    }
    bail!("Cargo produced no consumer executable")
}

/// Validated raw fixture identity and codec storage statistics.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Fixture {
    name: String,
    samples: usize,
    hash: String,
    states: u64,
    bits: u32,
    id_bytes: usize,
    node_bytes: usize,
}

/// Parses the shared kernel's marker, rejecting incomplete or malformed output.
fn fixture_metadata(log: &Path) -> Result<Fixture> {
    let text = fs::read_to_string(log)?;
    let mut parts = text
        .lines()
        .find_map(|line| line.strip_prefix("CODEC_FIXTURE "))
        .context("missing validated fixture metadata")?
        .split_whitespace();
    let name = parts.next().context("missing fixture name")?.to_owned();
    let fields: BTreeMap<_, _> = parts
        .map(|part| part.split_once('=').context("malformed fixture field"))
        .collect::<Result<_>>()?;
    ensure!(fields.len() == 6, "unexpected fixture fields");
    let number =
        |key| -> Result<u64> { Ok(fields.get(key).context("missing fixture field")?.parse()?) };
    let hash = fields
        .get("hash")
        .context("missing fixture hash")?
        .to_string();
    ensure!(
        hash.len() == 16 && u64::from_str_radix(&hash, 16).is_ok(),
        "invalid fixture hash"
    );
    let fixture = Fixture {
        name,
        samples: usize::try_from(number("samples")?)?,
        hash,
        states: number("states")?,
        bits: u32::try_from(number("bits")?)?,
        id_bytes: usize::try_from(number("id_bytes")?)?,
        node_bytes: usize::try_from(number("node_bytes")?)?,
    };
    ensure!(
        fixture.samples == 256 && fixture.states > 0,
        "unexpected fixture size"
    );
    Ok(fixture)
}

/// Criterion's confidence interval, expressed in the same units as its statistic.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Interval {
    confidence_level: f64,
    lower_bound: f64,
    upper_bound: f64,
}

/// One estimate and its sampling uncertainty.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Statistic {
    point_estimate: f64,
    standard_error: f64,
    confidence_interval: Interval,
}

/// Criterion estimates normalized from batch time to nanoseconds per board.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Estimates {
    mean: Statistic,
    median: Statistic,
    std_dev: Statistic,
    median_abs_dev: Statistic,
}

/// Normalized estimates and the number of Criterion samples used.
#[derive(Debug, Serialize, Deserialize)]
struct Measurement {
    n: usize,
    ns_per_board: Estimates,
    boards_per_second: f64,
}

/// Reads all four operations and converts every estimate/interval exactly once.
fn measurements(home: &Path, fixture: &Fixture) -> Result<BTreeMap<String, Measurement>> {
    let mut results = BTreeMap::new();
    for operation in OPERATIONS {
        let directory = home
            .join(format!("board_codec_{}", fixture.name))
            .join(format!("{operation}_256/new"));
        let mut values: Estimates = read_json(&directory.join("estimates.json"))?;
        let sample: Value = read_json(&directory.join("sample.json"))?;
        let n = sample["iters"]
            .as_array()
            .context("missing sample iterations")?
            .len();
        ensure!(n >= 10, "incomplete Criterion samples");
        for stat in [
            &mut values.mean,
            &mut values.median,
            &mut values.std_dev,
            &mut values.median_abs_dev,
        ] {
            let scale = f64::from(u32::try_from(fixture.samples)?);
            stat.point_estimate /= scale;
            stat.standard_error /= scale;
            stat.confidence_interval.lower_bound /= scale;
            stat.confidence_interval.upper_bound /= scale;
        }
        ensure!(values.mean.point_estimate > 0.0, "invalid mean time");
        let boards_per_second = 1e9 / values.mean.point_estimate;
        results.insert(
            operation.to_owned(),
            Measurement {
                n,
                ns_per_board: values,
                boards_per_second,
            },
        );
    }
    Ok(results)
}

/// Checkpoint for one configuration; absent measurements remain absent on failure.
#[derive(Debug, Serialize, Deserialize)]
struct Case {
    #[serde(flatten)]
    config: Config,
    status: Status,
    attempt: usize,
    directory: Option<PathBuf>,
    build: Option<ProcessRun>,
    validation: Option<ProcessRun>,
    benchmark: Option<ProcessRun>,
    fixture: Option<Fixture>,
    measurements: BTreeMap<String, Measurement>,
    binary_sha256: Option<String>,
    binary_bytes: Option<u64>,
    error: Option<String>,
}

impl Case {
    /// Creates an unmeasured configuration without fabricated sizes or timings.
    fn pending(config: Config) -> Self {
        Self {
            config,
            status: Status::Pending,
            attempt: 0,
            directory: None,
            build: None,
            validation: None,
            benchmark: None,
            fixture: None,
            measurements: BTreeMap::new(),
            binary_sha256: None,
            binary_bytes: None,
            error: None,
        }
    }
}

/// Loads existing progress only after the complete experiment identity matches.
fn load_progress(output: &Path, identity: &Value, grid: &[Config]) -> Result<(Value, Vec<Case>)> {
    let manifest: Value = read_json(&output.join("manifest.json"))?;
    ensure!(
        manifest["identity"] == *identity,
        "resume requires identical Rust launcher, sources, toolchain, grid and timing settings; use a fresh output directory for old Python runs"
    );
    let cases: Vec<Case> = read_json(&output.join("results.json"))?;
    ensure!(
        cases
            .iter()
            .map(|case| case.config)
            .eq(grid.iter().copied()),
        "checkpoint grid mismatch"
    );
    Ok((manifest, cases))
}

/// Writes all progress and both human-readable and machine-readable reports.
fn report(output: &Path, cases: &[Case]) -> Result<()> {
    write_json(&output.join("results.json"), &cases)?;
    let mut csv = File::create(output.join("results.csv"))?;
    writeln!(
        csv,
        "rows,holes,status,states,bits,id_bytes,node_bytes,build_seconds,operation,n,mean_ns,median_ns,std_dev_ns,median_abs_dev_ns,mean_ci_low_ns,mean_ci_high_ns,boards_per_second"
    )?;
    let mut markdown = File::create(output.join("report.md"))?;
    writeln!(
        markdown,
        "# Ten-column codec sweep\n\nPreliminary ns/board from 256-board batches; setup is excluded. Build times use shared, prewarmed dependencies. Node bytes exclude arithmetic constants and compiler scratch. Timeouts leave support unknown. See manifest.json, results.json, logs and Criterion samples for provenance and uncertainty.\n\n| Rows | Holes | Status | States | ID bytes | Node bytes | Build s | Encode native | Decode native | Encode rows | Decode rows |\n|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )?;
    for case in cases {
        let sizes = case
            .fixture
            .as_ref()
            .map(|f| format!("{},{},{},{}", f.states, f.bits, f.id_bytes, f.node_bytes))
            .unwrap_or_else(|| ",,,".to_owned());
        let build = case
            .build
            .as_ref()
            .map(|b| format!("{:.3}", b.seconds))
            .unwrap_or_default();
        for operation in OPERATIONS {
            write!(
                csv,
                "{},{},{},{},{},{},",
                case.config.rows,
                case.config.holes,
                case.status.label(),
                sizes,
                build,
                operation
            )?;
            if let Some(m) = case.measurements.get(operation) {
                let e = &m.ns_per_board;
                writeln!(
                    csv,
                    "{},{},{},{},{},{},{},{}",
                    m.n,
                    e.mean.point_estimate,
                    e.median.point_estimate,
                    e.std_dev.point_estimate,
                    e.median_abs_dev.point_estimate,
                    e.mean.confidence_interval.lower_bound,
                    e.mean.confidence_interval.upper_bound,
                    m.boards_per_second
                )?;
            } else {
                writeln!(csv, ",,,,,,,")?;
            }
        }
        write!(
            markdown,
            "| {} | {} | {} | ",
            case.config.rows,
            case.config.holes,
            case.status.label()
        )?;
        if let Some(f) = &case.fixture {
            write!(
                markdown,
                "{} | {} | {} | ",
                f.states, f.id_bytes, f.node_bytes
            )?;
        } else {
            write!(markdown, "— | — | — | ")?;
        }
        write!(
            markdown,
            "{} |",
            if build.is_empty() { "—" } else { &build }
        )?;
        for operation in [
            "encode_native",
            "decode_native",
            "encode_rows",
            "decode_rows",
        ] {
            if let Some(m) = case.measurements.get(operation) {
                write!(markdown, " {:.2} |", m.ns_per_board.median.point_estimate)?;
            } else {
                write!(markdown, " — |")?;
            }
        }
        writeln!(markdown)?;
    }
    Ok(())
}

/// Allocates a new attempt directory without overwriting earlier logs or binaries.
fn next_attempt(root: &Path) -> Result<usize> {
    let mut latest = 0;
    if root.exists() {
        for entry in fs::read_dir(root)? {
            let name = entry?.file_name();
            if let Some(n) = name
                .to_str()
                .and_then(|s| s.strip_prefix("attempt-"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                latest = latest.max(n);
            }
        }
    }
    latest.checked_add(1).context("attempt counter overflow")
}

/// Compiles, validates and measures one configuration using shared Rust kernels.
fn measure_case(
    root: &Path,
    output: &Path,
    args: &Args,
    case: &mut Case,
    cancelled: &AtomicBool,
) -> Result<()> {
    let directory = case
        .directory
        .as_ref()
        .context("missing attempt directory")?
        .clone();
    let Config { rows, holes } = case.config;
    let name = format!("codec-sweep-r{rows}-h{holes}");
    let mut command = prepare_project(root, &directory, &name, Some(case.config))?;
    let build = run_process(
        &mut command,
        &directory.join("build.log"),
        Duration::from_secs_f64(args.compile_timeout),
        cancelled,
    )?;
    case.status = compile_status(&build)?;
    case.build = Some(build);
    if case.status != Status::Compiling {
        return Ok(());
    }
    let binary = directory.join("executable");
    fs::copy(executable(&directory.join("build.log"))?, &binary)?;
    case.binary_sha256 = Some(fingerprint(&binary)?);
    case.binary_bytes = Some(fs::metadata(&binary)?.len());
    let validation = run_process(
        Command::new(&binary).arg("--test").current_dir(root),
        &directory.join("validation.log"),
        Duration::from_secs_f64(args.benchmark_timeout),
        cancelled,
    )?;
    case.status = Status::ValidationError;
    let valid = validation.succeeded();
    case.validation = Some(validation);
    if !valid {
        return Ok(());
    }
    let fixture = fixture_metadata(&directory.join("validation.log"))?;
    let expected = if rows == 4 && holes == 2 {
        "standard_4_2_reachable_shared".to_owned()
    } else {
        format!("standard_{rows}_{holes}")
    };
    ensure!(
        fixture.name == expected,
        "consumer ran the wrong configuration"
    );
    case.fixture = Some(fixture);
    let cooldown = Instant::now();
    while cooldown.elapsed().as_secs_f64() < args.cooldown_seconds {
        ensure!(!cancelled.load(Ordering::Relaxed), "sweep interrupted");
        thread::sleep(Duration::from_millis(20));
    }
    let mut command = Command::new(&binary);
    command
        .current_dir(root)
        .env("CRITERION_HOME", output.join("criterion"))
        .args(["--bench", "--noplot", "--sample-size"])
        .arg(args.sample_size.to_string())
        .arg("--warm-up-time")
        .arg(args.warm_up_time.to_string())
        .arg("--measurement-time")
        .arg(args.measurement_time.to_string())
        .args(["--save-baseline", "grid-baseline"]);
    let benchmark = run_process(
        &mut command,
        &directory.join("benchmark.log"),
        Duration::from_secs_f64(args.benchmark_timeout),
        cancelled,
    )?;
    case.status = if benchmark.timed_out {
        Status::BenchmarkTimeout
    } else {
        Status::BenchmarkError
    };
    let valid = benchmark.succeeded();
    case.benchmark = Some(benchmark);
    if valid {
        let fixture = case.fixture.as_ref().context("missing validated fixture")?;
        ensure!(
            fixture_metadata(&directory.join("benchmark.log"))? == *fixture,
            "fixture changed before timing"
        );
        case.measurements = measurements(&output.join("criterion"), fixture)?;
        case.status = Status::Ok;
    }
    Ok(())
}

/// Validates CLI settings, locks the dataset and runs the bounded grid sequentially.
fn main() -> Result<()> {
    ensure!(cfg!(unix), "codec_sweep requires Unix process groups");
    let mut args = Args::parse();
    args.rows.sort_unstable();
    args.rows.dedup();
    args.holes.sort_unstable();
    args.holes.dedup();
    ensure!(
        args.rows.iter().all(|r| (2..=6).contains(r)) && args.holes.iter().all(|h| *h <= 5),
        "grid must use rows 2–6 and holes 0–5"
    );
    ensure!(
        args.sample_size >= 10,
        "at least ten Criterion samples are required"
    );
    for value in [
        args.compile_timeout,
        args.benchmark_timeout,
        args.warm_up_time,
        args.measurement_time,
    ] {
        ensure!(
            value.is_finite() && value > 0.0 && Duration::try_from_secs_f64(value).is_ok(),
            "positive finite time limits are required"
        );
    }
    ensure!(
        args.cooldown_seconds.is_finite() && (0.0..=60.0).contains(&args.cooldown_seconds),
        "cooldown must be in 0..60 seconds"
    );
    let root = Path::new(BENCH_DIR)
        .parent()
        .and_then(Path::parent)
        .context("missing workspace root")?
        .canonicalize()?;
    fs::create_dir_all(&args.output)?;
    args.output = args.output.canonicalize()?;
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(args.output.join(".sweep.lock"))?;
    lock.try_lock()
        .context("another sweep owns this output directory")?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancelled);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
    let grid: Vec<_> = args
        .rows
        .iter()
        .flat_map(|&rows| args.holes.iter().map(move |&holes| Config { rows, holes }))
        .collect();
    let environment: BTreeMap<_, _> =
        ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"]
            .into_iter()
            .map(|name| (name, std::env::var(name).ok()))
            .collect();
    let identity = json!({
        "grid": grid,
        "sources": source_identity(&root)?,
        "rustc": capture(&root, "rustc", &["-Vv"], &cancelled),
        "platform": capture(&root, "uname", &["-smr"], &cancelled),
        "cpu": capture(&root, "sysctl", &["-n", "machdep.cpu.brand_string"], &cancelled),
        "build_environment": environment,
        "timing": {
            "samples": args.sample_size,
            "warmup": args.warm_up_time,
            "measurement": args.measurement_time,
        },
    });
    ensure!(!cancelled.load(Ordering::Relaxed), "sweep interrupted");
    let (mut manifest, mut cases) = if args.resume {
        load_progress(&args.output, &identity, &grid)?
    } else {
        ensure!(
            fs::read_dir(&args.output)?
                .all(|entry| entry.is_ok_and(|e| e.file_name() == ".sweep.lock")),
            "output must be empty; use --resume for the same experiment"
        );
        (
            json!({
                "identity": identity,
                "head": capture(&root, "git", &["rev-parse", "HEAD"], &cancelled),
                "git_status": capture(&root, "git", &["status", "--short"], &cancelled),
                "invocations": [],
                "dependency_builds": [],
            }),
            grid.into_iter().map(Case::pending).collect(),
        )
    };
    manifest["invocations"]
        .as_array_mut()
        .context("invalid invocation history")?
        .push(json!({
            "unix_seconds": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),
            "arguments": args,
            "cargo": capture(&root, "cargo", &["-Vv"], &cancelled),
            "uptime": capture(&root, "uptime", &[], &cancelled),
            "power": capture(&root, "pmset", &["-g"], &cancelled),
            "battery": capture(&root, "pmset", &["-g", "batt"], &cancelled),
        }));
    write_json(&args.output.join("manifest.json"), &manifest)?;
    report(&args.output, &cases)?;
    println!("Preparing shared dependencies (excluded from per-configuration build time)");
    let warmup = run_process(
        &mut prepare_project(
            &root,
            &args.output.join("dependencies"),
            "codec-sweep-dependencies",
            None,
        )?,
        &args.output.join("dependencies.log"),
        Duration::from_secs_f64(args.compile_timeout.max(300.0)),
        &cancelled,
    )?;
    let valid = warmup.succeeded();
    manifest["dependency_builds"]
        .as_array_mut()
        .context("invalid dependency history")?
        .push(serde_json::to_value(warmup)?);
    write_json(&args.output.join("manifest.json"), &manifest)?;
    ensure!(valid, "dependency preparation failed; see dependencies.log");
    for index in 0..cases.len() {
        if cases[index].status == Status::Ok {
            continue;
        }
        ensure!(
            !cancelled.load(Ordering::Relaxed),
            "sweep interrupted; resume to continue"
        );
        let config = cases[index].config;
        let case_root = args
            .output
            .join("cases")
            .join(format!("r{}-h{}", config.rows, config.holes));
        let attempt = next_attempt(&case_root)?;
        let directory = case_root.join(format!("attempt-{attempt:03}"));
        fs::create_dir_all(&directory)?;
        cases[index] = Case {
            status: Status::Compiling,
            attempt,
            directory: Some(directory.clone()),
            ..Case::pending(config)
        };
        report(&args.output, &cases)?;
        println!(
            "[{}/{}] rows={} holes={}: compiling",
            index + 1,
            cases.len(),
            config.rows,
            config.holes
        );
        if let Err(error) = measure_case(&root, &args.output, &args, &mut cases[index], &cancelled)
        {
            cases[index].status = if cancelled.load(Ordering::Relaxed) {
                Status::Interrupted
            } else {
                Status::ResultError
            };
            cases[index].error = Some(format!("{error:#}"));
        }
        write_json(&directory.join("result.json"), &cases[index])?;
        report(&args.output, &cases)?;
        println!(
            "[{}/{}] rows={} holes={}: {}",
            index + 1,
            cases.len(),
            config.rows,
            config.holes,
            cases[index].status.label()
        );
        ensure!(
            !cancelled.load(Ordering::Relaxed),
            "sweep interrupted; completed cases are saved"
        );
    }
    println!("Results: {}", args.output.join("report.md").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_timeouts_preserve_unknown_support() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let log = directory.path().join("build.log");
        let mut build = ProcessRun {
            command: Vec::new(),
            seconds: 0.1,
            returncode: Some(1),
            timed_out: false,
            timeout_seconds: 0.1,
            log: log.clone(),
        };
        for (diagnostic, expected) in [
            ("codec count exceeds u64", Status::CountOverflow),
            (
                "codec exceeds the symbolic workspace budget",
                Status::WorkspaceLimit,
            ),
            ("codec rule diagram exceeds node budget", Status::NodeLimit),
            ("unrelated compiler failure", Status::CompileError),
        ] {
            fs::write(&log, diagnostic)?;
            assert_eq!(compile_status(&build)?, expected);
            build.timed_out = true;
            assert_eq!(compile_status(&build)?, Status::CompileTimeout);
            build.timed_out = false;
        }
        Ok(())
    }

    /// Writes the same marker emitted by the shared benchmark kernel.
    fn fixture(directory: &Path) -> Result<Fixture> {
        let log = directory.join("validation.log");
        fs::write(
            &log,
            "CODEC_FIXTURE standard_2_0 samples=256 hash=0123456789abcdef states=100 bits=7 id_bytes=2 node_bytes=80\n",
        )?;
        fixture_metadata(&log)
    }

    #[test]
    fn malformed_fixture_metadata_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        assert_eq!(fixture(directory.path())?.states, 100);
        let log = directory.path().join("validation.log");
        for marker in [
            "no marker",
            "CODEC_FIXTURE standard_2_0 samples=256",
            "CODEC_FIXTURE standard_2_0 samples=0 hash=0123456789abcdef states=100 bits=7 id_bytes=2 node_bytes=80",
            "CODEC_FIXTURE standard_2_0 samples=256 hash=invalid states=100 bits=7 id_bytes=2 node_bytes=80",
        ] {
            fs::write(&log, marker)?;
            assert!(fixture_metadata(&log).is_err());
        }
        Ok(())
    }

    #[test]
    fn criterion_statistics_and_intervals_are_scaled_once() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let fixture = fixture(directory.path())?;
        let statistic = json!({"point_estimate": 2560.0, "standard_error": 256.0,
            "confidence_interval": {"confidence_level": 0.95, "lower_bound": 2304.0, "upper_bound": 2816.0}});
        for operation in OPERATIONS {
            let path = directory
                .path()
                .join(format!("board_codec_standard_2_0/{operation}_256/new"));
            fs::create_dir_all(&path)?;
            write_json(
                &path.join("estimates.json"),
                &json!({"mean":statistic, "median":statistic, "std_dev":statistic, "median_abs_dev":statistic}),
            )?;
            write_json(
                &path.join("sample.json"),
                &json!({"iters":vec![1; 20], "times":vec![2560; 20]}),
            )?;
        }
        let result = measurements(directory.path(), &fixture)?;
        assert_eq!(result.len(), 4);
        for measured in result.values() {
            assert_eq!(measured.n, 20);
            for stat in [
                &measured.ns_per_board.mean,
                &measured.ns_per_board.median,
                &measured.ns_per_board.std_dev,
                &measured.ns_per_board.median_abs_dev,
            ] {
                assert_eq!(stat.point_estimate, 10.0);
                assert_eq!(stat.standard_error, 1.0);
                assert_eq!(stat.confidence_interval.lower_bound, 9.0);
                assert_eq!(stat.confidence_interval.upper_bound, 11.0);
                assert_eq!(stat.confidence_interval.confidence_level, 0.95);
            }
            assert_eq!(measured.boards_per_second, 100_000_000.0);
        }
        Ok(())
    }

    #[test]
    fn failed_cases_have_no_fabricated_sizes_or_timings() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let case = Case {
            status: Status::CompileTimeout,
            ..Case::pending(Config { rows: 6, holes: 5 })
        };
        report(directory.path(), &[case])?;
        let csv = fs::read_to_string(directory.path().join("results.csv"))?;
        assert_eq!(csv.lines().count(), 5);
        for line in csv.lines().skip(1) {
            let fields: Vec<_> = line.split(',').collect();
            assert_eq!(fields.len(), 17);
            assert!(fields[3..8].iter().all(|value| value.is_empty()));
            assert!(fields[9..].iter().all(|value| value.is_empty()));
        }
        assert!(
            fs::read_to_string(directory.path().join("report.md"))?
                .contains("compile_timeout | — | — | —")
        );
        Ok(())
    }

    #[test]
    fn incompatible_resume_leaves_existing_progress_intact() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let manifest = directory.path().join("manifest.json");
        let results = directory.path().join("results.json");
        write_json(&manifest, &json!({"identity": {"sources":"old"}}))?;
        write_json(&results, &json!([]))?;
        let before = (fs::read(&manifest)?, fs::read(&results)?);
        assert!(load_progress(directory.path(), &json!({"sources":"new"}), &[]).is_err());
        assert_eq!(before, (fs::read(&manifest)?, fs::read(&results)?));
        Ok(())
    }

    #[test]
    fn retries_preserve_attempts_even_with_gaps() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for name in ["attempt-001", "attempt-003", "other"] {
            fs::create_dir(directory.path().join(name))?;
        }
        assert_eq!(next_attempt(directory.path())?, 4);
        Ok(())
    }

    #[test]
    fn matching_resume_preserves_completed_cases() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let identity = json!({"sources": "same"});
        let config = Config { rows: 2, holes: 0 };
        write_json(
            &directory.path().join("manifest.json"),
            &json!({"identity": identity}),
        )?;
        let case = Case {
            status: Status::Ok,
            attempt: 3,
            ..Case::pending(config)
        };
        report(directory.path(), &[case])?;
        let (_, cases) = load_progress(directory.path(), &identity, &[config])?;
        assert_eq!(cases[0].status, Status::Ok);
        assert_eq!(cases[0].attempt, 3);
        assert!(
            load_progress(directory.path(), &identity, &[Config { rows: 3, holes: 0 }]).is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn cancellation_stops_the_running_subprocess() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(50));
                cancelled.store(true, Ordering::Relaxed);
            });
            let result = run_process(
                Command::new("sh").args(["-c", "sleep 30"]),
                &directory.path().join("cancel.log"),
                Duration::from_secs(30),
                &cancelled,
            );
            assert!(result.is_err_and(|error| error.to_string().contains("interrupted")));
            Ok(())
        })
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_deadline_terminates_and_reaps() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let started = Instant::now();
        let run = run_process(
            Command::new("sh").args(["-c", "sleep 30"]),
            &directory.path().join("timeout.log"),
            Duration::from_millis(50),
            &AtomicBool::new(false),
        )?;
        assert!(run.timed_out && !run.succeeded());
        assert!(run.returncode.is_some_and(|code| code < 0));
        assert!(started.elapsed() < Duration::from_secs(5));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_cleanup_escalates_when_term_is_ignored() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let log = directory.path().join("stubborn.log");
        let stream = File::create(&log)?;
        let mut command = Command::new("sh");
        command
            .args(["-c", "trap '' TERM; echo READY; while :; do sleep 1; done"])
            .stdout(stream)
            .process_group(0);
        let mut process = ProcessGroup(command.spawn()?);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !fs::read_to_string(&log)?.contains("READY") {
            ensure!(Instant::now() < deadline, "child did not become ready");
            thread::sleep(Duration::from_millis(10));
        }
        let status = process.stop(Duration::from_millis(50))?;
        assert_eq!(status.signal(), Some(9));
        assert_eq!(process.stop(Duration::ZERO)?, status);
        Ok(())
    }
}
