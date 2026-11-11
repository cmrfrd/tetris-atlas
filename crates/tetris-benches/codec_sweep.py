#!/usr/bin/env python3
"""Measure the bounded ten-column codec grid without compiling every case together."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OPERATIONS = ("encode_rows", "encode_native", "decode_rows", "decode_native")


def save_json(path, value):
    """Replace a checkpoint only after its complete JSON has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(command, log, timeout, *, cwd=ROOT, env=None):
    """Bound a command and terminate its entire process group on timeout."""
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
                break
            if not timed_out:
                raise
    return {
        "command": [str(part) for part in command], "cwd": str(cwd),
        "seconds": time.monotonic() - started, "returncode": process.returncode,
        "timed_out": timed_out, "timeout_seconds": timeout, "log": str(log),
    }


def capture(command):
    """Record available host and tool information without requiring macOS."""
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def source_identity():
    """Fingerprint the compiler inputs and shared benchmark definitions for resume."""
    paths = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    for crate in ("tetris-game", "tetris-utils", "tetris-macros", "tetris-codec"):
        paths += sorted((ROOT / "crates" / crate).rglob("*.rs"))
        paths.append(ROOT / "crates" / crate / "Cargo.toml")
    paths += sorted((ROOT / ".cargo").glob("*.toml"))
    paths += [HERE / "codec_sweep.py", HERE / "codec_sweep.rs"]
    paths += [HERE / "src/board_codec" / name for name in ("fixtures.rs", "runner.rs")]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def prepare_project(directory, name, rows=None, holes=None):
    """Create an isolated Cargo consumer with the workspace's release settings."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = f'''[package]
name = "{name}"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
tetris-codec = {{ path = {json.dumps(str(ROOT / 'crates/tetris-codec'))} }}
tetris-game = {{ path = {json.dumps(str(ROOT / 'crates/tetris-game'))} }}
criterion = {{ version = "=0.5.1", features = ["html_reports"] }}

[profile.release]
debug = true
lto = "fat"
codegen-units = 1
opt-level = 3
'''
    (directory / "Cargo.toml").write_text(manifest)
    shutil.copy2(ROOT / "Cargo.lock", directory / "Cargo.lock")
    source = "fn main() {}\n"
    if rows is not None:
        source = f'''#![feature(generic_const_exprs)]
#![allow(incomplete_features, long_running_const_eval)]
#[path = {json.dumps(str(HERE / 'codec_sweep.rs'))}]
mod sweep;
use criterion::{{Criterion, criterion_group, criterion_main}};
fn benchmark(c: &mut Criterion) {{ sweep::run::<{rows}, {holes}>(c); }}
criterion_group!(benches, benchmark);
criterion_main!(benches);
'''
    (directory / "src").mkdir(exist_ok=True)
    (directory / "src/main.rs").write_text(source)
    return [
        "cargo", "build", "--release", "--offline", "--message-format=json",
        "--manifest-path", str(directory / "Cargo.toml"),
        "--target-dir", str(ROOT / "target"),
    ]


def compile_status(build):
    """Keep timeouts and unclassified compiler failures distinct from capacity limits."""
    if build["timed_out"]:
        return "compile_timeout"
    if build["returncode"] == 0:
        return "compiled"
    text = Path(build["log"]).read_text()
    if "codec count exceeds u64" in text:
        return "count_overflow"
    if "codec exceeds the symbolic workspace budget" in text:
        return "workspace_limit"
    if "codec rule diagram exceeds node budget" in text:
        return "node_limit"
    return "compile_error"


def executable(build):
    """Read Cargo's exact executable path instead of guessing its filename."""
    for line in Path(build["log"]).read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") == "compiler-artifact" and message.get("executable"):
            return Path(message["executable"])
    raise ValueError("Cargo produced no executable artifact")


def fixture_metadata(log):
    """Read the permanent kernel's validated fixture identity and codec sizes."""
    match = re.search(r"CODEC_FIXTURE (\S+) (.+)", log.read_text())
    if not match:
        raise ValueError("missing validated fixture metadata")
    fields = dict(field.split("=", 1) for field in match[2].split())
    expected = {"samples", "hash", "states", "bits", "id_bytes", "node_bytes"}
    if fields.keys() != expected or int(fields["samples"]) != 256:
        raise ValueError("unexpected fixture metadata")
    return {"name": match[1], **{
        key: value if key == "hash" else int(value) for key, value in fields.items()
    }}


def estimates(home, fixture):
    """Convert Criterion's batch estimates to ns/board, retaining confidence intervals."""
    measurements = {}
    for operation in OPERATIONS:
        folder = home / f"board_codec_{fixture['name']}" / f"{operation}_256" / "new"
        raw = json.loads((folder / "estimates.json").read_text())
        samples = json.loads((folder / "sample.json").read_text())
        converted = {"n": len(samples["iters"])}
        for statistic in ("mean", "median", "std_dev", "median_abs_dev"):
            value = raw[statistic]
            interval = value["confidence_interval"]
            converted[statistic + "_ns"] = value["point_estimate"] / fixture["samples"]
            converted[statistic + "_ci_ns"] = [
                interval[bound] / fixture["samples"] for bound in ("lower_bound", "upper_bound")
            ]
        converted["boards_per_second"] = 1e9 / converted["mean_ns"]
        measurements[operation] = converted
    return measurements


def report(output, results):
    """Checkpoint all configurations and export a compact table plus detailed CSV."""
    save_json(output / "results.json", results)
    columns = ["rows", "holes", "status", "states", "bits", "id_bytes", "node_bytes",
               "build_seconds", "operation", "n", "mean_ns", "median_ns", "std_dev_ns",
               "median_abs_dev_ns", "mean_ci_low_ns", "mean_ci_high_ns", "boards_per_second"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for operation in OPERATIONS:
                measurement = result.get("measurements", {}).get(operation, {})
                row = {key: result[key] for key in ("rows", "holes", "status")}
                row.update({key: result.get("fixture", {}).get(key, "")
                            for key in ("states", "bits", "id_bytes", "node_bytes")})
                row.update({key: measurement.get(key, "") for key in columns if key in measurement})
                row.update(operation=operation, build_seconds=result.get("build", {}).get("seconds", ""))
                row["mean_ci_low_ns"], row["mean_ci_high_ns"] = measurement.get("mean_ci_ns", ["", ""])
                writer.writerow(row)
    lines = ["# Ten-column codec sweep", "",
             "Preliminary fixed-fixture timings in ns/board. Setup is excluded; compile times use shared, prewarmed dependencies.",
             "Node bytes exclude arithmetic constants and compiler scratch. Timeouts leave support unknown.",
             "See manifest.json, results.json, logs and Criterion samples for source/host provenance, dispersion and intervals.", "",
             "| Rows | Holes | Status | States | ID bytes | Node bytes | Build s | Encode native | Decode native | Encode rows | Decode rows |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        fixture = result.get("fixture", {})
        values = [str(result["rows"]), str(result["holes"]), result["status"]]
        values += [str(fixture.get(key, "—")) for key in ("states", "id_bytes", "node_bytes")]
        build = result.get("build", {}).get("seconds")
        values += [f"{build:.1f}" if build is not None else "—"]
        for operation in ("encode_native", "decode_native", "encode_rows", "decode_rows"):
            value = result.get("measurements", {}).get(operation, {}).get("median_ns")
            values.append(f"{value:.2f}" if value is not None else "—")
        lines.append("| " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, choices=range(2, 7), default=list(range(2, 7)))
    parser.add_argument("--holes", nargs="+", type=int, choices=range(6), default=list(range(6)))
    parser.add_argument("--compile-timeout", type=float, default=180)
    parser.add_argument("--benchmark-timeout", type=float, default=60)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--warm-up-time", type=float, default=0.2)
    parser.add_argument("--measurement-time", type=float, default=0.5)
    parser.add_argument("--cooldown-seconds", type=float, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 10 or min(args.compile_timeout, args.benchmark_timeout,
                                   args.warm_up_time, args.measurement_time) <= 0 or not 0 <= args.cooldown_seconds <= 60:
        parser.error("positive time limits, at least ten samples and cooldown in 0..60 are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "columns": 10, "rows": sorted(set(args.rows)), "holes": sorted(set(args.holes)),
        "sources": source_identity(), "rustc": capture(["rustc", "-Vv"]),
        "timing": {key: getattr(args, key) for key in ("sample_size", "warm_up_time", "measurement_time")},
        "build_environment": {key: os.environ.get(key) for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET")},
        "platform": platform.platform(), "host": capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            parser.error("resume requires identical sources, toolchain, grid and timing settings")
    else:
        if any(output.iterdir()):
            parser.error("output must be empty; use --resume for the same experiment")
        manifest = {"identity": identity, "head": capture(["git", "rev-parse", "HEAD"]),
                    "git_status": capture(["git", "status", "--short"]), "invocations": []}
    manifest["invocations"].append({
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arguments": vars(args) | {"output": str(output)},
        "platform": platform.platform(), "machine": platform.machine(),
        "cargo": capture(["cargo", "-Vv"]),
        "environment": {key: os.environ.get(key) for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET")},
        "host": capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "uptime": capture(["uptime"]), "power": capture(["pmset", "-g"]),
        "battery": capture(["pmset", "-g", "batt"]),
    })
    save_json(manifest_path, manifest)
    results = json.loads((output / "results.json").read_text()) if args.resume else [
        {"rows": r, "holes": h, "status": "pending"} for r in identity["rows"] for h in identity["holes"]
    ]
    report(output, results)
    print("Preparing shared dependencies (excluded from per-configuration build time)", flush=True)
    command = prepare_project(output / "dependencies", "codec-sweep-dependencies")
    warmup = run(command, output / "dependencies.log", max(300, args.compile_timeout))
    manifest.setdefault("dependency_builds", []).append(warmup)
    save_json(manifest_path, manifest)
    if warmup["timed_out"] or warmup["returncode"] != 0:
        raise SystemExit("dependency preparation failed; see dependencies.log")
    for index, result in enumerate(results, 1):
        if result["status"] == "ok":
            continue
        r, h = result["rows"], result["holes"]
        case_root = output / "cases" / f"r{r}-h{h}"
        attempt = len(list(case_root.glob("attempt-*"))) + 1
        case = case_root / f"attempt-{attempt:03d}"
        print(f"[{index}/{len(results)}] rows={r} holes={h}: compiling", flush=True)
        result.clear()
        result.update(rows=r, holes=h, status="compiling", attempt=attempt, case=str(case))
        command = prepare_project(case, f"codec-sweep-r{r}-h{h}", r, h)
        result["build"] = run(command, case / "build.log", args.compile_timeout)
        result["status"] = compile_status(result["build"])
        try:
            if result["status"] == "compiled":
                binary = case / "executable"
                shutil.copy2(executable(result["build"]), binary)
                result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
                result["binary_bytes"] = binary.stat().st_size
                smoke = run([str(binary), "--test"], case / "validation.log", args.benchmark_timeout)
                result["validation"] = smoke
                result["status"] = "validation_error"
                if not smoke["timed_out"] and smoke["returncode"] == 0:
                    result["fixture"] = fixture_metadata(case / "validation.log")
                    time.sleep(args.cooldown_seconds)
                    env = dict(os.environ, CRITERION_HOME=str(output / "criterion"))
                    command = [str(binary), "--bench", "--noplot", "--sample-size", str(args.sample_size),
                               "--warm-up-time", str(args.warm_up_time), "--measurement-time", str(args.measurement_time),
                               "--save-baseline", "grid-baseline"]
                    benchmark = run(command, case / "benchmark.log", args.benchmark_timeout, env=env)
                    result["benchmark"] = benchmark
                    result["status"] = "benchmark_timeout" if benchmark["timed_out"] else "benchmark_error"
                    if not benchmark["timed_out"] and benchmark["returncode"] == 0:
                        if fixture_metadata(case / "benchmark.log") != result["fixture"]:
                            raise ValueError("fixture changed between validation and timing")
                        result["measurements"] = estimates(output / "criterion", result["fixture"])
                        result["status"] = "ok"
        except (ValueError, KeyError, FileNotFoundError) as error:
            result.update(status="result_error", error=str(error))
        save_json(case / "result.json", result)
        report(output, results)
        print(f"[{index}/{len(results)}] rows={r} holes={h}: {result['status']} ({result['build']['seconds']:.1f}s build)", flush=True)
    print(f"Results: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
