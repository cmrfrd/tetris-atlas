//! Allocation instrumentation only. Timings from this executable are not evidence.
use anyhow::{Result, ensure};
use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, Read},
};
use tetris_board_set::BoardSet;

mod support;

use support::alloc_counter;

#[global_allocator]
static ALLOC: alloc_counter::Counter = alloc_counter::Counter;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "board_set_memory FIXTURE BACKEND reserved|growing"
    );
    let mut reader = BufReader::new(File::open(&args[1])?);
    let mut magic = [0; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic == b"BSMIC001", "bad fixture");
    reader.read_exact(&mut magic)?;
    let n = usize::try_from(u64::from_le_bytes(magic))?;
    let mut boards = Vec::with_capacity(n);
    let mut fresh = Vec::with_capacity(n);
    for _ in 0..n {
        let mut key = [0; 10];
        for col in &mut key {
            let mut b = [0; 4];
            reader.read_exact(&mut b)?;
            *col = u32::from_le_bytes(b);
        }
        let mut flags = [0; 2];
        reader.read_exact(&mut flags)?;
        ensure!(flags[0] <= 1 && flags[1] <= 1, "invalid flags");
        boards.push(key);
        fresh.push(flags[0]);
    }
    ensure!(reader.read(&mut [0])? == 0, "trailing input");
    drop(reader);
    let mut expected = boards.clone();
    expected.sort_unstable();
    expected.dedup();
    let capacity = match args[3].as_str() {
        "reserved" => expected.len(),
        "growing" => 0,
        _ => anyhow::bail!("invalid policy"),
    };
    let before = alloc_counter::live();
    alloc_counter::begin();
    macro_rules! measure {
        ($set:expr) => {{
            let mut set = $set;
            let object = size_of_val(&set);
            for (&key, &flag) in boards.iter().zip(&fresh) {
                ensure!(set.insert(key) == (flag != 0), "insert mismatch");
            }
            ensure!(set.len() == expected.len(), "size mismatch");
            for key in &boards {
                ensure!(set.contains(key), "missing key");
            }
            let retained = alloc_counter::live() - before;
            let peak = alloc_counter::peak() - before;
            let (allocations, reallocations, realloc_moves, realloc_moved_bytes) = alloc_counter::requests();
            // Reserve common validation output before measuring iterator scratch.
            // This keeps on-demand ordering allocations visible and excludes the
            // same output buffer for every collection implementation.
            let mut actual: Vec<[u32; 10]> = Vec::with_capacity(expected.len());
            let iteration_before = alloc_counter::live();
            alloc_counter::begin();
            let iterator = set.iter();
            let iteration_retained = alloc_counter::live() - iteration_before;
            actual.extend(iterator.map(|key| *std::borrow::Borrow::<[u32; 10]>::borrow(&key)));
            let iteration_peak = alloc_counter::peak() - iteration_before;
            let peak_with_iteration = peak.max(retained + iteration_peak);
            actual.sort_unstable();
            ensure!(actual == expected, "iteration mismatch");
            drop(actual);
            set.clear();
            set.clear();
            ensure!(set.len() == 0, "clear mismatch");
            let after_clear = alloc_counter::live() - before;
            drop(set);
            ensure!(alloc_counter::live() == before, "leaked allocations");
            println!("{}", serde_json::json!({
                "backend": args[2],
                "policy": args[3],
                "records": n,
                "unique": expected.len(),
                "object_bytes": object,
                "build_allocations": allocations,
                "build_reallocations": reallocations,
                "build_realloc_moves": realloc_moves,
                "build_realloc_moved_bytes": realloc_moved_bytes,
                "retained_bytes": retained,
                "peak_requested_bytes": peak,
                "after_clear_bytes": after_clear,
                "iteration_retained_bytes": iteration_retained,
                "iteration_scratch_peak_bytes": iteration_peak,
                "peak_with_iteration_scratch_bytes": peak_with_iteration,
            }));
        }};
    }
    match args[2].as_str() {
        "trie" => measure!(BoardSet::<10>::with_capacity(capacity)),
        "std" => measure!(HashSet::<[u32; 10]>::with_capacity(capacity)),
        _ => anyhow::bail!("unknown backend"),
    }
    Ok(())
}
