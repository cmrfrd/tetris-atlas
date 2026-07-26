#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! tetris_maxplus_cert — convex (LP) search inside a CEGIS loop for a max-plus
//! piecewise-linear survival certificate (eigen-surface v + radius R + height bound H_max).

mod cegis;
mod closure;
mod engine;
mod lp;
mod policy;

use std::time::Instant;
use tetris_game::TetrisPiece;

use crate::cegis::{Status, m0_fixed_order_radius, run_cegis};
use crate::policy::Band;

fn parse_u64(args: &[String], pfx: &str, default: u64) -> u64 {
    args.iter()
        .find_map(|a| a.strip_prefix(pfx).and_then(|s| s.parse::<u64>().ok()))
        .unwrap_or(default)
}

fn round_v(v: &[f64; engine::W]) -> [i64; engine::W] {
    let mut out = [0i64; engine::W];
    for j in 0..engine::W {
        out[j] = v[j].round() as i64;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .iter()
        .find_map(|a| a.strip_prefix("mode="))
        .unwrap_or("cegis")
        .to_string();
    let band = Band {
        k_holes: parse_u64(&args, "holes=", 1) as u32,
        h_band: parse_u64(&args, "band=", 12) as i64,
    };
    let cap = parse_u64(&args, "cap=", 2) as usize * 1_000_000;
    let iters = parse_u64(&args, "iters=", 30) as usize;
    let depth = parse_u64(&args, "depth=", 2) as u8;

    println!(
        "tetris_maxplus_cert — mode={mode} holes={} band={} cap={cap} iters={iters} depth={depth}",
        band.k_holes, band.h_band
    );
    let t0 = Instant::now();

    match mode.as_str() {
        "m0" => {
            // M0 validates the fit_ball+engine pipeline over a horizon the myopic 1-ply
            // flatten player survives (4 bags); long survival needs a beam (out of scope).
            let order = TetrisPiece::all().to_vec();
            match m0_fixed_order_radius(&order, 7 * 4, &band) {
                Some(r) => {
                    println!("M0 fixed-order OISZTLJ (4 bags): survived, fitted radius R = {r:.3}")
                }
                None => {
                    println!("M0 fixed-order OISZTLJ (4 bags): LEAKED in-band (band too tight?)")
                }
            }
        }
        "cegis" => {
            let rep = run_cegis(&band, cap, iters, depth);
            print!(
                "CEGIS: iters={} depth={depth} |samples-driven| size={} max_osc={} r_cap={} (enforced) R_lp={:.3} H_max={:.3} -> ",
                rep.iters, rep.size, rep.max_osc, rep.r_cap, rep.cert.r, rep.cert.h_max
            );
            match rep.status {
                Status::Certified => {
                    println!(
                        "CERTIFIED (closed carrier; v rounded = {:?})",
                        round_v(&rep.cert.v)
                    )
                }
                // REFUTED means "this osc-greedy policy cannot close this band," not a class-level impossibility.
                Status::Refuted { binding_r } => {
                    let kind = if rep.hard_leak {
                        "hard leak — player stranded"
                    } else {
                        "R hit band / no convergence"
                    };
                    println!("REFUTED (policy-conditional; {kind}) binding_r ~ {binding_r:.3}")
                }
                Status::Exploded => {
                    println!("EXPLODED (carrier > cap; not certifiable at this band)")
                }
            }
        }
        _ => {
            eprintln!("unknown mode '{mode}', defaulting to cegis");
            let rep = run_cegis(&band, cap, iters, depth);
            print!(
                "CEGIS: iters={} depth={depth} |samples-driven| size={} max_osc={} r_cap={} (enforced) R_lp={:.3} H_max={:.3} -> ",
                rep.iters, rep.size, rep.max_osc, rep.r_cap, rep.cert.r, rep.cert.h_max
            );
            match rep.status {
                Status::Certified => {
                    println!(
                        "CERTIFIED (closed carrier; v rounded = {:?})",
                        round_v(&rep.cert.v)
                    )
                }
                // REFUTED means "this osc-greedy policy cannot close this band," not a class-level impossibility.
                Status::Refuted { binding_r } => {
                    let kind = if rep.hard_leak {
                        "hard leak — player stranded"
                    } else {
                        "R hit band / no convergence"
                    };
                    println!("REFUTED (policy-conditional; {kind}) binding_r ~ {binding_r:.3}")
                }
                Status::Exploded => {
                    println!("EXPLODED (carrier > cap; not certifiable at this band)")
                }
            }
        }
    }
    println!("wall = {:.2}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use good_lp::solvers::clarabel::clarabel;
    use good_lp::{Solution, SolverModel, constraint, variable, variables};

    /// Trivial LP: minimize x subject to x >= 3. Optimum is 3.0.
    /// Proves the good_lp + clarabel toolchain links and the modeling API is as expected.
    fn smoke_lp() -> f64 {
        let mut vars = variables!();
        let x = vars.add(variable().min(0.0));
        let sol = vars
            .minimise(x)
            .using(clarabel)
            .with(constraint!(x >= 3.0))
            .solve()
            .expect("smoke LP must solve");
        sol.value(x)
    }

    #[test]
    fn smoke_lp_is_three() {
        assert!((smoke_lp() - 3.0).abs() < 1e-6, "got {}", smoke_lp());
    }
}
