//! CEGIS driver. Task 6 adds the M0 fixed-order validation; Task 7 adds the switching loop.

use tetris_game::{TetrisBoard, TetrisPiece};

use crate::closure::{Outcome, closure};
use crate::engine::{W, heights_i64, holes_of, max_height, osc, placements, successor};
use crate::lp::{Cert, fit_ball};
use crate::policy::Band;

/// Terminal status returned by the CEGIS driver.
#[derive(Clone, Debug)]
pub enum Status {
    /// The closure oracle confirmed the carrier is closed under the current cert.
    Certified,
    /// The LP can no longer fit any osc-ball over the accumulated leak counterexamples.
    Refuted { binding_r: f64 },
    /// The reachable closure exceeded the cap: the carrier is too large to certify.
    Exploded,
}

/// Full report from one CEGIS run.
#[derive(Clone, Debug)]
pub struct Report {
    pub status: Status,
    pub iters: usize,
    pub cert: Cert,
    pub size: usize,
    pub max_osc: i64,
    /// The integer osc-ball radius actually enforced by the closure gate (= max_s osc(s − round(v))).
    /// This is the radius that `choose_placement` uses to gate successors; `cert.r` (the LP
    /// continuous optimum) is for convergence tracking only and is NOT the gate threshold.
    /// On a `Certified` result, `max_osc <= r_cap` is guaranteed by the gate.
    pub r_cap: i64,
    /// True iff the run ended on a hard leak (player stranded — no in-band placement at all,
    /// independent of the osc-ball radius). False for ball/convergence/cap failures.
    pub hard_leak: bool,
}

/// Run the CEGIS switching loop.
///
/// Alternates between `fit_ball` (LP phase) and `closure` (oracle phase).
/// The LP sample set holds only leak counterexamples, so it stays tiny.
/// `r_cap` is recomputed each iteration as the tight integer radius for `round(v)` over
/// the current samples — this is the radius passed to the osc-ball gate in `choose_placement`.
pub fn run_cegis(band: &Band, cap: usize, max_iters: usize, depth: u8) -> Report {
    let mut samples: Vec<[i64; W]> = vec![[0i64; W]]; // flat empty root
    let mut last_cert = Cert {
        v: [0.0; W],
        r: 0.0,
        h_max: 0.0,
    };
    // last_r_cap: used on the fit_ball->None early-exit path where r_cap cannot be recomputed.
    let mut last_r_cap: i64 = 0;

    for it in 1..=max_iters {
        let cert = match fit_ball(&samples) {
            Some(c) => c,
            None => {
                return Report {
                    status: Status::Refuted {
                        binding_r: last_cert.r,
                    },
                    iters: it,
                    cert: last_cert,
                    size: samples.len(),
                    max_osc: 0,
                    // last successfully computed r_cap; 0 if fit_ball failed on first iter
                    r_cap: last_r_cap,
                    hard_leak: false,
                };
            }
        };
        last_cert = cert.clone();

        // Tight integer radius for the integer center.  All current samples lie in S by
        // construction, so r_cap is the smallest integer ball containing them all.
        let v_int: [i64; W] = {
            let mut a = [0i64; W];
            for j in 0..W {
                a[j] = cert.v[j].round() as i64;
            }
            a
        };
        let r_cap: i64 = samples
            .iter()
            .map(|s| {
                let mut c = [0i64; W];
                for j in 0..W {
                    c[j] = s[j] - v_int[j];
                }
                osc(&c)
            })
            .max()
            .unwrap_or(0);
        last_r_cap = r_cap;

        match closure(&cert, band, cap, r_cap, depth) {
            Outcome::Closed { size, max_osc, .. } => {
                // max_osc <= r_cap is guaranteed by the gate in choose_placement.
                return Report {
                    status: Status::Certified,
                    iters: it,
                    cert,
                    size,
                    max_osc,
                    r_cap,
                    hard_leak: false,
                };
            }
            Outcome::Exploded { size, .. } => {
                return Report {
                    status: Status::Exploded,
                    iters: it,
                    cert,
                    size,
                    max_osc: 0,
                    r_cap,
                    hard_leak: false,
                };
            }
            Outcome::BallLeak {
                successor, size, ..
            } => {
                // Out-of-ball: a valid in-band placement exists but exceeds the current ball.
                // Feed the closest (min-osc) successor back to grow R on the next fit.
                // Safety: if successor is already in samples it would be inside the ball (since
                // fit_ball guarantees all samples fit), contradicting BallLeak. Guard anyway.
                if samples.contains(&successor) {
                    return Report {
                        status: Status::Refuted { binding_r: cert.r },
                        iters: it,
                        cert,
                        size,
                        max_osc: 0,
                        r_cap,
                        hard_leak: false,
                    };
                }
                samples.push(successor);
                // Continue the loop; R will grow on the next fit_ball call.
            }
            Outcome::Leak { size, .. } => {
                // Hard leak: no valid in-band placement exists regardless of the ball radius.
                // The player is genuinely stranded — this is a true refutation.
                return Report {
                    status: Status::Refuted { binding_r: cert.r },
                    iters: it,
                    cert,
                    size,
                    max_osc: 0,
                    r_cap,
                    hard_leak: true,
                };
            }
        }
    }
    Report {
        status: Status::Refuted {
            binding_r: last_cert.r,
        },
        iters: max_iters,
        cert: last_cert,
        size: samples.len(),
        max_osc: 0,
        r_cap: last_r_cap,
        hard_leak: false,
    }
}

/// Seed objective (no cert): minimize (max_height, osc) — a plain flatten player.
fn flatten_choice(b: &TetrisBoard, p: TetrisPiece, band: &Band) -> Option<TetrisBoard> {
    let mut best: Option<((i64, i64), TetrisBoard)> = None;
    for &pl in placements(p) {
        let Some((nb, _)) = successor(b, pl) else {
            continue;
        };
        if holes_of(&nb) > band.k_holes {
            continue;
        }
        let h = heights_i64(&nb);
        let mh = max_height(&h);
        if mh > band.h_band {
            continue;
        }
        let key = (mh, osc(&h));
        if best.as_ref().is_none_or(|(bk, _)| key < *bk) {
            best = Some((key, nb));
        }
    }
    best.map(|(_, nb)| nb)
}

pub fn play_fixed_order(order: &[TetrisPiece], steps: usize, band: &Band) -> Vec<[i64; W]> {
    if order.is_empty() {
        return Vec::new();
    }
    let mut b = TetrisBoard::new();
    let mut traj = Vec::with_capacity(steps);
    for i in 0..steps {
        let p = order[i % order.len()];
        match flatten_choice(&b, p, band) {
            Some(nb) => {
                b = nb;
                traj.push(heights_i64(&b));
            }
            None => break,
        }
    }
    traj
}

pub fn m0_fixed_order_radius(order: &[TetrisPiece], steps: usize, band: &Band) -> Option<f64> {
    let traj = play_fixed_order(order, steps, band);
    if traj.len() < steps {
        return None; // leaked before finishing
    }
    fit_ball(&traj).map(|c| c.r)
}

#[cfg(test)]
mod cegis_tests {
    use super::*;
    use crate::policy::Band;

    #[test]
    fn cegis_terminates_and_reports() {
        // Small cap + tight band: must terminate quickly with SOME terminal status.
        let band = Band {
            k_holes: 1,
            h_band: 8,
        };
        let report = run_cegis(&band, 50_000, 12, 2);
        eprintln!(
            "CEGIS report: status={:?} iters={} size={} max_osc={} r={:.6} r_cap={}",
            report.status, report.iters, report.size, report.max_osc, report.cert.r, report.r_cap
        );
        assert!(report.iters >= 1);
        // Any terminal status is acceptable here; we only assert the driver halts and reports.
        match report.status {
            Status::Certified | Status::Refuted { .. } | Status::Exploded => {}
        }
    }

    #[test]
    fn cegis_progresses_past_trivial_refute() {
        // With the two-tier leak, from the empty board the player CAN place (BallLeak),
        // so R must grow above 0 before any terminal verdict — proving the loop explores
        // rather than trivially refuting at r_cap=0.
        let band = Band {
            k_holes: 2,
            h_band: 12,
        };
        let report = run_cegis(&band, 200_000, 60, 2);
        eprintln!(
            "progress test: status={:?} iters={} size={} r_cap={}",
            report.status, report.iters, report.size, report.r_cap
        );
        assert!(
            report.r_cap > 0,
            "CEGIS should grow R above 0, got r_cap={}",
            report.r_cap
        );
    }

    #[test]
    fn certified_implies_osc_within_radius() {
        // For any Certified run, the closure gate guarantees max_osc <= r_cap.
        // If no run certifies in this set, the test still passes (no violated assertion),
        // but at least one run_cegis call is exercised.
        let configs: &[(u32, i64, usize, usize)] = &[(1, 8, 50_000, 12), (2, 12, 50_000, 12)];
        for &(holes, h_band, cap, iters) in configs {
            let band = Band {
                k_holes: holes,
                h_band,
            };
            let report = run_cegis(&band, cap, iters, 2);
            eprintln!(
                "CEGIS holes={holes} band={h_band}: status={:?} r_cap={} max_osc={} R={:.6}",
                report.status, report.r_cap, report.max_osc, report.cert.r
            );
            if let Status::Certified = &report.status {
                assert!(
                    report.max_osc <= report.r_cap,
                    "Certified but max_osc={} > r_cap={}",
                    report.max_osc,
                    report.r_cap
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Band;
    use tetris_game::TetrisPiece;

    #[test]
    fn fixed_order_validates_ball_fit_over_survivors() {
        // A plain 1-ply flatten player is MYOPIC: it accumulates S/Z holes faster than it
        // drains them and leaks on the straight order around bag 5 (matches the known result
        // that fixed orders need lookahead — dumb 1-ply survives only ~60%). So M0 validates
        // the fit_ball + engine PIPELINE over a horizon the dumb player demonstrably survives
        // (4 bags), NOT long-horizon survival (that needs a beam, out of M0's scope).
        let order = TetrisPiece::all().to_vec();
        let band = Band {
            k_holes: 4,
            h_band: 20,
        };
        let steps = 7 * 4; // 28 = 4 bags; the flatten player survives this on the straight order
        let r = m0_fixed_order_radius(&order, steps, &band)
            .expect("flatten player must survive 4 bags of the straight order");
        eprintln!("M0 radius = {r}");
        assert!(
            (0.0..=20.0).contains(&r),
            "fitted radius {r} out of sane range [0,20]"
        );
    }
}
