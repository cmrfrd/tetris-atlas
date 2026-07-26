//! Reachable (board, bag_mask) closure under the osc-greedy policy vs the switching 7-bag
//! adversary. Adapts tetris_policy/main.rs. The closed reachable set IS the survival invariant;
//! the certificate (v, R) is its compact over-approximation.

use rustc_hash::FxHashSet;
use tetris_game::{TetrisBoard, TetrisPiece};

use crate::engine::{W, heights_i64, max_height, osc};
use crate::lp::Cert;
use crate::policy::{Band, choose_placement, choose_placement_lookahead};

const PIECES: [TetrisPiece; 7] = TetrisPiece::all();
const FULL_MASK: u8 = 0b0111_1111;

#[derive(Clone, Debug)]
pub enum Outcome {
    Closed {
        size: usize,
        max_osc: i64,
        mh: i64,
    },
    Leak {
        state: [i64; W],
        piece_idx: usize,
        size: usize,
    },
    /// Out-of-ball: a valid placement exists but exceeds the osc ball; `successor` is the
    /// closest (min-osc) successor — feed it back to grow R.
    BallLeak {
        successor: [i64; W],
        piece_idx: usize,
        size: usize,
    },
    Exploded {
        size: usize,
        mh: i64,
    },
}

/// Expand the reachable closure under the osc-greedy policy gated by `r_cap`.
///
/// `r_cap` is the integer osc-ball radius (against the integer center `round(v)`).
/// Any successor with `osc(h - round(v)) > r_cap` is treated as a leak (the gate rejects it).
/// On `Closed`, `max_osc` is guaranteed `<= r_cap`.
pub fn closure(cert: &Cert, band: &Band, cap: usize, r_cap: i64, depth: u8) -> Outcome {
    let vi: [i64; W] = {
        let mut out = [0i64; W];
        for j in 0..W {
            out[j] = cert.v[j].round() as i64;
        }
        out
    };
    let centered_osc = |h: &[i64; W]| -> i64 {
        let mut c = [0i64; W];
        for j in 0..W {
            c[j] = h[j] - vi[j];
        }
        osc(&c)
    };

    let root = TetrisBoard::new();
    let mut seen: FxHashSet<(TetrisBoard, u8)> = FxHashSet::default();
    let mut stack: Vec<(TetrisBoard, u8)> = vec![(root, FULL_MASK)];
    seen.insert((root, FULL_MASK));
    // root osc is always <= r_cap (empty board is osc 0); not recounted here.
    let (mut max_osc, mut mh) = (0i64, 0i64);

    while let Some((b, mask)) = stack.pop() {
        for (pi, p) in PIECES.iter().enumerate() {
            let bit = 1u8 << pi;
            if mask & bit == 0 {
                continue;
            }
            let rem = mask & !bit;
            let rem_full = if rem == 0 { FULL_MASK } else { rem };
            match choose_placement_lookahead(&b, *p, rem_full, cert, band, r_cap, depth) {
                None => {
                    // No in-ball placement; check whether ANY valid in-band placement exists
                    // (ignoring the ball). If yes → BallLeak (grow R). If no → hard Leak.
                    // BallLeak relaxed check stays 1-ply (closest out-of-ball successor for R-growth).
                    match choose_placement(&b, *p, cert, band, i64::MAX) {
                        Some((_, h_relax)) => {
                            return Outcome::BallLeak {
                                successor: h_relax,
                                piece_idx: pi,
                                size: seen.len(),
                            };
                        }
                        None => {
                            return Outcome::Leak {
                                state: heights_i64(&b),
                                piece_idx: pi,
                                size: seen.len(),
                            };
                        }
                    }
                }
                Some((nb, h)) => {
                    max_osc = max_osc.max(centered_osc(&h));
                    mh = mh.max(max_height(&h));
                    if seen.insert((nb, rem_full)) {
                        if seen.len() > cap {
                            return Outcome::Exploded {
                                size: seen.len(),
                                mh,
                            };
                        }
                        stack.push((nb, rem_full));
                    }
                }
            }
        }
    }
    Outcome::Closed {
        size: seen.len(),
        max_osc,
        mh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::Cert;
    use crate::policy::Band;

    #[test]
    fn impossibly_tight_band_leaks_immediately() {
        // h_band 0: the empty root cannot place any first piece in-band -> Leak at size 1.
        // r_cap=40 (large) so the leak is due to the tight band, not the osc-ball gate.
        let cert = Cert {
            v: [0.0; W],
            r: 0.0,
            h_max: 0.0,
        };
        let band = Band {
            k_holes: 0,
            h_band: 0,
        };
        match closure(&cert, &band, 1_000, 40, 2) {
            Outcome::Leak { size, .. } => assert_eq!(size, 1),
            other => assert!(false, "expected Leak, got {other:?}"),
        }
    }
}
