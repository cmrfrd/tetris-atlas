//! osc-greedy player: home the surface toward the candidate eigen-surface v, staying in-band.

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::engine::{W, heights_i64, holes_of, max_height, osc, placements, successor};
use crate::lp::Cert;

const PIECES: [TetrisPiece; 7] = TetrisPiece::all();
const FULL_MASK: u8 = 0b0111_1111;

#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub k_holes: u32,
    pub h_band: i64,
}

/// Among in-band, non-losing placements of `p`, pick the one minimizing osc(succ - round(v)),
/// tie-broken by lower max-height, subject to the hard osc-ball gate `osc <= r_cap`.
/// None = leak (no qualifying placement).
pub fn choose_placement(
    b: &TetrisBoard,
    p: TetrisPiece,
    cert: &Cert,
    band: &Band,
    r_cap: i64,
) -> Option<(TetrisBoard, [i64; W])> {
    let vi: [i64; W] = {
        let mut out = [0i64; W];
        for j in 0..W {
            out[j] = cert.v[j].round() as i64;
        }
        out
    };
    let mut best: Option<(i64, i64, TetrisBoard, [i64; W])> = None; // (osc_key, height_key, board, heights)
    for &pl in placements(p) {
        let Some((nb, _cleared)) = successor(b, pl) else {
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
        let mut centered = [0i64; W];
        for j in 0..W {
            centered[j] = h[j] - vi[j];
        }
        let o = osc(&centered);
        // Hard osc-ball gate: reject successors outside the integer osc-ball.
        if o > r_cap {
            continue;
        }
        let key = (o, mh);
        let take = match &best {
            None => true,
            Some((bo, bh, _, _)) => key < (*bo, *bh),
        };
        if take {
            best = Some((key.0, key.1, nb, h));
        }
    }
    best.map(|(_, _, nb, h)| (nb, h))
}

/// In-ball + in-band + non-losing successor of placing `pl` on `b`, else None.
fn ball_successor(
    b: &TetrisBoard,
    pl: TetrisPiecePlacement,
    vi: &[i64; W],
    r_cap: i64,
    band: &Band,
) -> Option<(TetrisBoard, [i64; W])> {
    let (nb, _) = successor(b, pl)?; // None if it tops out
    if holes_of(&nb) > band.k_holes {
        return None;
    }
    let h = heights_i64(&nb);
    if max_height(&h) > band.h_band {
        return None;
    }
    let mut c = [0i64; W];
    for j in 0..W {
        c[j] = h[j] - vi[j];
    }
    if osc(&c) > r_cap {
        return None; // osc-ball gate
    }
    Some((nb, h))
}

/// True if, from board `b` with current bag `mask`, the player can keep the surface in the ball
/// for `depth` plies vs the switching adversary (adversary = AND over all bag pieces; player = OR
/// over placements). depth==0 is trivially true. Refills the bag to FULL_MASK when it empties.
fn survives(b: &TetrisBoard, mask: u8, vi: &[i64; W], r_cap: i64, band: &Band, depth: u8) -> bool {
    if depth == 0 {
        return true;
    }
    let mask = if mask == 0 { FULL_MASK } else { mask };
    for (pi, p) in PIECES.iter().enumerate() {
        let bit = 1u8 << pi;
        if mask & bit == 0 {
            continue;
        }
        let rem = mask & !bit;
        let answered =
            placements(*p)
                .iter()
                .any(|pl| match ball_successor(b, *pl, vi, r_cap, band) {
                    Some((nb, _)) => survives(&nb, rem, vi, r_cap, band, depth - 1),
                    None => false,
                });
        if !answered {
            return false; // adversary found a piece the player can't answer in-ball
        }
    }
    true
}

/// Depth-`depth` robust lookahead. The player must place `p` now; `rem_mask` is the bag AFTER p.
/// Among in-ball placements of `p`, prefer those from which the player `survives(depth-1)`, then
/// lower osc, then lower max-height. None only if NO in-ball placement of `p` exists (a leak).
/// NOTE: depth==1 reduces exactly to the 1-ply min-osc behavior (survives(0)==true for all).
pub fn choose_placement_lookahead(
    b: &TetrisBoard,
    p: TetrisPiece,
    rem_mask: u8,
    cert: &Cert,
    band: &Band,
    r_cap: i64,
    depth: u8,
) -> Option<(TetrisBoard, [i64; W])> {
    let mut vi = [0i64; W];
    for j in 0..W {
        vi[j] = cert.v[j].round() as i64;
    }
    // Flat 5-tuple key: (NOT survives, osc, max_height, board, heights).
    // Sorted ascending: survivors first, then low osc, then low max-height.
    let mut best: Option<(bool, i64, i64, TetrisBoard, [i64; W])> = None;
    for &pl in placements(p) {
        if let Some((nb, h)) = ball_successor(b, pl, &vi, r_cap, band) {
            let s = survives(&nb, rem_mask, &vi, r_cap, band, depth.saturating_sub(1));
            let mut c = [0i64; W];
            for j in 0..W {
                c[j] = h[j] - vi[j];
            }
            let (not_s, osc_v, mh) = (!s, osc(&c), max_height(&h));
            if best
                .as_ref()
                .is_none_or(|(bs, bo, bh, _, _)| (not_s, osc_v, mh) < (*bs, *bo, *bh))
            {
                best = Some((not_s, osc_v, mh, nb, h));
            }
        }
    }
    best.map(|(_, _, _, nb, h)| (nb, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::Cert;
    use tetris_game::{TetrisBoard, TetrisPiece};

    fn flat_cert() -> Cert {
        Cert {
            v: [0.0; W],
            r: 20.0,
            h_max: 20.0,
        }
    }

    #[test]
    fn picks_a_flat_keeping_move_for_o_on_empty() {
        let b = TetrisBoard::new();
        let band = Band {
            k_holes: 1,
            h_band: 20,
        };
        // r_cap=40 (large): gate is permissive; placement should succeed via band filter.
        let (_, h) = choose_placement(&b, TetrisPiece::O_PIECE, &flat_cert(), &band, 40)
            .expect("O on empty has in-band placements");
        // O leaves osc 2 (a 2-tall bump) no matter where; just assert it is finite & in band.
        assert!(max_height(&h) <= 20);
    }

    #[test]
    fn tight_band_can_force_a_leak() {
        // With h_band = 0 nothing can be placed (every placement raises height), so it leaks.
        // r_cap=40 (large): the leak is due to the band constraint, not the osc-ball gate.
        let b = TetrisBoard::new();
        let band = Band {
            k_holes: 0,
            h_band: 0,
        };
        assert!(choose_placement(&b, TetrisPiece::I_PIECE, &flat_cert(), &band, 40).is_none());
    }

    #[test]
    fn gate_rejects_out_of_ball_successor() {
        // cert with v=[0.0;W], loose band (k_holes=4, h_band=20), r_cap=0.
        // Every O placement on empty board places a 2-wide, 2-tall block, leaving
        // two columns at height 2 and the rest at 0: osc = 2 > 0 = r_cap.
        // The osc-ball gate (not the band) must reject them all.
        // Without FIX 1 (no gate), choose_placement would return Some here.
        let b = TetrisBoard::new();
        let cert = Cert {
            v: [0.0; W],
            r: 0.0,
            h_max: 20.0,
        };
        let band = Band {
            k_holes: 4,
            h_band: 20,
        };
        assert!(
            choose_placement(&b, TetrisPiece::O_PIECE, &cert, &band, 0).is_none(),
            "osc-ball gate r_cap=0 must reject all O placements (osc > 0)"
        );
    }

    #[test]
    fn lookahead_returns_in_ball_move_on_empty() {
        use crate::lp::Cert;
        let cert = Cert {
            v: [0.0; W],
            r: 20.0,
            h_max: 20.0,
        };
        let band = Band {
            k_holes: 2,
            h_band: 12,
        };
        // generous ball so an in-ball placement exists; depth 2 must still return Some.
        let got = choose_placement_lookahead(
            &TetrisBoard::new(),
            TetrisPiece::O_PIECE,
            FULL_MASK,
            &cert,
            &band,
            5,
            2,
        );
        assert!(
            got.is_some(),
            "depth-2 lookahead should find an in-ball O placement on empty"
        );
    }
}
