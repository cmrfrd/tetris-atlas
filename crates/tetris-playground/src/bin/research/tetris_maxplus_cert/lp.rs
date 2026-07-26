//! The certificate LP: smallest common osc-ball over a sample set.

use good_lp::solvers::clarabel::clarabel;
use good_lp::{Expression, Solution, SolverModel, constraint, variable, variables};

use crate::engine::W;

#[derive(Clone, Debug)]
pub struct Cert {
    pub v: [f64; W],
    /// LP continuous-optimum radius (min over v of max_s osc(s−v)); for convergence tracking only.
    /// The ENFORCED gate radius is the integer r_cap computed in run_cegis, NOT this value.
    pub r: f64,
    pub h_max: f64,
}

/// Minimize R s.t. every sample lies in the osc-ball {h : osc(h - v) <= R} and within H_max <= 20.
///
/// The convexity rationale: each h^s is a constant, so every constraint is linear in
/// (v, R, H_max, M_s, m_s). Minimizing R drives M_s -> max_j(h^s_j - v_j) and
/// m_s -> min_j(...), so M_s - m_s = osc(h^s - v). The objective is therefore
/// min_v max_s osc(h^s - v) — the tightest common osc-ball.
///
/// `h_max` is pinned after solving to the tight, deterministic height bound
/// (the maximum column height seen in `samples`, capped at 20.0) rather than the
/// solver's arbitrary feasible value.
pub fn fit_ball(samples: &[[i64; W]]) -> Option<Cert> {
    if samples.is_empty() {
        // Degenerate: the empty board fits trivially.
        return Some(Cert {
            v: [0.0; W],
            r: 0.0,
            h_max: 0.0,
        });
    }

    let mut vars = variables!();
    // Free shift variables v[0..W]; gauge-fix v[0] == 0.
    let v: Vec<_> = (0..W).map(|_| vars.add(variable())).collect();
    // R >= 0 (the osc radius).
    let r = vars.add(variable().min(0.0));
    // H_max in [0, 20] (the height bound).
    let h_max = vars.add(variable().min(0.0).max(20.0));
    // Per-sample aux: m_up = M_s (upper), m_lo = m_s (lower), both free.
    // ALL variables must be added to `vars` BEFORE calling `.minimise`.
    let aux: Vec<(good_lp::Variable, good_lp::Variable)> = samples
        .iter()
        .map(|_| (vars.add(variable()), vars.add(variable())))
        .collect();

    let mut model = vars.minimise(Expression::from(r)).using(clarabel);
    // Gauge fix: v_0 = 0 (osc is invariant to a uniform shift of v).
    model = model.with(constraint!(v[0] == 0.0));

    for (s, &(m_up, m_lo)) in samples.iter().zip(aux.iter()) {
        for j in 0..W {
            let hsj = s[j] as f64;
            // M_s >= h_sj - v_j  <=>  M_s + v_j >= h_sj
            model = model.with(constraint!(m_up + v[j] >= hsj));
            // m_s <= h_sj - v_j  <=>  m_s + v_j <= h_sj
            model = model.with(constraint!(m_lo + v[j] <= hsj));
            // absolute-height face
            model = model.with(constraint!(h_max >= hsj));
        }
        // osc(h_s - v) = M_s - m_s <= R
        model = model.with(constraint!(m_up - m_lo <= r));
    }

    let sol = model.solve().ok()?;
    let mut v_out = [0.0f64; W];
    for j in 0..W {
        v_out[j] = sol.value(v[j]);
    }
    let r_out = sol.value(r);
    // Check the infeasibility guard against the solver value BEFORE pinning.
    // (Over-tall samples make the LP infeasible so sol is already None above,
    //  but this guard catches any numerical slack from the solver.)
    let hmax_solver = sol.value(h_max);
    if hmax_solver > 20.0 + 1e-6 {
        return None;
    }
    // Pin h_max to the tight, deterministic height bound: max column height seen in samples,
    // capped at 20.0.  The solver's feasible value is arbitrary within [0, 20].
    let hmax_out = (samples
        .iter()
        .map(|s| s.iter().copied().max().unwrap_or(0))
        .max()
        .unwrap_or(0) as f64)
        .min(20.0);
    Some(Cert {
        v: v_out,
        r: r_out,
        h_max: hmax_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_flat_surfaces_fit_radius_zero() {
        // Both surfaces are flat (osc 0) at different absolute heights:
        // a single v can center both with R = 0.
        let s1 = [2i64; W];
        let s2 = [5i64; W];
        let c = fit_ball(&[s1, s2]).expect("flat surfaces must fit");
        assert!(c.r < 1e-6, "R = {}", c.r);
        // h_max is now pinned to the sample max (5), not the solver value.
        assert!(
            c.h_max >= 5.0 - 1e-6 && c.h_max <= 20.0 + 1e-6,
            "H_max = {}",
            c.h_max
        );
    }

    #[test]
    fn rough_surface_needs_radius_at_least_its_osc() {
        // Two orthogonal spikes: s1 has a peak at col 0, s2 has a peak at col 1.
        // Proof that R >= 6 for ANY free-v choice (gauge v[0] = 0):
        //   From col 0 of s1: M_1 >= 6;  from col 1 of s1: m_1 <= -v_1.
        //   From col 1 of s2: M_2 >= 6-v_1;  from col 0 of s2: m_2 <= 0.
        //   So (M_1 - m_1) + (M_2 - m_2) >= (6 + v_1) + (6 - v_1) = 12 => R >= 6.
        let mut s1 = [0i64; W];
        let mut s2 = [0i64; W];
        s1[0] = 6;
        s2[1] = 6;
        let c = fit_ball(&[s1, s2]).expect("two-spike samples must fit");
        assert!(c.r >= 6.0 - 1e-6, "R = {} should be >= 6", c.r);
    }

    #[test]
    fn over_tall_sample_is_infeasible() {
        // A column at height 25 cannot satisfy H_max <= 20.
        let mut s = [0i64; W];
        s[3] = 25;
        assert!(fit_ball(&[s]).is_none());
    }
}
