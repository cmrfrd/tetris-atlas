//! # Bag-Aware Inductive Invariant Checker via SMT (CEGIS)
//!
//! Uses Z3 to find per-bag-state inductive invariants P_B(board) under
//! adversarial piece selection within the 7-bag randomizer.
//!
//! ## Key insight
//!
//! Standard 1-step inductiveness checks the same invariant P for all pieces.
//! But the 7-bag constrains the adversary: they can only give pieces FROM the
//! current bag. By tracking bag state, the invariant can be:
//! - Looser when many pieces remain (player has options)
//! - Tighter when few remain (less adversary freedom)
//! - Specific when 1 piece left (no adversary choice at all)
//!
//! If we find P_B for each bag state B such that all 448 transitions are safe,
//! this proves infinite play under the 7-bag randomizer.
//!
//! ## Usage
//!
//! ```sh
//! RUST_LOG=info cargo run --release -p tetris-playground --bin tetris_invariant_smt
//! ```

mod cegis;
mod check;
mod config;
mod encoding;

use tracing::{error, info};

use config::*;
use encoding::precompute_placements;

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("tetris_invariant_smt (bag-aware CEGIS)");
    info!("board             = {}x{}", BOARD_COLS, BOARD_ROWS);
    info!("bv_width          = {}", BV_WIDTH);
    info!(
        "partial-bag base  = height <= {}, total_holes <= {}, roughness <= {}",
        MAX_HEIGHT, MAX_TOTAL_HOLES, MAX_ROUGHNESS
    );
    info!(
        "full-bag base     = height <= {}, roughness <= {}",
        FULL_BAG_MAX_HEIGHT, FULL_BAG_MAX_ROUGHNESS
    );

    let placements = precompute_placements();
    info!("placements        = {}", placements.len());

    // Verify empty board satisfies full-bag invariant
    let full_bag_base = InvariantParams {
        max_height: FULL_BAG_MAX_HEIGHT,
        max_total_holes: 0,
        max_roughness: FULL_BAG_MAX_ROUGHNESS,
        min_flat_block: 4,
    };
    let empty = tetris_game::TetrisBoard::new();
    let empty_ok = encoding::board_satisfies_invariant_with_params(&empty, &full_bag_base);
    info!("empty board in P  = {}", empty_ok);
    if !empty_ok {
        error!("empty board does not satisfy full-bag base invariant");
        std::process::exit(2);
    }

    // Run bag-aware CEGIS
    info!("--- bag-aware CEGIS ---");
    let max_rounds = 500;
    let success = cegis::run_bag_cegis(&placements, max_rounds);

    if !success {
        std::process::exit(1);
    }
}
