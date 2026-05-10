# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
PCA visualization of a Tetris game state trajectory.

Reads the binary trajectory file produced by the tetris_state_trajectory binary,
runs PCA on all board states to reduce them to 2D, then plots the trajectory
as a colored path through the latent space.

Binary format: fixed-size 218-byte records, no header.
  board [u8; 200] | piece u8 | placement u8 | height u32le | lines_cleared u32le
  | cumulative_lines u32le | cell_count u32le

Usage:
    uv run crates/tetris-playground/src/bin/visualizations/state_trajectory/plot_trajectory.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

TRAJECTORY_FILE = "artifacts/data/state_trajectory.bin"
RECORD_SIZE = 218
NCOLS = 10
NROWS = 20


def load_binary_trajectory(path: Path):
    """Load the binary trajectory file into numpy arrays."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // RECORD_SIZE
    if n == 0:
        print("No records found in trajectory file")
        sys.exit(1)
    raw = raw[: n * RECORD_SIZE].reshape(n, RECORD_SIZE)

    boards = raw[:, :200].astype(np.float32)
    pieces = raw[:, 200]
    placements = raw[:, 201]

    # Read u32 LE fields
    u32_block = raw[:, 202:].view(np.uint8).reshape(n, 16)
    heights = np.frombuffer(u32_block[:, 0:4].tobytes(), dtype=np.uint32).astype(np.float32)
    lines_cleared = np.frombuffer(u32_block[:, 4:8].tobytes(), dtype=np.uint32).astype(np.float32)
    cumulative_lines = np.frombuffer(u32_block[:, 8:12].tobytes(), dtype=np.uint32).astype(
        np.float32
    )
    cell_counts = np.frombuffer(u32_block[:, 12:16].tobytes(), dtype=np.uint32).astype(np.float32)

    return boards, pieces, placements, heights, lines_cleared, cumulative_lines, cell_counts


def boards_to_features(boards: np.ndarray) -> np.ndarray:
    """Convert raw (n, 200) board arrays to (n, 41) feature vectors. Higher = worse."""
    n = boards.shape[0]
    cols = boards.reshape(n, NCOLS, NROWS)
    row_indices = np.arange(1, NROWS + 1, dtype=np.float32)
    col_heights = (cols * row_indices[np.newaxis, np.newaxis, :]).max(axis=2)
    col_filled = cols.sum(axis=2)
    col_holes = col_heights - col_filled
    adj_height_diffs = np.abs(np.diff(col_heights, axis=1))
    height_mask = row_indices[np.newaxis, np.newaxis, :] <= col_heights[:, :, np.newaxis]
    hole_mask = height_mask & (cols == 0)
    filled_above = np.cumsum(cols[:, :, ::-1], axis=2)[:, :, ::-1]
    filled_above_shifted = np.zeros_like(filled_above)
    filled_above_shifted[:, :, :-1] = filled_above[:, :, 1:]
    blockades = (hole_mask * filled_above_shifted).sum(axis=2)
    padded = np.ones((n, NCOLS + 2, NROWS), dtype=cols.dtype)
    padded[:, 1:-1, :] = cols
    row_trans = np.abs(np.diff(padded, axis=1)).sum(axis=(1, 2)).astype(np.float32).reshape(n, 1)
    floor = np.ones((n, NCOLS, 1), dtype=cols.dtype)
    ceil = np.zeros((n, NCOLS, 1), dtype=cols.dtype)
    padded_v = np.concatenate([floor, cols, ceil], axis=2)
    col_trans = np.abs(np.diff(padded_v, axis=2)).sum(axis=(1, 2)).astype(np.float32).reshape(n, 1)
    return np.concatenate([
        col_heights, col_holes, adj_height_diffs, blockades, row_trans, col_trans,
    ], axis=1)


def oriented_pca(features: np.ndarray, n_components: int = 2):
    """Origin-centered PCA: empty board (zeros) maps to origin, worse boards farther out.

    Scales by std (no mean subtraction) then SVD. Flips axes so positive = worse.
    """
    norms = np.linalg.norm(features, axis=1)
    stds = features.std(axis=0)
    stds[stds == 0] = 1
    features_scaled = features / stds

    U, S, Vt = np.linalg.svd(features_scaled, full_matrices=False)
    components = Vt[:n_components]
    total_var = (S ** 2).sum()
    var_explained = (S[:n_components] ** 2) / total_var
    coords = features_scaled @ components.T

    for k in range(n_components):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        if corr < 0:
            coords[:, k] *= -1
    return coords, var_explained


def main():
    path = Path(TRAJECTORY_FILE)
    if not path.exists():
        print(f"Trajectory file not found: {path}")
        print("Run the Rust binary first:")
        print("  cargo run --release -p tetris-playground --bin tetris_state_trajectory")
        sys.exit(1)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loading trajectory from {path} ({file_size_mb:.1f} MB)...")
    boards, pieces, placements, heights, lines_cleared, cumulative_lines, cell_counts = (
        load_binary_trajectory(path)
    )
    n = len(boards)
    print(f"Loaded {n:,} steps")

    # Build feature vectors: [col_heights(10) | col_holes(10)]
    print("Extracting features [col_heights | col_holes]...")
    features = boards_to_features(boards)
    norms = np.linalg.norm(features, axis=1)
    print(f"Feature norm — min: {norms.min():.1f}, mean: {norms.mean():.1f}, max: {norms.max():.1f}")

    # Oriented PCA: positive direction = higher norm = worse state
    print("Running oriented PCA...")
    coords, var_explained = oriented_pca(features)
    print(f"Variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")
    for k in range(2):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        print(f"  PC{k+1} vs norm correlation: {corr:+.3f}")

    # Build line segments for colored trajectory
    points = coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.linspace(0, 1, n - 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- Top-left: trajectory colored by time ---
    ax = axes[0, 0]
    lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 1))
    lc.set_array(t)
    lc.set_linewidth(0.3)
    ax.add_collection(lc)
    ax.scatter(*coords[0], c="lime", s=120, zorder=5, edgecolors="black", label="Start")
    ax.scatter(*coords[-1], c="red", s=120, zorder=5, edgecolors="black", label="End")
    ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%}) — worse →")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%}) — worse →")
    ax.set_title(f"Trajectory colored by time ({n:,} steps)")
    ax.legend(loc="upper right")
    fig.colorbar(lc, ax=ax, label="Game progress (0=start, 1=end)")

    # --- Top-right: density heatmap colored by height ---
    ax2 = axes[0, 1]
    hb = ax2.hexbin(
        coords[:, 0],
        coords[:, 1],
        C=heights,
        reduce_C_function=np.mean,
        gridsize=60,
        cmap="hot",
        mincnt=1,
    )
    ax2.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax2.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax2.set_xlabel(f"PC1 ({var_explained[0]:.1%}) — worse →")
    ax2.set_ylabel(f"PC2 ({var_explained[1]:.1%}) — worse →")
    ax2.set_title("Mean board height (hexbin)")
    fig.colorbar(hb, ax=ax2, label="Mean height")

    # --- Bottom-left: time series of height + cumulative lines ---
    ax3 = axes[1, 0]
    steps = np.arange(n)
    ax3.plot(steps, heights, linewidth=0.4, alpha=0.7, label="Height", color="tab:blue")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        steps, cumulative_lines, linewidth=0.8, alpha=0.9, label="Cumulative lines", color="tab:orange"
    )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Board height", color="tab:blue")
    ax3_twin.set_ylabel("Cumulative lines cleared", color="tab:orange")
    ax3.set_title("Height and lines over time")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # --- Bottom-right: density heatmap colored by feature norm ---
    ax4 = axes[1, 1]
    hb2 = ax4.hexbin(
        coords[:, 0],
        coords[:, 1],
        C=norms,
        reduce_C_function=np.mean,
        gridsize=60,
        cmap="inferno",
        mincnt=1,
    )
    ax4.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax4.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax4.set_xlabel(f"PC1 ({var_explained[0]:.1%}) — worse →")
    ax4.set_ylabel(f"PC2 ({var_explained[1]:.1%}) — worse →")
    ax4.set_title("Mean feature norm [heights|holes] (hexbin)")
    fig.colorbar(hb2, ax=ax4, label="Mean ‖[heights|holes]‖")

    plt.tight_layout()

    out_path = "artifacts/data/state_trajectory_pca.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
