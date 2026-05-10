# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "plotly",
# ]
# ///
"""
Interactive 3D PCA visualization of a Tetris game state trajectory.

Feature vector: [col_heights(10) | col_holes(10)], StandardScaler-normalized.
PCA axes oriented so increasing values = higher feature norm (worse board).

Rendered with Plotly (WebGL) — opens an interactive HTML page in your browser.

Reads the binary trajectory file produced by the tetris_state_trajectory binary.

Usage:
    uv run crates/tetris-playground/src/bin/visualizations/state_trajectory/plot_trajectory_3d.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    u32_block = raw[:, 202:].view(np.uint8).reshape(n, 16)
    heights = np.frombuffer(u32_block[:, 0:4].tobytes(), dtype=np.uint32).astype(np.float32)
    cumulative_lines = np.frombuffer(u32_block[:, 8:12].tobytes(), dtype=np.uint32).astype(
        np.float32
    )

    return boards, heights, cumulative_lines


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


def oriented_pca(features: np.ndarray, n_components: int = 3):
    """Origin-centered PCA: empty board (zeros) maps to origin, worse boards farther out."""
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
    boards, heights, cumulative_lines = load_binary_trajectory(path)
    n = len(boards)
    print(f"Loaded {n:,} steps")

    print("Extracting features [col_heights | col_holes]...")
    features = boards_to_features(boards)
    norms = np.linalg.norm(features, axis=1)
    print(f"Feature norm — min: {norms.min():.1f}, mean: {norms.mean():.1f}, max: {norms.max():.1f}")

    print("Running oriented PCA (3 components)...")
    coords, var_explained = oriented_pca(features, n_components=3)
    total_var = sum(var_explained)
    print(
        f"Variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, "
        f"PC3={var_explained[2]:.3f} (total={total_var:.3f})"
    )
    for k in range(3):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        print(f"  PC{k+1} vs norm correlation: {corr:+.3f}")

    # Subsample for WebGL performance
    max_points = 50_000
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        coords_sub = coords[idx]
        norms_sub = norms[idx]
        heights_sub = heights[idx]
        cumlines_sub = cumulative_lines[idx]
        steps_sub = idx
        print(f"Subsampled to {max_points:,} points for WebGL")
    else:
        coords_sub = coords
        norms_sub = norms
        heights_sub = heights
        cumlines_sub = cumulative_lines
        steps_sub = np.arange(n)

    n_sub = len(coords_sub)
    t_sub = np.linspace(0, 1, n_sub)

    axis_labels = dict(
        xaxis_title=f"PC1 ({var_explained[0]:.1%}) — worse →",
        yaxis_title=f"PC2 ({var_explained[1]:.1%}) — worse →",
        zaxis_title=f"PC3 ({var_explained[2]:.1%}) — worse →",
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Trajectory colored by time ({n:,} steps)",
            "States colored by ‖[heights|holes]‖",
        ],
    )

    # --- Left: trajectory line colored by time ---
    # Plotly doesn't support per-segment coloring on a single line trace,
    # so we use a scatter with mode="lines+markers" and very small markers
    # that carry the color, plus lines connecting them.
    # For clean results, use markers only (tiny) with connecting lines.
    hover_time = [
        f"step {s:,}<br>height {int(h)}<br>lines {int(cl):,}<br>norm {nm:.1f}"
        for s, h, cl, nm in zip(steps_sub, heights_sub, cumlines_sub, norms_sub)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=coords_sub[:, 0],
            y=coords_sub[:, 1],
            z=coords_sub[:, 2],
            mode="lines+markers",
            line=dict(color=t_sub, colorscale="Viridis", width=2),
            marker=dict(
                size=1,
                color=t_sub,
                colorscale="Viridis",
                colorbar=dict(title="Progress", x=0.45, len=0.8),
                opacity=0.6,
            ),
            hovertext=hover_time,
            hoverinfo="text",
            name="Trajectory",
        ),
        row=1, col=1,
    )

    # Start / end markers
    fig.add_trace(
        go.Scatter3d(
            x=[coords_sub[0, 0]], y=[coords_sub[0, 1]], z=[coords_sub[0, 2]],
            mode="markers",
            marker=dict(size=8, color="lime", line=dict(width=1, color="black")),
            name="Start",
            hovertext=[f"Start — step 0"],
            hoverinfo="text",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[coords_sub[-1, 0]], y=[coords_sub[-1, 1]], z=[coords_sub[-1, 2]],
            mode="markers",
            marker=dict(size=8, color="red", line=dict(width=1, color="black")),
            name="End",
            hovertext=[f"End — step {steps_sub[-1]:,}"],
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # --- Right: scatter colored by feature norm ---
    hover_norm = [
        f"step {s:,}<br>height {int(h)}<br>norm {nm:.1f}"
        for s, h, nm in zip(steps_sub, heights_sub, norms_sub)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=coords_sub[:, 0],
            y=coords_sub[:, 1],
            z=coords_sub[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=norms_sub,
                colorscale="Inferno",
                colorbar=dict(title="‖features‖", x=1.0, len=0.8),
                opacity=0.5,
            ),
            hovertext=hover_norm,
            hoverinfo="text",
            name="Norm",
        ),
        row=1, col=2,
    )

    # Layout
    scene_common = dict(
        **axis_labels,
        bgcolor="#0a0a0a",
        xaxis=dict(gridcolor="#222", zerolinecolor="#333"),
        yaxis=dict(gridcolor="#222", zerolinecolor="#333"),
        zaxis=dict(gridcolor="#222", zerolinecolor="#333"),
    )

    fig.update_layout(
        scene=scene_common,
        scene2=scene_common,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(color="gray"),
        title=dict(
            text=f"Tetris State Trajectory — 3D PCA on [col_heights|col_holes] ({n:,} steps)",
            font=dict(color="white", size=16),
        ),
        showlegend=True,
        legend=dict(font=dict(color="white")),
        width=1600,
        height=800,
    )

    out_html = "artifacts/data/state_trajectory_pca_3d.html"
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved interactive 3D plot to {out_html}")

    # Also save a static png
    try:
        out_png = "artifacts/data/state_trajectory_pca_3d.png"
        fig.write_image(out_png, width=1600, height=800, scale=2)
        print(f"Saved static image to {out_png}")
    except Exception as e:
        print(f"Static image export skipped (install kaleido for PNG): {e}")

    fig.show()


if __name__ == "__main__":
    main()
