# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "Pillow",
# ]
# ///
"""
Animated webp of a Tetris state trajectory in a 2D latent space.

Embedding modes:
  --embed pca        StandardScaler + PCA (default)
  --embed temporal   Rolling-average smoothed features + PCA — consecutive
                     states share most of their window so PCA places them
                     close together, producing smooth trajectories.

Feature vector: [col_heights(10) | col_holes(10)], 20-dim.
Axes oriented so increasing values = higher feature norm (worse board).

Usage:
    uv run .../animate_trajectory.py                     # PCA, fast
    uv run .../animate_trajectory.py --embed temporal    # smoothed PCA
    uv run .../animate_trajectory.py --embed temporal --speed 3
"""

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter1d

TRAJECTORY_FILE = "artifacts/data/state_trajectory.bin"
RECORD_SIZE = 218

# Animation parameters
FPS = 30
DURATION_SECS = 10  # short for quick testing — increase for full render
DPI = 100
FIG_SIZE = (10, 10)  # inches


def parse_args():
    parser = argparse.ArgumentParser(description="Animate Tetris state trajectory")
    parser.add_argument(
        "--embed", choices=["pca", "temporal"], default="pca",
        help="Embedding method: 'pca' (default) or 'temporal' (spectral on temporal graph).",
    )
    parser.add_argument(
        "--features", choices=["full", "colvals"], default="full",
        help="Feature vector: 'full' (41-dim heights/holes/bumps/blockades/transitions) "
             "or 'colvals' (10-dim normalized raw column u32 values).",
    )
    parser.add_argument(
        "--speed", type=float, default=0,
        help="Trajectory steps per second. 0 (default) = spread all steps across the video. "
             "Use 2-3 for a slow close-up of individual transitions.",
    )
    parser.add_argument("--duration", type=int, default=DURATION_SECS, help="Video duration in seconds")
    parser.add_argument("--tail", type=int, default=0, help="Tail length in trajectory steps (0 = auto)")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file path")
    return parser.parse_args()


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
    """Convert raw (n, 200) board arrays to (n, 41) feature vectors.

    All features are non-negative; higher value = worse board state.

    Layout (41 dims):
      col_heights      (10)  — height of each column
      col_holes         (10)  — holes per column (height - filled)
      adj_height_diffs  (9)   — |h[i] - h[i+1]| for adjacent columns (bumpiness)
      blockades         (10)  — filled cells above holes per column
      row_transitions   (1)   — total horizontal filled↔empty transitions
      col_transitions   (1)   — total vertical filled↔empty transitions

    Board layout: column-major, 10 cols x 20 rows.
    """
    n = boards.shape[0]
    cols = boards.reshape(n, NCOLS, NROWS)

    # Column heights
    row_indices = np.arange(1, NROWS + 1, dtype=np.float32)
    col_heights = (cols * row_indices[np.newaxis, np.newaxis, :]).max(axis=2)  # (n, 10)

    # Column filled counts & holes
    col_filled = cols.sum(axis=2)
    col_holes = col_heights - col_filled  # (n, 10)

    # Adjacent height differences (bumpiness)
    adj_height_diffs = np.abs(np.diff(col_heights, axis=1))  # (n, 9)

    # Blockades: for each hole, count filled cells above it in the same column
    # A cell is a hole if it's empty and below the column height
    height_mask = row_indices[np.newaxis, np.newaxis, :] <= col_heights[:, :, np.newaxis]  # (n, 10, 20)
    hole_mask = height_mask & (cols == 0)  # empty cells below the height
    # For each hole at row r, blockade = number of filled cells in rows r+1..height
    # Cumulative filled from top: reverse cumsum of cols along row axis
    filled_above = np.cumsum(cols[:, :, ::-1], axis=2)[:, :, ::-1]  # cumsum from top
    # Shift down by 1: filled_above[r] should be sum of cols[r+1..19]
    filled_above_shifted = np.zeros_like(filled_above)
    filled_above_shifted[:, :, :-1] = filled_above[:, :, 1:]
    blockades = (hole_mask * filled_above_shifted).sum(axis=2)  # (n, 10)

    # Row transitions: horizontal filled↔empty changes per row
    # Treat walls (col -1 and col 10) as filled
    padded = np.ones((n, NCOLS + 2, NROWS), dtype=cols.dtype)
    padded[:, 1:-1, :] = cols
    row_trans = np.abs(np.diff(padded, axis=1)).sum(axis=(1, 2)).astype(np.float32).reshape(n, 1)

    # Column transitions: vertical filled↔empty changes per column
    # Treat floor (row -1) as filled, top (row 20) as empty
    floor = np.ones((n, NCOLS, 1), dtype=cols.dtype)
    ceil = np.zeros((n, NCOLS, 1), dtype=cols.dtype)
    padded_v = np.concatenate([floor, cols, ceil], axis=2)
    col_trans = np.abs(np.diff(padded_v, axis=2)).sum(axis=(1, 2)).astype(np.float32).reshape(n, 1)

    return np.concatenate([
        col_heights, col_holes, adj_height_diffs, blockades, row_trans, col_trans,
    ], axis=1)  # (n, 41)


def boards_to_colvals(boards: np.ndarray) -> np.ndarray:
    """Convert raw (n, 200) board arrays to (n, 10) normalized column u32 values.

    Each column is a 20-bit bitmask. Reconstruct the u32 value from bits,
    then normalize by dividing each column by its max observed value so
    each dimension is in [0, 1].
    """
    n = boards.shape[0]
    cols = boards.reshape(n, NCOLS, NROWS)
    powers = (2.0 ** np.arange(NROWS, dtype=np.float64))[np.newaxis, np.newaxis, :]
    col_vals = (cols * powers).sum(axis=2).astype(np.float32)  # (n, 10)
    # Normalize each column by its max observed value
    col_max = col_vals.max(axis=0)
    col_max[col_max == 0] = 1
    return col_vals / col_max


def orient_axes(coords, norms, n_components):
    """Flip each axis so positive direction correlates with higher feature norm."""
    for k in range(n_components):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        if corr < 0:
            coords[:, k] *= -1
    return coords


def _scale_no_center(features):
    """Scale features by std without centering — empty board (zeros) maps to origin."""
    stds = features.std(axis=0)
    stds[stds == 0] = 1
    return features / stds


def _origin_pca(features_scaled, n_components):
    """PCA via SVD without mean-centering, so the zero vector stays at origin."""
    U, S, Vt = np.linalg.svd(features_scaled, full_matrices=False)
    components = Vt[:n_components]
    total_var = (S ** 2).sum()
    ve = (S[:n_components] ** 2) / total_var
    coords = features_scaled @ components.T
    return coords, ve


def embed_pca(features: np.ndarray, n_components: int = 2):
    """Origin-centered PCA: empty board (zeros) maps to origin, worse boards farther out.

    Scales by std (no mean subtraction) so the zero-feature vector stays at origin,
    then SVD for principal components.

    Returns (coords, axis_labels, info_lines).
    """
    norms = np.linalg.norm(features, axis=1)
    features_scaled = _scale_no_center(features)
    coords, ve = _origin_pca(features_scaled, n_components)
    coords = orient_axes(coords, norms, n_components)

    step_dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mean_step = step_dists.mean()

    labels = [f"PC{k+1} ({ve[k]:.1%}) — worse →" for k in range(n_components)]
    info = [
        f"Origin-centered PCA (empty board = origin)",
        f"Variance explained: {', '.join(f'PC{k+1}={ve[k]:.3f}' for k in range(n_components))}",
        f"Mean step distance: {mean_step:.4f}",
    ]
    for k in range(n_components):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        info.append(f"  PC{k+1} vs norm correlation: {corr:+.3f}")
    return coords, labels, info


def embed_temporal(features: np.ndarray, n_components: int = 2, smooth_window: int = 50):
    """Temporally-smoothed origin-centered PCA.

    Scale by std (no mean), smooth along time, then SVD. Empty board feature
    vector (zeros) maps to origin; temporal smoothing keeps consecutive states
    close together.

    Returns (coords, axis_labels, info_lines).
    """
    norms = np.linalg.norm(features, axis=1)
    features_scaled = _scale_no_center(features)

    # Smooth each feature dimension independently along the time axis
    features_smooth = uniform_filter1d(features_scaled, size=smooth_window, axis=0)

    coords, ve = _origin_pca(features_smooth, n_components)
    coords = orient_axes(coords, norms, n_components)

    # Measure how smooth the trajectory is: mean step distance
    step_dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mean_step = step_dists.mean()

    labels = [f"PC{k+1} ({ve[k]:.1%}, smooth={smooth_window}) — worse →" for k in range(n_components)]
    info = [
        f"Origin-centered temporal PCA (empty board = origin, window={smooth_window})",
        f"Variance explained: {', '.join(f'PC{k+1}={ve[k]:.3f}' for k in range(n_components))}",
        f"Mean step distance: {mean_step:.4f}",
    ]
    for k in range(n_components):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        info.append(f"  PC{k+1} vs norm correlation: {corr:+.3f}")
    return coords, labels, info


def render_background(coords, axis_labels, segments_all, xlim, ylim, embed_name, heights):
    """Render the static background (axes, height heatmap, faded trail, start marker) to a PIL Image."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(axis_labels[0], color="gray", fontsize=11)
    ax.set_ylabel(axis_labels[1], color="gray", fontsize=11)
    ax.tick_params(colors="gray", labelsize=9)
    ax.set_title("Tetris State Trajectory", color="white", fontsize=13, pad=12)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Faint height heatmap — hexbin of mean height per region
    hb = ax.hexbin(
        coords[:, 0], coords[:, 1],
        C=heights,
        reduce_C_function=np.mean,
        gridsize=40,
        cmap="YlOrRd",
        mincnt=1,
        alpha=0.15,
        zorder=1,
        rasterized=True,
    )
    fig.colorbar(hb, ax=ax, shrink=0.6, label="Mean height", pad=0.02)

    full_trail = LineCollection(
        segments_all, colors=(1, 1, 1, 0.04), linewidths=0.3, rasterized=True, zorder=2,
    )
    ax.add_collection(full_trail)

    # Origin marker — the empty board (feature = zeros) maps here
    ax.scatter(0, 0, c="white", s=120, zorder=9, marker="+", linewidths=2, label="Empty board")

    ax.scatter(
        coords[0, 0], coords[0, 1], c="lime", s=80, zorder=8, edgecolors="white", linewidths=0.8,
        label="Start",
    )

    fig.tight_layout()

    # Capture axes bbox for coordinate mapping before closing
    bbox = ax.get_position()
    fig_w_px = FIG_SIZE[0] * DPI
    fig_h_px = FIG_SIZE[1] * DPI
    ax_bbox_px = (
        bbox.x0 * fig_w_px,   # left
        bbox.x1 * fig_w_px,   # right
        bbox.y0 * fig_h_px,   # bottom
        bbox.y1 * fig_h_px,   # top
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    bg_img = Image.open(buf).convert("RGB")

    plt.close(fig)
    return bg_img, ax_bbox_px


def data_to_pixel(coords_2d, xlim, ylim, ax_bbox_px, fig_h_px):
    """Convert PCA data coordinates to pixel coordinates in the rendered image."""
    ax_left, ax_right, ax_bottom, ax_top = ax_bbox_px

    x_frac = (coords_2d[:, 0] - xlim[0]) / (xlim[1] - xlim[0])
    y_frac = (coords_2d[:, 1] - ylim[0]) / (ylim[1] - ylim[0])

    px_x = ax_left + x_frac * (ax_right - ax_left)
    # matplotlib y=0 at bottom, PIL y=0 at top
    px_y = fig_h_px - (ax_bottom + y_frac * (ax_top - ax_bottom))

    return np.column_stack([px_x, px_y])


def main():
    args = parse_args()
    duration_secs = args.duration

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

    embed_name = args.embed
    feat_name = args.features
    if feat_name == "colvals":
        print("Extracting features [normalized column u32 values] (10-dim)...")
        features = boards_to_colvals(boards)
    else:
        print("Extracting features [heights|holes|bumps|blockades|transitions] (41-dim)...")
        features = boards_to_features(boards)
    norms = np.linalg.norm(features, axis=1)
    print(f"Feature norm — min: {norms.min():.1f}, mean: {norms.mean():.1f}, max: {norms.max():.1f}")

    # Embed
    print(f"Embedding: {embed_name}")
    if embed_name == "temporal":
        coords, axis_labels, info = embed_temporal(features, n_components=2)
    else:
        coords, axis_labels, info = embed_pca(features, n_components=2)
    for line in info:
        print(line)

    # Precompute segments
    points_all = coords.reshape(-1, 1, 2)
    segments_all = np.concatenate([points_all[:-1], points_all[1:]], axis=1)

    # Axis limits
    pad = 2
    xlim = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ylim = (coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    # Render static background
    print("Rendering background...")
    bg_img, ax_bbox_px = render_background(coords, axis_labels, segments_all, xlim, ylim, embed_name, heights)
    img_w, img_h = bg_img.size
    fig_h_px = FIG_SIZE[1] * DPI

    # Convert all PCA coords to pixel coords
    print("Computing pixel coordinates...")
    pixel_coords = data_to_pixel(coords, xlim, ylim, ax_bbox_px, fig_h_px)

    total_frames = FPS * duration_secs

    # Determine speed mode
    speed_tag = "slow" if args.speed > 0 else "fast"
    default_output = f"artifacts/data/state_trajectory_{feat_name}_{embed_name}_{speed_tag}.mp4"

    if args.speed > 0:
        # Slow mode: fixed steps per second with interpolation between steps
        steps_per_sec = args.speed
        total_steps_shown = int(steps_per_sec * duration_secs)
        total_steps_shown = min(total_steps_shown, n - 1)
        tail_length = args.tail if args.tail > 0 else max(10, int(steps_per_sec * 5))
        output_file = args.output or default_output
        print(f"Slow mode: {steps_per_sec} steps/s, {total_steps_shown} steps over {duration_secs}s")

        frame_frac = np.linspace(0, total_steps_shown, total_frames, endpoint=False)
    else:
        # Fast mode: spread all n steps across the video
        tail_length = args.tail if args.tail > 0 else 400
        output_file = args.output or default_output
        frame_frac = np.linspace(0, n - 1, total_frames).astype(float)
        print(f"Fast mode: ~{n // total_frames} trajectory steps per frame")

    # Plasma colormap for tail
    cmap = plt.get_cmap("plasma")

    # Try to load a monospace font for the HUD
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size=14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    print(f"Rendering {total_frames} frames at {FPS} fps ({duration_secs}s)...")
    print(f"Tail length: {tail_length} steps")
    print(f"Frame size: {img_w}x{img_h}")
    print(f"Output: {output_file}")

    # Start ffmpeg — stream raw RGB frames in, encode mp4 out
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{img_w}x{img_h}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_file,
    ]
    print(f"ffmpeg: {' '.join(ffmpeg_cmd)}")
    ffproc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)

    t_start = time.monotonic()
    log_every = 300

    for f_idx in range(total_frames):
        frac = frame_frac[f_idx]
        i = int(frac)
        t_lerp = frac - i

        i = min(i, n - 2)

        dot_x = pixel_coords[i, 0] + t_lerp * (pixel_coords[i + 1, 0] - pixel_coords[i, 0])
        dot_y = pixel_coords[i, 1] + t_lerp * (pixel_coords[i + 1, 1] - pixel_coords[i, 1])

        frame = bg_img.copy()
        draw = ImageDraw.Draw(frame)

        # Draw bright tail using layered polylines
        tail_start = max(0, i - tail_length)
        tail_len = i - tail_start
        if tail_len > 1:
            tail_pts = list(pixel_coords[tail_start : i + 1])
            tail_pts.append(np.array([dot_x, dot_y]))
            pts_list = [tuple(p) for p in tail_pts]
            for layer_alpha, layer_width, t_offset in [
                (0.15, 3, 0.0),
                (0.5, 2, 0.3),
                (1.0, 1, 0.7),
            ]:
                rgba = cmap(t_offset + (1.0 - t_offset) * 0.8)
                color = (
                    int(rgba[0] * 255),
                    int(rgba[1] * 255),
                    int(rgba[2] * 255),
                )
                draw.line(pts_list, fill=color, width=layer_width)

        # Leading dot (glow + core)
        r_glow = 10
        draw.ellipse(
            [dot_x - r_glow, dot_y - r_glow, dot_x + r_glow, dot_y + r_glow],
            fill=(0, 255, 255),
        )
        r_dot = 5
        draw.ellipse(
            [dot_x - r_dot, dot_y - r_dot, dot_x + r_dot, dot_y + r_dot],
            fill=(255, 255, 255),
        )

        # HUD text
        h = heights[i] + t_lerp * (heights[min(i + 1, n - 1)] - heights[i])
        cl = cumulative_lines[i]
        norm_val = norms[i] + t_lerp * (norms[min(i + 1, n - 1)] - norms[i])
        hud = f"step {i:>6,} / {n:,}   height {int(h):>2}   norm {norm_val:>5.1f}   lines {int(cl):>6,}"
        draw.text((15, img_h - 25), hud, fill=(160, 160, 160), font=font)

        # Stream raw RGB to ffmpeg
        ffproc.stdin.write(frame.tobytes())

        if (f_idx + 1) % log_every == 0:
            elapsed = time.monotonic() - t_start
            fps_render = (f_idx + 1) / elapsed
            eta = (total_frames - f_idx - 1) / fps_render
            print(f"  frame {f_idx + 1}/{total_frames}  ({fps_render:.0f} frames/s, ETA {eta:.0f}s)")

    ffproc.stdin.close()
    ret = ffproc.wait()
    elapsed = time.monotonic() - t_start

    if ret != 0:
        print(f"ffmpeg exited with code {ret}", file=sys.stderr)
        sys.exit(ret)

    out_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"Done in {elapsed:.1f}s — {out_size_mb:.1f} MB mp4 saved to {output_file}")


if __name__ == "__main__":
    main()
