# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "Pillow",
# ]
# ///
"""
Animated 3D webp of a Tetris state trajectory through PCA space.

The camera slowly orbits the point cloud while a tracing line with a leading dot
advances through the trajectory. Uses matplotlib's 3D projection rendered to
PIL images, then assembled into webp.

Feature vector: [col_heights(10) | col_holes(10)], StandardScaler-normalized.
PCA axes oriented so increasing values = higher feature norm (worse board).

Usage:
    uv run crates/tetris-playground/src/bin/visualizations/state_trajectory/animate_trajectory_3d.py
    uv run ... --speed 3 --duration 20   # slow mode
"""

import argparse
import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image, ImageDraw, ImageFont

TRAJECTORY_FILE = "artifacts/data/state_trajectory.bin"
RECORD_SIZE = 218
NCOLS = 10
NROWS = 20

FPS = 30
DURATION_SECS = 10
DPI = 100
FIG_SIZE = (10, 10)

# Camera: full orbits over the video duration
ELEV_BASE = 25
ELEV_AMP = 10       # gentle nod up/down
AZIM_START = -60
AZIM_ORBITS = 0.5   # half orbit over the video


def parse_args():
    parser = argparse.ArgumentParser(description="Animate 3D Tetris state trajectory")
    parser.add_argument("--speed", type=float, default=0, help="Steps/sec (0=spread all)")
    parser.add_argument("--duration", type=int, default=DURATION_SECS, help="Video seconds")
    parser.add_argument("--tail", type=int, default=0, help="Tail length (0=auto)")
    parser.add_argument("-o", "--output", type=str, default="", help="Output path")
    return parser.parse_args()


def load_binary_trajectory(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // RECORD_SIZE
    if n == 0:
        print("No records found")
        sys.exit(1)
    raw = raw[: n * RECORD_SIZE].reshape(n, RECORD_SIZE)
    boards = raw[:, :200].astype(np.float32)
    u32_block = raw[:, 202:].view(np.uint8).reshape(n, 16)
    heights = np.frombuffer(u32_block[:, 0:4].tobytes(), dtype=np.uint32).astype(np.float32)
    cumulative_lines = np.frombuffer(u32_block[:, 8:12].tobytes(), dtype=np.uint32).astype(np.float32)
    return boards, heights, cumulative_lines


def boards_to_features(boards):
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


def oriented_pca(features, n_components=3):
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


def render_frame(ax, fig, coords_sub, norms_sub, var_explained,
                 trail_coords, dot_coord, elev, azim, hud_text, n_total):
    """Render one frame by updating the 3D axes and returning a PIL Image."""
    ax.cla()

    # Faint background scatter
    ax.scatter(
        coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
        c=norms_sub, cmap="inferno", s=0.3, alpha=0.08, rasterized=True,
    )

    # Bright trail
    if len(trail_coords) > 1:
        pts = trail_coords.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t_arr = np.linspace(0, 1, len(segs))
        lc = Line3DCollection(segs, cmap="plasma", linewidths=1.5, alpha=0.9)
        lc.set_array(t_arr)
        ax.add_collection3d(lc)

    # Leading dot
    if dot_coord is not None:
        ax.scatter(*dot_coord, c="white", s=60, zorder=10, edgecolors="cyan", linewidths=1.5)

    # Start marker
    ax.scatter(*coords_sub[0], c="lime", s=50, zorder=8, edgecolors="white", linewidths=0.5)

    ax.set_xlim(coords_sub[:, 0].min() - 1, coords_sub[:, 0].max() + 1)
    ax.set_ylim(coords_sub[:, 1].min() - 1, coords_sub[:, 1].max() + 1)
    ax.set_zlim(coords_sub[:, 2].min() - 1, coords_sub[:, 2].max() + 1)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8, color="gray")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8, color="gray")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.0%})", fontsize=8, color="gray")
    ax.tick_params(labelsize=6, colors="gray")
    ax.set_facecolor("#0a0a0a")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222222")
    ax.yaxis.pane.set_edgecolor("#222222")
    ax.zaxis.pane.set_edgecolor("#222222")
    ax.grid(True, alpha=0.15)

    ax.view_init(elev=elev, azim=azim)

    # Render to image buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    w, h = int(fig.get_figwidth() * DPI), int(fig.get_figheight() * DPI)
    img = Image.frombytes("RGBA", (w, h), buf.read()).convert("RGB")

    # Draw HUD text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size=16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((15, h - 25), hud_text, fill=(160, 160, 160), font=font)

    return img


def main():
    args = parse_args()
    duration_secs = args.duration

    path = Path(TRAJECTORY_FILE)
    if not path.exists():
        print(f"Trajectory file not found: {path}")
        sys.exit(1)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Loading trajectory from {path} ({file_size_mb:.1f} MB)...")
    boards, heights, cumulative_lines = load_binary_trajectory(path)
    n = len(boards)
    print(f"Loaded {n:,} steps")

    print("Extracting features [col_heights | col_holes]...")
    features = boards_to_features(boards)
    norms = np.linalg.norm(features, axis=1)

    print("Running oriented PCA (3 components)...")
    coords, var_explained = oriented_pca(features, n_components=3)
    print(f"Variance explained: {var_explained[0]:.3f}, {var_explained[1]:.3f}, {var_explained[2]:.3f}")
    for k in range(3):
        corr = np.corrcoef(coords[:, k], norms)[0, 1]
        print(f"  PC{k+1} vs norm: {corr:+.3f}")

    # Subsample background scatter
    max_bg = 10_000
    if n > max_bg:
        bg_idx = np.linspace(0, n - 1, max_bg, dtype=int)
        coords_bg = coords[bg_idx]
        norms_bg = norms[bg_idx]
    else:
        coords_bg = coords
        norms_bg = norms

    total_frames = FPS * duration_secs

    if args.speed > 0:
        steps_per_sec = args.speed
        total_steps = min(int(steps_per_sec * duration_secs), n - 1)
        tail_length = args.tail if args.tail > 0 else max(10, int(steps_per_sec * 5))
        output_file = args.output or "artifacts/data/state_trajectory_3d_slow.webp"
        frame_frac = np.linspace(0, total_steps, total_frames, endpoint=False)
        print(f"Slow mode: {steps_per_sec} steps/s, {total_steps} steps")
    else:
        tail_length = args.tail if args.tail > 0 else 400
        output_file = args.output or "artifacts/data/state_trajectory_3d.webp"
        frame_frac = np.linspace(0, n - 1, total_frames).astype(float)

    print(f"Rendering {total_frames} frames at {FPS} fps ({duration_secs}s), tail={tail_length}")
    print(f"Output: {output_file}")

    # Set up figure once
    fig = plt.figure(figsize=FIG_SIZE, facecolor="#0a0a0a")
    ax = fig.add_subplot(111, projection="3d")

    frames = []
    t_start = time.monotonic()
    log_every = 30

    for f_idx in range(total_frames):
        frac = frame_frac[f_idx]
        i = min(int(frac), n - 2)
        t_lerp = frac - i

        # Interpolated dot
        dot = coords[i] + t_lerp * (coords[min(i + 1, n - 1)] - coords[i])

        # Trail
        tail_start = max(0, i - tail_length)
        if i > tail_start:
            trail = np.vstack([coords[tail_start:i + 1], dot.reshape(1, 3)])
        else:
            trail = dot.reshape(1, 3)

        # Camera orbit
        progress = f_idx / max(total_frames - 1, 1)
        azim = AZIM_START + 360 * AZIM_ORBITS * progress
        elev = ELEV_BASE + ELEV_AMP * np.sin(2 * np.pi * progress)

        # HUD
        h = heights[i] + t_lerp * (heights[min(i + 1, n - 1)] - heights[i])
        nm = norms[i] + t_lerp * (norms[min(i + 1, n - 1)] - norms[i])
        cl = cumulative_lines[i]
        hud = f"step {i:>6,} / {n:,}   height {int(h):>2}   norm {nm:>5.1f}   lines {int(cl):>6,}"

        img = render_frame(ax, fig, coords_bg, norms_bg, var_explained,
                           trail, dot, elev, azim, hud, n)
        frames.append(img)

        if (f_idx + 1) % log_every == 0:
            elapsed = time.monotonic() - t_start
            fps_r = (f_idx + 1) / elapsed
            eta = (total_frames - f_idx - 1) / fps_r
            print(f"  frame {f_idx + 1}/{total_frames}  ({fps_r:.1f} frames/s, ETA {eta:.0f}s)")

    plt.close(fig)
    elapsed = time.monotonic() - t_start
    print(f"Rendered {total_frames} frames in {elapsed:.1f}s ({total_frames / elapsed:.1f} frames/s)")

    print("Encoding webp...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    t_enc = time.monotonic()
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        quality=80,
    )
    enc_elapsed = time.monotonic() - t_enc
    out_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"Encoded in {enc_elapsed:.1f}s — {out_mb:.1f} MB webp saved to {output_file}")


if __name__ == "__main__":
    main()
