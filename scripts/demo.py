#!/usr/bin/env python3
"""
Example: frost-flower simulation on a 10 cm × 10 cm × 0.5 mm window pane.

Renders a two-panel video:
    left  – top-down slice through the frost layer (the pattern)
    right – cross-section at y = ny/2     (growth profile in z)

Recommended presets
-------------------
    --preset fast    (256² × 16,  ~100 steps)  ~30 s  – smoke test
    --preset demo    (512² × 32,  ~300 steps)  ~3  min – small ferns, quasi-2D
    --preset pretty  (1024² × 32, ~600 steps)  ~15 min – real 3D frost, pretty

Physics notes
-------------
The diffuse-interface width W0 = 3·min(dx, dy) sets the feature scale.
At the "demo" preset W0 = 585 μm > H_z, so growth is quasi-2D (ferns).  At the
"pretty" preset W0 = 293 μm < H_z, so real 3D branching forms.

Tip velocity v ≈ m·D/W0 with m ≈ 0.4 ⇒ v ≈ 15 mm/s at "demo", ~30 mm/s at
"pretty".  Features of size W0 are visible after T_sim ~ W0/v, i.e. ~40 ms and
~10 ms respectively.

The CFL safe time step is dictated by the diffusion term:
    dt_u ≈ 0.4 / (D · (2/dx² + 2/dy² + 2/dz²))
At "demo" (nz=32, D=2e-5) this is ≈ 2×10⁻⁶ s, so the backend sub-steps
~50× internally per external step at dt=1e-4.  The public API is unaffected.

Usage
-----
    python simulate_and_render.py --preset demo --output frost.mp4
    python simulate_and_render.py --preset pretty --steps 800
    python simulate_and_render.py --preset fast --no-render
"""

from __future__ import annotations
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from frostflower import FrostSimulator


# ---------------------------------------------------------------------------
# Domain (physical, in metres)
# ---------------------------------------------------------------------------
DOMAIN_X = 0.10  # 10 cm
DOMAIN_Y = 0.10  # 10 cm
DOMAIN_Z = 5.0e-4  # 0.5 mm


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    # name     nx    ny    nz   steps  stride
    "fast":   (256,  256,  16,   200,    5),
    "demo":   (512,  512,  32,   500,    5),
    "pretty": (1024, 1024, 32,   800,    5),
}


# ---------------------------------------------------------------------------
# Initial-condition generators
# ---------------------------------------------------------------------------
def make_seed_map(ny: int, nx: int, rng, n_particles: int = 40,
                  physical_radius_frac: float = 0.02) -> np.ndarray:
    """
    Seeds with a *physical* radius of ``physical_radius_frac * min(nx, ny)``
    so they always exceed the critical radius R_crit = W0 / (2(m-0.5)).
    At α=3, u_inf=0.5, R_crit ≈ 0.78·W0 ≈ 2.3·dx, so radius ≥ 3 cells
    is safe.  Default 2% of the pane gives very robust growth.
    """
    seed = np.zeros((ny, nx), dtype=np.float32)
    r_cells = max(3, int(physical_radius_frac * min(nx, ny)))
    for _ in range(n_particles):
        cy = int(rng.integers(r_cells + 2, ny - r_cells - 2))
        cx = int(rng.integers(r_cells + 2, nx - r_cells - 2))
        r  = int(rng.integers(r_cells, r_cells + 3))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        blob = ((yy * yy + xx * xx) <= r * r).astype(np.float32)
        seed[cy - r:cy + r + 1, cx - r:cx + r + 1] = np.maximum(
            seed[cy - r:cy + r + 1, cx - r:cx + r + 1], blob)
    return seed


def make_orientation_map(ny: int, nx: int, rng, n_grains: int = 12) -> np.ndarray:
    """Voronoi grain map with random six-fold rotations in [0, π/3)."""
    cy = rng.integers(0, ny, n_grains)
    cx = rng.integers(0, nx, n_grains)
    angles = rng.uniform(0.0, np.pi / 3.0, n_grains).astype(np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    d2 = np.stack([(yy - y) ** 2 + (xx - x) ** 2 for y, x in zip(cy, cx)])
    return angles[np.argmin(d2, axis=0)]


def make_glass_temp(ny: int, nx: int, rng) -> np.ndarray:
    """Radial gradient (cold frame, warmer centre) + small noise."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2.0, nx / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(ny, nx) / 2.0)
    base = 0.30 + 0.70 * r
    noise = 0.02 * rng.standard_normal((ny, nx)).astype(np.float32)
    return np.clip(base + noise, 0.05, 1.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_and_render(args) -> None:
    rng = np.random.default_rng(args.seed)

    nx_p, ny_p, nz_p, steps_p, stride_p = PRESETS[args.preset]
    nx = args.nx or nx_p
    ny = args.ny or ny_p
    nz = args.nz or nz_p
    steps = args.steps or steps_p
    stride = stride_p

    dx = DOMAIN_X / nx
    dy = DOMAIN_Y / ny
    dz = DOMAIN_Z / nz
    W0 = 3.0 * min(dx, dy)

    print(f"[setup] grid = {nx}×{ny}×{nz}  ({(nx * ny * nz) / 1e6:.1f}M cells)")
    print(f"[setup] dx = {dx * 1e6:7.2f} μm   dz = {dz * 1e6:6.2f} μm")
    print(
        f"[setup] W0 = {W0 * 1e6:7.2f} μm   H_z = {DOMAIN_Z * 1e3:.3f} mm"
        f"  -> {'quasi-2D' if W0 > DOMAIN_Z else 'true 3D'}"
    )

    sim = FrostSimulator(nx, ny, nz, dx, dy, dz)

    seed_map = make_seed_map(ny, nx, rng)
    ori_map = make_orientation_map(ny, nx, rng)
    temp_map = make_glass_temp(ny, nx, rng)

    print(f"[setup] seeds = {int(seed_map.sum())} cells")
    sim.set_initial_conditions(seed_map, temp_map, ori_map)

    sim.set_parameters(
        u_inf       = args.u_inf,
        eps6        = args.eps6,
        eps_z       = args.eps_z,
        diffusion_D = 2.0e-5,
        latent_K    = 0.1,
        alpha       = 3.0,          # NEW — was 1.0, this alone fixes the melt
        gamma       = 5.0,          # NEW — was 5.0, keeping arctan unsaturated
        dt          = args.dt,
        tau         = 0.03,         # NEW — 33× faster interface than τ=1
    )
    print(f"[setup] α={3.0}  γ={5.0}  τ={0.03}  "
          f"(m_max ≈ {(3.0/np.pi)*np.arctan(5.0*(args.u_inf+1.0)):.2f})")

    # ------------------------------------------------------------------
    # Figure / writer
    # ------------------------------------------------------------------
    frost_cmap = LinearSegmentedColormap.from_list(
        "frost",
        ["#050510", "#0f1f3a", "#1f5fa0", "#50b8e0", "#d0f0ff", "#ffffff"],
    )

    k_top = nz // 4  # top-down slice height
    j_cut = ny // 2  # cross-section position

    fig, (ax_top, ax_cross) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#101018")

    blank_top = np.zeros((ny, nx), dtype=np.float32)
    blank_cross = np.zeros((nz, nx), dtype=np.float32)

    im_top = ax_top.imshow(
        blank_top,
        cmap=frost_cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation="bilinear",
    )
    ax_top.set_title(f"top-down φ  (z = {k_top * dz * 1e6:.0f} μm)", color="white")
    ax_top.set_xlabel("x", color="white")
    ax_top.set_ylabel("y", color="white")
    ax_top.tick_params(colors="white")

    im_cross = ax_cross.imshow(
        blank_cross,
        cmap=frost_cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
    )
    ax_cross.set_title(f"cross-section  (y = {j_cut})", color="white")
    ax_cross.set_xlabel("x", color="white")
    ax_cross.set_ylabel(
        "z  (0 = glass, {:.0f} μm = room)".format(DOMAIN_Z * 1e6), color="white"
    )
    ax_cross.tick_params(colors="white")

    title = fig.suptitle("initialising…", color="white", fontsize=14)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Render loop (streaming: no frame accumulation)
    # ------------------------------------------------------------------
    do_render = not args.no_render
    if do_render and shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg not found in PATH; disabling video output.")
        do_render = False

    writer = None
    if do_render:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000)
        writer.setup(fig, args.output, dpi=100)

    total_frames = (steps + stride - 1) // stride
    t_start = time.time()
    phi_min, phi_max = 0.0, 0.0
    n_done = 0

    try:
        for frame_idx in range(total_frames):
            sim.step(stride, stream=0)
            phi = sim.get_phase()  # (nz, ny, nx)
            n_done += stride
            phi_min = float(phi.min())
            phi_max = float(phi.max())

            im_top.set_array(phi[k_top])
            im_cross.set_array(phi[:, j_cut, :])
            title.set_text(
                f"frost  step {n_done:4d}/{steps}   "
                f"t = {sim.get_sim_time():.4g} s   "
                f"φ_max = {phi_max:.3f}"
            )

            if writer is not None:
                writer.grab_frame(facecolor=fig.get_facecolor())

            if (frame_idx + 1) % max(1, total_frames // 10) == 0:
                elapsed = time.time() - t_start
                prog = (frame_idx + 1) / total_frames
                eta = elapsed / prog * (1 - prog)
                print(
                    f"[run ] frame {frame_idx + 1:3d}/{total_frames}"
                    f"  φ∈[{phi_min:.3f},{phi_max:.3f}]"
                    f"  {elapsed:6.1f}s elapsed  {eta:6.1f}s eta"
                )
    finally:
        if writer is not None:
            writer.finish()
        plt.close(fig)

    print(f"[done] sim finished in {time.time() - t_start:.1f}s")
    if do_render:
        print(f"[done] wrote {args.output}")

    # ------------------------------------------------------------------
    # Final state for further use
    # ------------------------------------------------------------------
    phi_final = sim.get_phase()
    npy_path = Path(args.output).with_suffix(".npy")
    np.save(npy_path, phi_final)
    print(
        f"[done] saved final phase field → {npy_path}"
        f"  shape={phi_final.shape}  "
        f"max={phi_final.max():.3f}"
    )

    if phi_max < 0.6:
        print("[warn] φ_max < 0.6 at end of run. Growth stalled. Try:")
        print("       • increase --u-inf (try 0.8)")
        print("       • increase --steps (try 2–3× as many)")
        print("       • decrease --eps-z (try 0.2) for more z-growth")
        print("       • if φ_max dropped below 0.5 at any point, seeds died:")
        print("         check that alpha > 1 and that seed radius > W0/(2(m-0.5))")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preset", choices=list(PRESETS), default="demo")
    p.add_argument("--nx", type=int, default=None, help="override preset's nx")
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz", type=int, default=None)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override preset's number of external steps",
    )
    p.add_argument("--output", type=str, default="frost.mp4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--u-inf", type=float, default=0.5, dest="u_inf")
    p.add_argument("--eps6", type=float, default=0.05)
    p.add_argument("--eps-z", type=float, default=0.5, dest="eps_z")
    p.add_argument(
        "--dt",
        type=float,
        default=1.0e-4,
        help="external time step, seconds (backend sub-steps to respect CFL)",
    )
    p.add_argument(
        "--no-render", action="store_true", help="skip video output; still saves .npy"
    )
    args = p.parse_args()

    run_and_render(args)


if __name__ == "__main__":
    main()
