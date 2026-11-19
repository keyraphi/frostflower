"""
GPU-accelerated 3D phase-field window-frost simulator.

The solver integrates a coupled anisotropic Allen-Cahn phase-field
equation and a Fick diffusion equation for water-vapour
supersaturation on an anisotropic voxel grid (typically dz << dx = dy),
which is the natural representation of a thin laterally spreading
frost film on a cold pane of glass.

Accessors return host NumPy arrays of shape ``(nz, ny, nx)`` so that a
horizontal slice at height ``k`` is ``phase[k, :, :]`` and can be fed
directly to ``matplotlib.pyplot.imshow``, ``igl``, or Blender's
Geometry Nodes ``Volume to Mesh``.
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


class FrostSimulator:
    """
    Stateful 3D frost growth simulator on an anisotropic voxel grid.

    Grid geometry
    -------------
    ``nx`` cells along x, physical extent ``nx * dx``
    ``ny`` cells along y, physical extent ``ny * dy``
    ``nz`` cells along z, physical extent ``nz * dz``

    Typical physical parameters for a 1 m window at 2048 lateral samples:

        nx = ny = 2048, dx = dy = 5e-4 m   # 0.5 mm lateral
        nz = 64,        dz      = 7.8e-6 m # 7.8 um vertical

    so the frost layer and its vapour boundary layer are resolved
    vertically while the lateral domain covers the full pane.
    """

    def __init__(self, nx: int, ny: int, nz: int, dx: float = 0.0005000000237487257, dy: float = 0.0005000000237487257, dz: float = 4.999999873689376e-06, stream: int = 0) -> None:
        """
        Construct a new simulator and allocate device buffers.

        Parameters
        ----------
        nx, ny, nz : int
            Grid resolution. All three must be > 0.
        dx, dy : float, optional
            Lateral voxel spacing, in metres. Default 5e-4 m (0.5 mm).
        dz : float, optional
            Vertical voxel spacing, in metres. Should be much smaller
            than ``dx`` so the thin frost film is vertically resolved.
            Default 5e-6 m (5 um).
        stream : int, optional
            CUDA stream handle for allocations and subsequent kernel
            launches. In PyTorch use ``torch.cuda.current_stream().cuda_stream``.
            Default 0 (the default/null stream).
        """

    def set_initial_conditions(self, seed_map: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], glass_temp_map: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], orientation_map: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], growth_mask: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')] | None = None, stream: int = 0) -> None:
        """
        Set the initial conditions of the simulation.

        Parameters
        ----------
        seed_map : ndarray[float32, (ny, nx)]
            Nucleation map. Any voxel in the two bottom layers where
            ``seed_map > 0.5`` is set to ice (``phi = 1``). Use this
            to paint scratches, dust, or procedural nuclei.
        glass_temp_map : ndarray[float32, (ny, nx)]
            Spatially varying glass undercooling, in degrees K below
            the dew point. Positive values promote growth; negative
            values suppress it. Use a radial gradient for a warm frame
            and cold centre, or a flat field for a uniform pane.
        orientation_map : ndarray[float32, (ny, nx)]
            Local six-fold crystal rotation in ``[0, pi/3)``, typically
            generated from a Voronoi decomposition of random seeds.
            Different cells will grow visually distinct frost grains.
        growth_mask : ndarray[float32, (ny, nx)], optional
            Smooth multiplier applied to the driving force. 1 = full
            growth, 0 = no growth. Useful for keep-out zones or
            painted trajectories. If ``None``, growth is unconstrained.
        stream : int, optional
            CUDA stream handle. Default 0.

        All input maps must have shape exactly ``(ny, nx)`` and be
        C-contiguous float32 NumPy arrays on the CPU.
        """

    def set_parameters(self, u_inf: float, eps6: float, eps_z: float, diffusion_D: float, latent_K: float, alpha: float, gamma: float, dt: float) -> None:
        """
        Set the physical parameters of the phase-field model.

        Parameters
        ----------
        u_inf : float
            Ambient supersaturation at the top boundary (room air
            reservoir). Higher = faster growth.
        eps6 : float
            Strength of the six-fold hexagonal anisotropy. Controls
            the sharpness of the dendritic tips (0 = isotropic, up to
            ~1/35 before the anisotropy function becomes non-convex).
        eps_z : float
            Out-of-plane anisotropy penalty. The key morphology knob:
            values of order 0.5-1.0 strongly suppress vertical growth
            and produce the quasi-2D fern/feather look of real window
            frost. Too small = coral blobs; too large = pure 2D.
        diffusion_D : float
            Vapour diffusion coefficient. Together with ``dt`` this
            sets the width of the depletion boundary layer around each
            frost needle.
        latent_K : float
            Latent heat coupling strength. Positive values reproduce
            the Mullins-Sekerka instability that produces side
            branches.
        alpha : float
            Prefactor of the arctan driving-force term.
        gamma : float
            Sharpness of the arctan driving-force nonlinearity.
        dt : float
            Target timestep. The backend may take internally smaller
            substeps (e.g. for the explicit phase-field update) and an
            implicit solve for diffusion, so this is the amount of
            physical time advanced per external ``step()`` call.
        """

    def step(self, num_steps: int, stream: int = 0) -> None:
        """
        Advance the simulation by ``num_steps`` external timesteps.

        Parameters
        ----------
        num_steps : int
            Number of external steps to take. Each step advances
            physical time by ``dt``. Must be >= 0.
        stream : int, optional
            CUDA stream handle. The backend synchronises before
            returning so the caller can immediately read back grids.
        """

    def get_phase(self, stream: int = 0) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', device='cpu')]:
        """
        Return the current phase field as a host NumPy array.

        Returns
        -------
        phi : ndarray[float32, (nz, ny, nx)]
            Phase field in ``[0, 1]``. ``phi[k, j, i]`` is the value at
            grid cell ``(i, j, k)``; horizontal slices ``phi[k]`` are
            ``(ny, nx)`` image-shaped.
        """

    def get_supersaturation(self, stream: int = 0) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', device='cpu')]:
        """
        Return the current supersaturation field as a host NumPy array.

        Returns
        -------
        u : ndarray[float32, (nz, ny, nx)]
            Supersaturation u(x, y, z). Values near ``u_inf`` far from
            any frost; depleted (near 0) in the boundary layer around
            growing needles.
        """

    def get_shape(self) -> tuple[int, int, int]:
        """Return the grid resolution as ``(nx, ny, nz)``."""

    def get_num_steps(self) -> int:
        """Return the number of external steps taken so far."""

    def get_sim_time(self) -> float:
        """
        Return the total physical time advanced so far.

        Equal to ``get_num_steps() * dt``, where ``dt`` is the value
        most recently set via :meth:`set_parameters`.
        """

    def __repr__(self) -> "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >": ...
