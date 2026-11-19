#include "frostflower_backend.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <format>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Aliases
// ---------------------------------------------------------------------------
// Input maps: NumPy float32 arrays on CPU.
using Map2DIn = nb::ndarray<nb::numpy, float, nb::shape<-1, -1>, nb::c_contig,
                            nb::device::cpu>;

// Outputs: NumPy arrays on the host, owned by the Python object.
using Field3DOut = nb::ndarray<nb::numpy, float, nb::shape<-1, -1, -1>,
                               nb::c_contig, nb::device::cpu>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

size_t positive_dim(size_t v, const char *name) {
  if (v == 0)
    throw std::invalid_argument(std::format("{} must be > 0", name));
  return v;
}

float positive_spacing(float v, const char *name) {
  if (!(v > 0.0f))
    throw std::invalid_argument(std::format("{} must be > 0", name));
  return v;
}

nb::capsule float_owner(float *p) {
  return nb::capsule(
      p, [](void *q) noexcept { delete[] static_cast<float *>(q); });
}

} // namespace

// ---------------------------------------------------------------------------
// FrostSimulator
// ---------------------------------------------------------------------------
class FrostSimulator {
public:
  FrostSimulator(size_t nx, size_t ny, size_t nz, float dx = 5e-4f,
                 float dy = 5e-4f, float dz = 5e-6f, uint64_t stream = 0)
      : m_nx(positive_dim(nx, "nx")), m_ny(positive_dim(ny, "ny")),
        m_nz(positive_dim(nz, "nz")), m_dx(positive_spacing(dx, "dx")),
        m_dy(positive_spacing(dy, "dy")), m_dz(positive_spacing(dz, "dz")),
        m_impl(frostflower::backend::create(m_nx, m_ny, m_nz, m_dx, m_dy, m_dz,
                                            stream)) {}

  ~FrostSimulator() { frostflower::backend::destroy(m_impl); }

  FrostSimulator(const FrostSimulator &) = delete;
  FrostSimulator &operator=(const FrostSimulator &) = delete;

  // ---- Initial conditions -----------------------------------------------
  void set_initial_conditions(const Map2DIn &seed_map,
                              const Map2DIn &glass_temp_map,
                              const Map2DIn &orientation_map,
                              std::optional<Map2DIn> growth_mask,
                              uint64_t stream = 0) {
    auto check = [&](const Map2DIn &m, const char *name) {
      if (m.shape(0) != m_ny || m.shape(1) != m_nx)
        throw std::invalid_argument(
            std::format("{} must have shape (ny={}, nx={}); got ({}, {})", name,
                        m_ny, m_nx, m.shape(0), m.shape(1)));
    };
    check(seed_map, "seed_map");
    check(glass_temp_map, "glass_temp_map");
    check(orientation_map, "orientation_map");

    const float *gm = nullptr;
    if (growth_mask) {
      check(*growth_mask, "growth_mask");
      gm = growth_mask->data();
    }

    frostflower::backend::set_initial_conditions(
        m_impl, seed_map.data(), glass_temp_map.data(), orientation_map.data(),
        gm, stream);
  }

  // ---- Physics parameters -----------------------------------------------
  void set_parameters(float u_inf, float eps6, float eps_z, float diffusion_D,
                      float latent_K, float alpha, float gamma, float dt) {
    frostflower::backend::SimParams p{
        .u_inf = u_inf,
        .eps6 = eps6,
        .eps_z = eps_z,
        .diffusion_D = diffusion_D,
        .latent_K = latent_K,
        .alpha = alpha,
        .gamma = gamma,
        .dt = dt,
    };
    frostflower::backend::set_parameters(m_impl, p);
  }

  // ---- Stepping ---------------------------------------------------------
  void step(int num_steps, uint64_t stream = 0) {
    if (num_steps < 0)
      throw std::invalid_argument("num_steps must be >= 0");
    frostflower::backend::step(m_impl, num_steps, stream);
  }

  // ---- Grid accessors ---------------------------------------------------
  Field3DOut get_phase(uint64_t stream = 0) const {
    const size_t total = m_nx * m_ny * m_nz;
    auto *data = new float[total];
    frostflower::backend::copy_phase_to_host(m_impl, data, stream);
    return Field3DOut(data, {m_nz, m_ny, m_nx}, float_owner(data));
  }

  Field3DOut get_supersaturation(uint64_t stream = 0) const {
    const size_t total = m_nx * m_ny * m_nz;
    auto *data = new float[total];
    frostflower::backend::copy_supersaturation_to_host(m_impl, data, stream);
    return Field3DOut(data, {m_nz, m_ny, m_nx}, float_owner(data));
  }

  // ---- Metadata ---------------------------------------------------------
  [[nodiscard]] auto get_shape() const -> std::tuple<size_t, size_t, size_t> {
    return {m_nx, m_ny, m_nz};
  }

  [[nodiscard]] auto get_num_steps() const -> int {
    return frostflower::backend::get_num_steps(m_impl);
  }

  [[nodiscard]] auto get_sim_time() const -> float {
    return frostflower::backend::get_sim_time(m_impl);
  }

private:
  size_t m_nx, m_ny, m_nz;
  float m_dx, m_dy, m_dz;
  frostflower::backend::SimulatorImpl *m_impl;
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
NB_MODULE(frostflower_module, m) {
  m.doc() = R"doc(
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
    )doc";

  m.attr("__version__") = "0.1.0";

  nb::class_<FrostSimulator>(m, "FrostSimulator", R"doc(
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
    )doc")

      .def(nb::init<size_t, size_t, size_t, float, float, float, uint64_t>(),
           "nx"_a, "ny"_a, "nz"_a, "dx"_a = 5e-4f, "dy"_a = 5e-4f,
           "dz"_a = 5e-6f, "stream"_a = 0,
           R"doc(
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
         )doc")

      // ---- set_initial_conditions
      // --------------------------------------------
      .def("set_initial_conditions", &FrostSimulator::set_initial_conditions,
           "seed_map"_a, "glass_temp_map"_a, "orientation_map"_a,
           "growth_mask"_a = nb::none(), "stream"_a = 0,
           R"doc(
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
         )doc")

      // ---- set_parameters
      // ----------------------------------------------------
      .def("set_parameters", &FrostSimulator::set_parameters, "u_inf"_a,
           "eps6"_a, "eps_z"_a, "diffusion_D"_a, "latent_K"_a, "alpha"_a,
           "gamma"_a, "dt"_a,
           R"doc(
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
         )doc")

      // ---- step
      // --------------------------------------------------------------
      .def("step", &FrostSimulator::step, "num_steps"_a, "stream"_a = 0,
           R"doc(
            Advance the simulation by ``num_steps`` external timesteps.

            Parameters
            ----------
            num_steps : int
                Number of external steps to take. Each step advances
                physical time by ``dt``. Must be >= 0.
            stream : int, optional
                CUDA stream handle. The backend synchronises before
                returning so the caller can immediately read back grids.
         )doc")

      // ---- get_phase
      // ---------------------------------------------------------
      .def("get_phase", &FrostSimulator::get_phase, "stream"_a = 0,
           R"doc(
            Return the current phase field as a host NumPy array.

            Returns
            -------
            phi : ndarray[float32, (nz, ny, nx)]
                Phase field in ``[0, 1]``. ``phi[k, j, i]`` is the value at
                grid cell ``(i, j, k)``; horizontal slices ``phi[k]`` are
                ``(ny, nx)`` image-shaped.
         )doc")

      // ---- get_supersaturation
      // -----------------------------------------------
      .def("get_supersaturation", &FrostSimulator::get_supersaturation,
           "stream"_a = 0,
           R"doc(
            Return the current supersaturation field as a host NumPy array.

            Returns
            -------
            u : ndarray[float32, (nz, ny, nx)]
                Supersaturation u(x, y, z). Values near ``u_inf`` far from
                any frost; depleted (near 0) in the boundary layer around
                growing needles.
         )doc")

      // ---- metadata
      // ----------------------------------------------------------
      .def("get_shape", &FrostSimulator::get_shape, R"doc(
            Return the grid resolution as ``(nx, ny, nz)``.
         )doc")
      .def("get_num_steps", &FrostSimulator::get_num_steps, R"doc(
            Return the number of external steps taken so far.
         )doc")
      .def("get_sim_time", &FrostSimulator::get_sim_time, R"doc(
            Return the total physical time advanced so far.

            Equal to ``get_num_steps() * dt``, where ``dt`` is the value
            most recently set via :meth:`set_parameters`.
         )doc")
      .def("__repr__", [](const FrostSimulator &self) {
        auto [nx, ny, nz] = self.get_shape();
        return std::format(
            "<FrostSimulator shape=({}, {}, {}), steps={}, t={:.4g}s>", nx, ny,
            nz, self.get_num_steps(), self.get_sim_time());
      });
}
