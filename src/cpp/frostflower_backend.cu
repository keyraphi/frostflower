#include "frostflower_backend.h"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)

namespace frostflower::backend {

namespace {

// POD struct passed by value to every kernel.
struct KernelArgs {
  int nx, ny, nz;
  float dx, dy, dz;
  float W0, z0;
  float eps6, eps_z;
  float alpha, gamma;
  float diffusion_D, latent_K;
  float u_inf;
  float tau;
  float *phi;
  float *phi_new;
  float *u;
  float *u_new;
  float *J_anis;
  float *glass_temp;
  float *growth_mask;
  int has_growth_mask;
};

__device__ __forceinline__ int dev_idx(const KernelArgs &a, int i, int j,
                                       int k) {
  return (k * a.ny + j) * a.nx + i;
}

// Six-fold hexagonal Chebyshev polynomial
__device__ __forceinline__ float cheb6(float nx, float ny) {
  float nx2 = nx * nx, ny2 = ny * ny;
  float nx4 = nx2 * nx2, ny4 = ny2 * ny2;
  return nx4 * nx2 - 15.f * nx4 * ny2 + 15.f * nx2 * ny4 - ny4 * ny2;
}

__device__ __forceinline__ void cheb6_grad(float nx, float ny, float &df6_dnx,
                                           float &df6_dny) {
  float nx2 = nx * nx, ny2 = ny * ny;
  float nx3 = nx2 * nx, ny3 = ny2 * ny;
  float nx4 = nx2 * nx2, ny4 = ny2 * ny2;
  float nx5 = nx4 * nx, ny5 = ny4 * ny;
  df6_dnx = 6.f * nx5 - 60.f * nx3 * ny2 + 30.f * nx * ny4;
  df6_dny = -30.f * nx4 * ny + 60.f * nx2 * ny3 - 6.f * ny5;
}

__device__ __forceinline__ void
compute_W_and_g_tan(float nx, float ny, float nz, float W0, float eps6,
                    float eps_z, float &W, float &gtx, float &gty, float &gtz) {
  float f6 = cheb6(nx, ny);
  float df6_dnx, df6_dny;
  cheb6_grad(nx, ny, df6_dnx, df6_dny);

  W = W0 * (1.f + eps6 * f6 + eps_z * nz * nz);

  // ∂W/∂n
  float gx = W0 * eps6 * df6_dnx;
  float gy = W0 * eps6 * df6_dny;
  float gz = W0 * 2.f * eps_z * nz;

  // Tangential projection: g_tan = g - (g·n)·n
  float gn_n = gx * nx + gy * ny + gz * nz;
  gtx = gx - gn_n * nx;
  gty = gy - gn_n * ny;
  gtz = gz - gn_n * nz;
}

// -----------------------------------------------------------------------------
// Kernel: initial conditions
// -----------------------------------------------------------------------------
__global__ void init_kernel(KernelArgs a, const float *__restrict__ seed_map,
                            const float *__restrict__ glass_temp_map,
                            const float *__restrict__ orientation_map,
                            const float *__restrict__ growth_mask, int has_gm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= a.nx || j >= a.ny || k >= a.nz)
    return;

  int ij = j * a.nx + i;
  int gid = dev_idx(a, i, j, k);

  // Seeds: φ = 1 in the two bottom layers wherever seed_map > 0.5
  float phi_val = (k < 2 && seed_map[ij] > 0.5f) ? 1.f : 0.f;
  a.phi[gid] = phi_val;
  a.phi_new[gid] = phi_val;
  a.u[gid] = a.u_inf;
  a.u_new[gid] = a.u_inf;

  // 2D maps copied only when k == 0
  if (k == 0) {
    a.glass_temp[ij] = glass_temp_map[ij];
    if (has_gm)
      a.growth_mask[ij] = growth_mask[ij];
    else
      a.growth_mask[ij] = 1.f;
  }
}

// -----------------------------------------------------------------------------
// Kernel: anisotropy flux J = |∇φ| · W · g_tan (stored as float3 per cell)
// -----------------------------------------------------------------------------
__global__ void compute_J_anis_kernel(KernelArgs a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= a.nx || j >= a.ny || k >= a.nz)
    return;

  auto P = [&](int ii, int jj, int kk) {
    ii = max(0, min(a.nx - 1, ii));
    jj = max(0, min(a.ny - 1, jj));
    kk = max(0, min(a.nz - 1, kk));
    return a.phi[(kk * a.ny + jj) * a.nx + ii];
  };

  float gx = (P(i + 1, j, k) - P(i - 1, j, k)) / (2.f * a.dx);
  float gy = (P(i, j + 1, k) - P(i, j - 1, k)) / (2.f * a.dy);
  float gz = (P(i, j, k + 1) - P(i, j, k - 1)) / (2.f * a.dz);

  float gnorm_sq = gx * gx + gy * gy + gz * gz + 1e-20f;
  float gnorm = sqrtf(gnorm_sq);

  float nx = gx / gnorm;
  float ny = gy / gnorm;
  float nz = gz / gnorm;

  float W, gtx, gty, gtz;
  compute_W_and_g_tan(nx, ny, nz, a.W0, a.eps6, a.eps_z, W, gtx, gty, gtz);

  int base = 3 * dev_idx(a, i, j, k);
  a.J_anis[base + 0] = gnorm * W * gtx;
  a.J_anis[base + 1] = gnorm * W * gty;
  a.J_anis[base + 2] = gnorm * W * gtz;
}

// -----------------------------------------------------------------------------
// Kernel: one explicit phase-field step
// -----------------------------------------------------------------------------
__global__ void phase_field_kernel(KernelArgs a, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= a.nx || j >= a.ny || k >= a.nz)
    return;

  if (k == a.nz - 1) {
    a.phi_new[dev_idx(a, i, j, k)] = 0.f;
    return;
  }

  auto P = [&](int ii, int jj, int kk) {
    ii = max(0, min(a.nx - 1, ii));
    jj = max(0, min(a.ny - 1, jj));
    kk = max(0, min(a.nz - 1, kk));
    return a.phi[(kk * a.ny + jj) * a.nx + ii];
  };

  float phi = P(i, j, k);
  float phi_xm = P(i - 1, j, k);
  float phi_xp = P(i + 1, j, k);
  float phi_ym = P(i, j - 1, k);
  float phi_yp = P(i, j + 1, k);
  float phi_zm = P(i, j, k - 1);
  float phi_zp = P(i, j, k + 1);

  float lap = (phi_xp - 2.f * phi + phi_xm) / (a.dx * a.dx) +
              (phi_yp - 2.f * phi + phi_ym) / (a.dy * a.dy) +
              (phi_zp - 2.f * phi + phi_zm) / (a.dz * a.dz);

  // Anisotropy divergence
  auto Jx = [&](int ii, int jj, int kk) {
    ii = max(0, min(a.nx - 1, ii));
    jj = max(0, min(a.ny - 1, jj));
    kk = max(0, min(a.nz - 1, kk));
    return a.J_anis[3 * ((kk * a.ny + jj) * a.nx + ii) + 0];
  };
  auto Jy = [&](int ii, int jj, int kk) {
    ii = max(0, min(a.nx - 1, ii));
    jj = max(0, min(a.ny - 1, jj));
    kk = max(0, min(a.nz - 1, kk));
    return a.J_anis[3 * ((kk * a.ny + jj) * a.nx + ii) + 1];
  };
  auto Jz = [&](int ii, int jj, int kk) {
    ii = max(0, min(a.nx - 1, ii));
    jj = max(0, min(a.ny - 1, jj));
    kk = max(0, min(a.nz - 1, kk));
    return a.J_anis[3 * ((kk * a.ny + jj) * a.nx + ii) + 2];
  };

  float div_J = (Jx(i + 1, j, k) - Jx(i - 1, j, k)) / (2.f * a.dx) +
                (Jy(i, j + 1, k) - Jy(i, j - 1, k)) / (2.f * a.dy) +
                (Jz(i, j, k + 1) - Jz(i, j, k - 1)) / (2.f * a.dz);

  // Driving force m(u, T_glass)
  int ij = j * a.nx + i;
  float u_here = a.u[dev_idx(a, i, j, k)];
  float T_glass = a.glass_temp[ij];
  float z_phys = (k + 0.5f) * a.dz;
  float delta_T = T_glass * expf(-z_phys / a.z0);
  float m =
      (a.alpha / 3.14159265358979323846f) * atanf(a.gamma * (u_here + delta_T));
  if (a.has_growth_mask)
    m *= a.growth_mask[ij];

  // Recompute W for the Laplacian prefactor
  float gx = (phi_xp - phi_xm) / (2.f * a.dx);
  float gy = (phi_yp - phi_ym) / (2.f * a.dy);
  float gz = (phi_zp - phi_zm) / (2.f * a.dz);
  float gn_sq = gx * gx + gy * gy + gz * gz + 1e-20f;
  float gnorm = sqrtf(gn_sq);
  float W, unused1, unused2, unused3;
  compute_W_and_g_tan(gx / gnorm, gy / gnorm, gz / gnorm, a.W0, a.eps6, a.eps_z,
                      W, unused1, unused2, unused3);

  float rhs = W * W * lap + div_J + phi * (1.f - phi) * (phi - 0.5f + m);

  float phi_next = phi + (dt / a.tau) * rhs;
  phi_next = fmaxf(0.f, fminf(1.f, phi_next));

  a.phi_new[dev_idx(a, i, j, k)] = phi_next;
}

// -----------------------------------------------------------------------------
// Kernel: one explicit diffusion step (with latent-heat source from Δφ)
// -----------------------------------------------------------------------------
__global__ void diffusion_kernel(KernelArgs a, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= a.nx || j >= a.ny || k >= a.nz)
    return;

  // Top Dirichlet: u = u_inf
  if (k == a.nz - 1) {
    a.u_new[dev_idx(a, i, j, k)] = a.u_inf;
    return;
  }

  auto U = [&](int ii, int jj, int kk) {
    ii = max(0, min(a.nx - 1, ii));
    jj = max(0, min(a.ny - 1, jj));
    kk = max(0, min(a.nz - 1, kk));
    return a.u[(kk * a.ny + jj) * a.nx + ii];
  };

  float u = U(i, j, k);
  float lap = (U(i + 1, j, k) - 2.f * u + U(i - 1, j, k)) / (a.dx * a.dx) +
              (U(i, j + 1, k) - 2.f * u + U(i, j - 1, k)) / (a.dy * a.dy) +
              (U(i, j, k + 1) - 2.f * u + U(i, j, k - 1)) / (a.dz * a.dz);

  // Latent heat: -K · (φ_new − φ_old). After the phase-field swap,
  // a.phi holds φ^{n+1} and a.phi_new holds φ^n.
  int gid = dev_idx(a, i, j, k);
  float dphi = a.phi[gid] - a.phi_new[gid];

  a.u_new[gid] = u + dt * a.diffusion_D * lap - a.latent_K * dphi;
}

} // namespace

// =============================================================================
// Host-side state
// =============================================================================
class SimulatorImpl {
public:
  size_t nx = 0, ny = 0, nz = 0;
  float dx = 0, dy = 0, dz = 0;
  float W0 = 0, z0 = 0;

  SimParams params{};

  thrust::device_vector<float> phi, phi_new;
  thrust::device_vector<float> u, u_new;
  thrust::device_vector<float> J_anis;      // 3 · nx·ny·nz
  thrust::device_vector<float> glass_temp;  // nx·ny
  thrust::device_vector<float> growth_mask; // nx·ny

  bool has_growth_mask = false;
  int step_count = 0;

  KernelArgs make_args() {
    KernelArgs a{};
    a.nx = (int)nx;
    a.ny = (int)ny;
    a.nz = (int)nz;
    a.dx = dx;
    a.dy = dy;
    a.dz = dz;
    a.W0 = W0;
    a.z0 = z0;
    a.eps6 = params.eps6;
    a.eps_z = params.eps_z;
    a.alpha = params.alpha;
    a.gamma = params.gamma;
    a.diffusion_D = params.diffusion_D;
    a.latent_K = params.latent_K;
    a.u_inf = params.u_inf;
    a.tau = params.tau;
    a.phi = thrust::raw_pointer_cast(phi.data());
    a.phi_new = thrust::raw_pointer_cast(phi_new.data());
    a.u = thrust::raw_pointer_cast(u.data());
    a.u_new = thrust::raw_pointer_cast(u_new.data());
    a.J_anis = thrust::raw_pointer_cast(J_anis.data());
    a.glass_temp = thrust::raw_pointer_cast(glass_temp.data());
    a.growth_mask = thrust::raw_pointer_cast(growth_mask.data());
    a.has_growth_mask = has_growth_mask ? 1 : 0;
    return a;
  }
};

namespace {

// Explicit-Euler CFL estimates with conservative safety factors.
float dt_phi_max(const SimulatorImpl &d) {
  float W_max =
      d.W0 * (1.f + std::abs(d.params.eps6) + std::abs(d.params.eps_z));
  float k = 2.f / (d.dx * d.dx) + 2.f / (d.dy * d.dy) + 2.f / (d.dz * d.dz);
  return 0.30f * d.params.tau / std::max(W_max * W_max * k, 1e-30f);
}

float dt_u_max(const SimulatorImpl &d) {
  float k = 2.f / (d.dx * d.dx) + 2.f / (d.dy * d.dy) + 2.f / (d.dz * d.dz);
  return 0.40f / std::max(d.params.diffusion_D * k, 1e-30f);
}

} // namespace

// =============================================================================
// Public C++ API
// =============================================================================

SimulatorImpl *create(size_t nx, size_t ny, size_t nz, float dx, float dy,
                      float dz, uint64_t /*stream*/) {
  if (nx == 0 || ny == 0 || nz == 0)
    throw std::invalid_argument("all grid dimensions must be > 0");

  auto *impl = new SimulatorImpl();
  impl->nx = nx;
  impl->ny = ny;
  impl->nz = nz;
  impl->dx = dx;
  impl->dy = dy;
  impl->dz = dz;

  // Diffuse-interface width: 3 lateral cells. See file header note.
  impl->W0 = std::max(3.f * std::min(dx, dy), 3.f * dz);
  impl->z0 = 1e-3f; // 1 mm undercooling decay length

  const size_t N = nx * ny * nz;
  impl->phi.resize(N, 0.f);
  impl->phi_new.resize(N, 0.f);
  impl->u.resize(N, 0.f);
  impl->u_new.resize(N, 0.f);
  impl->J_anis.resize(3 * N, 0.f);
  impl->glass_temp.resize(nx * ny, 0.f);
  impl->growth_mask.resize(nx * ny, 1.f);

  impl->params = SimParams{
      .u_inf = 0.5f,
      .eps6 = 0.05f,
      .eps_z = 0.5f,
      .diffusion_D = 2e-5f,
      .latent_K = 0.1f,
      .alpha = 3.0f,
      .gamma = 5.0f,
      .dt = 1e-4f,
      .tau = 0.03f,
  };

  return impl;
}

void destroy(SimulatorImpl *impl) { delete impl; }

void set_initial_conditions(SimulatorImpl *impl, const float *seed_map,
                            const float *glass_temp_map,
                            const float *orientation_map,
                            const float *growth_mask, uint64_t /*stream*/) {
  auto &d = *impl;
  d.has_growth_mask = (growth_mask != nullptr);

  // Upload maps to device scratch
  thrust::device_vector<float> d_seed(seed_map, seed_map + d.nx * d.ny);
  thrust::device_vector<float> d_gt(glass_temp_map,
                                    glass_temp_map + d.nx * d.ny);
  thrust::device_vector<float> d_ori(orientation_map,
                                     orientation_map + d.nx * d.ny);
  thrust::device_vector<float> d_gm;
  if (d.has_growth_mask) {
    d_gm.assign(growth_mask, growth_mask + d.nx * d.ny);
  }

  auto a = d.make_args();

  dim3 block(16, 16, 4);
  dim3 grid((unsigned)((d.nx + block.x - 1) / block.x),
            (unsigned)((d.ny + block.y - 1) / block.y),
            (unsigned)((d.nz + block.z - 1) / block.z));

  init_kernel<<<grid, block>>>(
      a, thrust::raw_pointer_cast(d_seed.data()),
      thrust::raw_pointer_cast(d_gt.data()),
      thrust::raw_pointer_cast(d_ori.data()),
      d.has_growth_mask ? thrust::raw_pointer_cast(d_gm.data()) : nullptr,
      d.has_growth_mask ? 1 : 0);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void set_parameters(SimulatorImpl *impl, const SimParams &params) {
  impl->params = params;
}

void step(SimulatorImpl *impl, int num_steps, uint64_t /*stream*/) {
  if (num_steps < 0)
    throw std::invalid_argument("num_steps must be >= 0");
  auto &d = *impl;

  dim3 block(16, 16, 4);
  dim3 grid((unsigned)((d.nx + block.x - 1) / block.x),
            (unsigned)((d.ny + block.y - 1) / block.y),
            (unsigned)((d.nz + block.z - 1) / block.z));

  for (int s = 0; s < num_steps; ++s) {
    const float dt_user = d.params.dt;
    const float dt_p = dt_phi_max(d);
    const float dt_u = dt_u_max(d);
    const float dt_sub = std::min({dt_user, dt_p, dt_u});
    if (dt_sub <= 0.f)
      continue;

    const int n_sub = (int)std::ceil(dt_user / dt_sub);
    const float dt_actual = dt_user / n_sub;

    for (int sub = 0; sub < n_sub; ++sub) {
      // 1) Compute anisotropy flux from the current phase field
      {
        auto a = d.make_args();
        compute_J_anis_kernel<<<grid, block>>>(a);
        CUDA_CHECK(cudaGetLastError());
      }
      // 2) Advance phase field
      {
        auto a = d.make_args();
        phase_field_kernel<<<grid, block>>>(a, dt_actual);
        CUDA_CHECK(cudaGetLastError());
        d.phi.swap(d.phi_new);
      }
      // 3) Advance diffusion (uses the freshly updated φ for Δφ)
      {
        auto a = d.make_args();
        diffusion_kernel<<<grid, block>>>(a, dt_actual);
        CUDA_CHECK(cudaGetLastError());
        d.u.swap(d.u_new);
      }
    }
    d.step_count++;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void copy_phase_to_host(const SimulatorImpl *impl, float *out,
                        uint64_t /*stream*/) {
  thrust::copy(impl->phi.begin(), impl->phi.end(), out);
}

void copy_supersaturation_to_host(const SimulatorImpl *impl, float *out,
                                  uint64_t /*stream*/) {
  thrust::copy(impl->u.begin(), impl->u.end(), out);
}

int get_num_steps(const SimulatorImpl *impl) { return impl->step_count; }

float get_sim_time(const SimulatorImpl *impl) {
  return (float)impl->step_count * impl->params.dt;
}

} // namespace frostflower::backend
