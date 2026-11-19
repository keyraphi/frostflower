#pragma once
#include <cstddef>
#include <cstdint>

namespace frostflower::backend {

class SimulatorImpl;

struct SimParams {
    float u_inf;
    float eps6;
    float eps_z;
    float diffusion_D;
    float latent_K;
    float alpha;
    float gamma;
    float dt;
};

SimulatorImpl* create(size_t nx, size_t ny, size_t nz,
                      float dx, float dy, float dz,
                      uint64_t stream);
void destroy(SimulatorImpl* impl);

void set_initial_conditions(SimulatorImpl* impl,
                            const float* seed_map,
                            const float* glass_temp_map,
                            const float* orientation_map,
                            const float* growth_mask,
                            uint64_t stream);

void set_parameters(SimulatorImpl* impl, const SimParams& params);

void step(SimulatorImpl* impl, int num_steps, uint64_t stream);

void copy_phase_to_host(const SimulatorImpl* impl, float* out, uint64_t stream);
void copy_supersaturation_to_host(const SimulatorImpl* impl, float* out, uint64_t stream);

int   get_num_steps(const SimulatorImpl* impl);
float get_sim_time (const SimulatorImpl* impl);

} // namespace frostflower::backend
