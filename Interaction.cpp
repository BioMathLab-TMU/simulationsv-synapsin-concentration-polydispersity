#include "Synapsin.h"
#include <fstream>
#include "GP.h"
#include "GF.h"
#include "Vesicles.h"
#include "Interaction.h"
#include "MyUtility.h"
#include <cmath>
#include "Saver.h"
#include "Srock2Coefficients.h"
#include "ProjectionNeighborhood.h"
#include <queue>
#include <mutex>
#include <array>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <numbers>
#include <stdexcept>

namespace {

    std::uint64_t particleKey(Agent object_type, int id) {
        const auto type_key = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(object_type));
        const auto id_key = static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(id));
        return (type_key << 32U) | id_key;
    }

    std::uint64_t nextDeterministicBits(std::uint64_t &state) {

        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t value = state;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    std::array<double, 3> deterministicSeparationDirection(
            int i,
            int j,
            Agent object_type_i,
            Agent object_type_j) {
        const std::uint64_t key_i = particleKey(object_type_i, i);
        const std::uint64_t key_j = particleKey(object_type_j, j);

        if (key_i == key_j) {

            return {1.0, 0.0, 0.0};
        }

        const std::uint64_t low_key = std::min(key_i, key_j);
        const std::uint64_t high_key = std::max(key_i, key_j);
        std::uint64_t state = low_key
                              ^ (high_key + 0x9e3779b97f4a7c15ULL
                                 + (low_key << 6U) + (low_key >> 2U));

        const auto component = [&state]() {

            const double unit = static_cast<double>(
                                        nextDeterministicBits(state) >> 11U)
                                * (1.0 / 9007199254740992.0);
            return 2.0 * unit - 1.0;
        };

        std::array<double, 3> direction = {
                component(), component(), component()
        };
        double norm = std::sqrt(direction[0] * direction[0]
                                + direction[1] * direction[1]
                                + direction[2] * direction[2]);

        if (norm < 1e-12) {
            direction = {1.0, 0.0, 0.0};
            norm = 1.0;
        }

        const double orientation = key_i < key_j ? 1.0 : -1.0;
        direction[0] *= orientation / norm;
        direction[1] *= orientation / norm;
        direction[2] *= orientation / norm;
        return direction;
    }

    void addForceComponents(
            Interaction::ParticleForceComponents &target,
            const Interaction::ParticleForceComponents &source,
            double direction_sign = 1.0) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            target.electrostatic[axis] +=
                    direction_sign * source.electrostatic[axis];
            target.van_der_waals[axis] +=
                    direction_sign * source.van_der_waals[axis];
            target.hydrophobic[axis] +=
                    direction_sign * source.hydrophobic[axis];
            target.steric_core[axis] +=
                    direction_sign * source.steric_core[axis];
        }
    }

    std::vector<Interaction::ParticleForceComponents> &forceComponentStorage(
            Agent object_type) {
        if (object_type == Agent::Synapsin_agent) {
            return Interaction::synapsin_force_components;
        }
        return Interaction::vesicle_force_components;
    }

    using ParticleState = std::vector<std::array<double, 3>>;

    ParticleState captureParticleState(Agent object_type) {
        const auto &locations = object_type == Agent::Synapsin_agent
                ? Synapsin::locations : Vesicles::locations;
        ParticleState state(locations.size());
        for (std::size_t i = 0; i < locations.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                state[i][axis] = static_cast<double>(locations[i][axis]);
            }
        }
        return state;
    }

    void applyParticleState(
            ParticleState &desired_state,
            Agent object_type,
            Interaction::RuntimeDiagnostics &runtime,
            bool srock2_stage = false) {
        constexpr double reflection_detection_tolerance_nm = 1e-4;
        for (std::size_t i = 0; i < desired_state.size(); ++i) {
            const auto current = GF::getLocation(
                    object_type, static_cast<int>(i));
            std::array<double, 3> displacement{};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(desired_state[i][axis])) {
                    throw std::runtime_error(
                            "SK-ROCK generated a non-finite stage position");
                }
                displacement[axis] = desired_state[i][axis]
                                     - static_cast<double>(current[axis]);
            }

            GF::updateParticlePosition(
                    static_cast<int>(i), displacement, object_type);
            const auto actual = GF::getLocation(
                    object_type, static_cast<int>(i));
            const double correction = MyUtility::euclideanSize(
                    static_cast<double>(actual[0]) - desired_state[i][0],
                    static_cast<double>(actual[1]) - desired_state[i][1],
                    static_cast<double>(actual[2]) - desired_state[i][2]);
            if (correction > reflection_detection_tolerance_nm) {
                if (srock2_stage) {
                    ++runtime.srock2_internal_reflected_updates;
                    runtime.srock2_max_internal_boundary_correction_nm =
                            std::max(
                                    runtime.srock2_max_internal_boundary_correction_nm,
                                    correction);
                } else {
                    ++runtime.sk_rock_internal_reflected_updates;
                    runtime.sk_rock_max_internal_boundary_correction_nm =
                            std::max(
                                    runtime.sk_rock_max_internal_boundary_correction_nm,
                                    correction);
                }
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                desired_state[i][axis] = static_cast<double>(actual[axis]);
            }
        }
    }

    void rebuildParticleMappings() {
        Synapsin::whichSynapsinWhere();
        Vesicles::whichVesiclesWhere();
    }

    ParticleState currentDrift(Agent object_type) {
        const auto &forces = object_type == Agent::Synapsin_agent
                ? Synapsin::synapsin_forces : Vesicles::vesicles_forces;
        ParticleState drift(forces.size());
        for (std::size_t i = 0; i < forces.size(); ++i) {
            const double mobility = 1.0 / std::max(
                    GF::getDrag(object_type, static_cast<int>(i)), 1e-12);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                drift[i][axis] = forces[i][axis] * mobility;
            }
        }
        return drift;
    }

    ParticleState brownianIncrement(
            Agent object_type,
            double macro_step,
            std::normal_distribution<float> &normal_distribution,
            bool preserve_legacy_float_arithmetic = false) {
        const int count = GF::getNumberAgent(object_type);
        ParticleState increment(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            const double diffusion = object_type == Agent::Synapsin_agent
                    ? static_cast<double>(Synapsin::diffusion_coefficient)
                    : static_cast<double>(Vesicles::diffusion_coefficient[i]);
            const double noise_scale = preserve_legacy_float_arithmetic
                    ? static_cast<double>(std::sqrt(
                            2.0F * static_cast<float>(diffusion)
                            * static_cast<float>(macro_step)))
                    : std::sqrt(2.0 * diffusion * macro_step);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                increment[static_cast<std::size_t>(i)][axis] =
                        noise_scale * normal_distribution(GP::gen);
            }
        }
        return increment;
    }

    ParticleState affineSkRockStage(
            const ParticleState &older,
            const ParticleState &newer,
            const ParticleState &drift,
            double older_coefficient,
            double newer_coefficient,
            double drift_coefficient) {
        ParticleState stage(newer.size());
        for (std::size_t i = 0; i < newer.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                stage[i][axis] = older_coefficient * older[i][axis]
                                 + newer_coefficient * newer[i][axis]
                                 + drift_coefficient * drift[i][axis];
            }
        }
        return stage;
    }

}

std::vector<Interaction::ParticleForceComponents>
        Interaction::synapsin_force_components;
std::vector<Interaction::ParticleForceComponents>
        Interaction::vesicle_force_components;
Interaction::OverlapProjectionDiagnostics
        Interaction::last_overlap_projection_diagnostics{};
Interaction::RuntimeDiagnostics Interaction::runtime_diagnostics{};

Interaction::Interaction() = default;

void Interaction::resetRuntimeDiagnostics() {
    runtime_diagnostics = {};
    last_overlap_projection_diagnostics = {};
}

void Interaction::recordEulerParticleProposal(
        Agent object_type,
        const std::array<double, 3> &raw_drift,
        const std::array<double, 3> &used_drift,
        const std::array<double, 3> &brownian,
        const std::array<double, 3> &total_proposal,
        double radius_nm,
        double brownian_sigma_nm) {
    auto &runtime = runtime_diagnostics;
    ++runtime.euler_particle_proposal_count;

    const auto finiteVector = [](const std::array<double, 3> &values) {
        return std::all_of(
                values.begin(), values.end(),
                [](double value) { return std::isfinite(value); });
    };
    if (!finiteVector(raw_drift)
        || !finiteVector(used_drift)
        || !finiteVector(brownian)
        || !finiteVector(total_proposal)
        || !std::isfinite(radius_nm) || radius_nm <= 0.0
        || !std::isfinite(brownian_sigma_nm) || brownian_sigma_nm <= 0.0) {
        ++runtime.nonfinite_euler_proposal_count;
        const char *label = object_type == Agent::Synapsin_agent
                            ? "synapsin" : "vesicle";
        throw std::runtime_error(
                std::string("Non-finite Euler-Maruyama proposal for ") + label);
    }

    const double raw_norm = MyUtility::euclideanSize(raw_drift);
    const double used_norm = MyUtility::euclideanSize(used_drift);
    const double brownian_norm = MyUtility::euclideanSize(brownian);
    const double proposal_norm = MyUtility::euclideanSize(total_proposal);
    runtime.maximum_raw_drift_displacement_nm = std::max(
            runtime.maximum_raw_drift_displacement_nm, raw_norm);
    runtime.maximum_used_drift_displacement_nm = std::max(
            runtime.maximum_used_drift_displacement_nm, used_norm);
    runtime.maximum_used_drift_fraction_of_radius = std::max(
            runtime.maximum_used_drift_fraction_of_radius,
            used_norm / radius_nm);
    runtime.maximum_brownian_displacement_nm = std::max(
            runtime.maximum_brownian_displacement_nm, brownian_norm);
    runtime.maximum_brownian_displacement_sigma_units = std::max(
            runtime.maximum_brownian_displacement_sigma_units,
            brownian_norm / brownian_sigma_nm);
    runtime.maximum_total_proposal_displacement_nm = std::max(
            runtime.maximum_total_proposal_displacement_nm, proposal_norm);
    runtime.maximum_total_proposal_fraction_of_radius = std::max(
            runtime.maximum_total_proposal_fraction_of_radius,
            proposal_norm / radius_nm);
}

void Interaction::recordRuntimeStep() {
    auto &runtime = runtime_diagnostics;
    ++runtime.dynamic_step_count;

    const auto countLimited = [](const std::vector<double> &scales) {
        return static_cast<std::int64_t>(std::count_if(
                scales.begin(), scales.end(),
                [](double scale) { return scale < 1.0 - 1e-12; }));
    };
    const std::int64_t limited_synapsin = countLimited(Synapsin::drift_scale);
    const std::int64_t limited_vesicle = countLimited(Vesicles::drift_scale);
    const std::int64_t limited_total = limited_synapsin + limited_vesicle;
    const std::int64_t particle_total =
            static_cast<std::int64_t>(Synapsin::locations.size())
            + static_cast<std::int64_t>(Vesicles::locations.size());

    runtime.drift_limited_synapsin_events += limited_synapsin;
    runtime.drift_limited_vesicle_events += limited_vesicle;
    runtime.drift_limited_particle_events += limited_total;
    if (limited_total > 0) ++runtime.drift_limiter_active_steps;
    const double limited_fraction = particle_total > 0
            ? static_cast<double>(limited_total)
              / static_cast<double>(particle_total)
            : 0.0;
    runtime.sum_drift_limited_fraction += limited_fraction;
    runtime.max_drift_limited_fraction = std::max(
            runtime.max_drift_limited_fraction, limited_fraction);

    if (GP::enable_drift_component_diagnostics) {
        const auto recordLimiterSources = [&runtime](
                Agent object_type,
                const std::vector<double> &scales,
                const std::vector<std::array<double, 3>> &forces,
                const std::vector<ParticleForceComponents> &components) {
            const bool components_available = components.size() == forces.size();
            for (std::size_t i = 0; i < scales.size(); ++i) {
                if (scales[i] >= 1.0 - 1e-12) continue;

                ++runtime.drift_component_diagnostic_events;
                const double raw_to_cap_ratio = 1.0 / std::max(scales[i], 1e-300);
                runtime.sum_limited_raw_to_cap_ratio += raw_to_cap_ratio;
                runtime.max_limited_raw_to_cap_ratio = std::max(
                        runtime.max_limited_raw_to_cap_ratio,
                        raw_to_cap_ratio);

                if (!components_available || i >= components.size()) {
                    ++runtime.drift_component_unresolved_events;
                    continue;
                }

                const double force_norm = MyUtility::euclideanSize(forces[i]);
                if (!(force_norm > 1e-15) || !std::isfinite(force_norm)) {
                    ++runtime.drift_component_unresolved_events;
                    continue;
                }

                const auto alignedContribution = [&forces, &components, i, force_norm](
                        const std::array<double, 3> ParticleForceComponents::*member) {
                    const auto &component = components[i].*member;
                    double dot_product = 0.0;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        dot_product += component[axis] * forces[i][axis];
                    }
                    return std::max(0.0, dot_product / force_norm);
                };

                const std::array<double, 4> aligned = {
                        alignedContribution(&ParticleForceComponents::electrostatic),
                        alignedContribution(&ParticleForceComponents::van_der_waals),
                        alignedContribution(&ParticleForceComponents::hydrophobic),
                        alignedContribution(&ParticleForceComponents::steric_core)
                };
                const double aligned_sum = std::accumulate(
                        aligned.begin(), aligned.end(), 0.0);
                if (!(aligned_sum > 1e-15) || !std::isfinite(aligned_sum)) {
                    ++runtime.drift_component_unresolved_events;
                    continue;
                }

                runtime.sum_limited_electrostatic_aligned_share +=
                        aligned[0] / aligned_sum;
                runtime.sum_limited_van_der_waals_aligned_share +=
                        aligned[1] / aligned_sum;
                runtime.sum_limited_hydrophobic_aligned_share +=
                        aligned[2] / aligned_sum;
                runtime.sum_limited_steric_core_aligned_share +=
                        aligned[3] / aligned_sum;

                const auto dominant = static_cast<std::size_t>(std::distance(
                        aligned.begin(),
                        std::max_element(aligned.begin(), aligned.end())));
                if (dominant == 0) {
                    ++runtime.drift_dominant_electrostatic_events;
                } else if (dominant == 1) {
                    ++runtime.drift_dominant_van_der_waals_events;
                } else if (dominant == 2) {
                    ++runtime.drift_dominant_hydrophobic_events;
                } else {
                    ++runtime.drift_dominant_steric_core_events;
                }
            }
            static_cast<void>(object_type);
        };

        recordLimiterSources(
                Agent::Synapsin_agent,
                Synapsin::drift_scale,
                Synapsin::synapsin_forces,
                synapsin_force_components);
        recordLimiterSources(
                Agent::Vesicle_agent,
                Vesicles::drift_scale,
                Vesicles::vesicles_forces,
                vesicle_force_components);
    }

    const auto &projection = last_overlap_projection_diagnostics;
    if (projection.pair_count > 0) ++runtime.projection_active_steps;
    if (GP::enable_overlap_projection) {
        if (projection.converged) {
            ++runtime.projection_converged_steps;
        } else {
            ++runtime.projection_nonconverged_steps;
        }
        if (projection.hit_iteration_limit) {
            ++runtime.projection_iteration_limit_steps;
        }
    }
    runtime.projection_pair_events += projection.pair_count;
    runtime.projection_corrected_particle_events +=
            projection.corrected_particle_count;
    runtime.projection_iteration_count += projection.iterations_completed;
    runtime.max_projection_iterations_used = std::max(
            runtime.max_projection_iterations_used,
            projection.iterations_completed);
    runtime.max_projection_vesicle_neighbor_cell_reach = std::max(
            runtime.max_projection_vesicle_neighbor_cell_reach,
            projection.vesicle_neighbor_cell_reach);
    runtime.max_projection_excess_overlap_nm = std::max(
            runtime.max_projection_excess_overlap_nm,
            projection.max_excess_overlap_nm);
    runtime.projection_residual_pair_events += projection.residual_pair_count;
    runtime.max_projection_residual_excess_nm = std::max(
            runtime.max_projection_residual_excess_nm,
            projection.residual_max_excess_overlap_nm);
    runtime.max_projection_vesicle_center_cutoff_nm = std::max(
            runtime.max_projection_vesicle_center_cutoff_nm,
            projection.vesicle_center_cutoff_nm);
}

void Interaction::advanceSkRockMacroStep() {
    if (GP::shape != 0) {
        throw std::invalid_argument(
                "The experimental SK-ROCK integrator currently supports only the cubic reflecting domain");
    }
    if (GP::sk_rock_stages < 2) {
        throw std::invalid_argument(
                "SK-ROCK requires at least two stages");
    }

    const int stages = GP::sk_rock_stages;
    const double macro_step = static_cast<double>(GP::deltaT);
    const double omega_zero = 1.0
                              + GP::sk_rock_damping
                                / static_cast<double>(stages * stages);

    std::vector<double> chebyshev_t(
            static_cast<std::size_t>(stages + 1), 0.0);
    std::vector<double> chebyshev_u(
            static_cast<std::size_t>(stages), 0.0);
    chebyshev_t[0] = 1.0;
    chebyshev_t[1] = omega_zero;
    for (int i = 2; i <= stages; ++i) {
        chebyshev_t[static_cast<std::size_t>(i)] =
                2.0 * omega_zero
                * chebyshev_t[static_cast<std::size_t>(i - 1)]
                - chebyshev_t[static_cast<std::size_t>(i - 2)];
    }
    chebyshev_u[0] = 1.0;
    if (stages > 1) {
        chebyshev_u[1] = 2.0 * omega_zero;
        for (int i = 2; i < stages; ++i) {
            chebyshev_u[static_cast<std::size_t>(i)] =
                    2.0 * omega_zero
                    * chebyshev_u[static_cast<std::size_t>(i - 1)]
                    - chebyshev_u[static_cast<std::size_t>(i - 2)];
        }
    }
    const double derivative = static_cast<double>(stages)
                              * chebyshev_u[static_cast<std::size_t>(stages - 1)];
    const double omega_one =
            chebyshev_t[static_cast<std::size_t>(stages)] / derivative;
    if (!std::isfinite(omega_one) || omega_one <= 0.0) {
        throw std::runtime_error("Invalid SK-ROCK Chebyshev coefficients");
    }

    const auto original_synapsin_locations = Synapsin::locations;
    const auto original_synapsin_positions = Synapsin::positions;
    const auto original_vesicle_locations = Vesicles::locations;
    const auto original_vesicle_positions = Vesicles::positions;
    const auto original_generator = GP::gen;
    const auto original_runtime = runtime_diagnostics;
    const bool original_boundary_diagnostics =
            GF::boundary_diagnostics_enabled;

    try {

        GF::boundary_diagnostics_enabled = false;

        const ParticleState x_zero_syn =
                captureParticleState(Agent::Synapsin_agent);
        const ParticleState x_zero_ves =
                captureParticleState(Agent::Vesicle_agent);

        std::normal_distribution<float> normal_distribution(0.0F, 1.0F);

        const ParticleState q_ves = brownianIncrement(
                Agent::Vesicle_agent, macro_step, normal_distribution);
        const ParticleState q_syn = brownianIncrement(
                Agent::Synapsin_agent, macro_step, normal_distribution);

        if (GP::f_off) {
            ParticleState exact_ves = x_zero_ves;
            ParticleState exact_syn = x_zero_syn;
            for (std::size_t i = 0; i < exact_ves.size(); ++i) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    exact_ves[i][axis] += q_ves[i][axis];
                }
            }
            for (std::size_t i = 0; i < exact_syn.size(); ++i) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    exact_syn[i][axis] += q_syn[i][axis];
                }
            }
            applyParticleState(
                    exact_ves, Agent::Vesicle_agent, runtime_diagnostics);
            applyParticleState(
                    exact_syn, Agent::Synapsin_agent, runtime_diagnostics);
            rebuildParticleMappings();
            ++runtime_diagnostics.sk_rock_macro_steps;
            Synapsin::drift_scale.assign(Synapsin::locations.size(), 1.0);
            Vesicles::drift_scale.assign(Vesicles::locations.size(), 1.0);
            GF::boundary_diagnostics_enabled = original_boundary_diagnostics;
            return;
        }

        const double mu_one = omega_one / omega_zero;
        const double nu_one = static_cast<double>(stages) * omega_one / 2.0;
        const double kappa_one =
                static_cast<double>(stages) * omega_one / omega_zero;

        ParticleState shifted_syn = x_zero_syn;
        ParticleState shifted_ves = x_zero_ves;
        for (std::size_t i = 0; i < shifted_syn.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                shifted_syn[i][axis] += nu_one * q_syn[i][axis];
            }
        }
        for (std::size_t i = 0; i < shifted_ves.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                shifted_ves[i][axis] += nu_one * q_ves[i][axis];
            }
        }
        applyParticleState(
                shifted_ves, Agent::Vesicle_agent, runtime_diagnostics);
        applyParticleState(
                shifted_syn, Agent::Synapsin_agent, runtime_diagnostics);
        rebuildParticleMappings();
        agentInteraction(false);
        ++runtime_diagnostics.sk_rock_force_evaluations;

        const ParticleState drift_one_syn =
                currentDrift(Agent::Synapsin_agent);
        const ParticleState drift_one_ves =
                currentDrift(Agent::Vesicle_agent);
        ParticleState k_newer_syn(x_zero_syn.size());
        ParticleState k_newer_ves(x_zero_ves.size());
        for (std::size_t i = 0; i < k_newer_syn.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                k_newer_syn[i][axis] = x_zero_syn[i][axis]
                        + mu_one * macro_step * drift_one_syn[i][axis]
                        + kappa_one * q_syn[i][axis];
            }
        }
        for (std::size_t i = 0; i < k_newer_ves.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                k_newer_ves[i][axis] = x_zero_ves[i][axis]
                        + mu_one * macro_step * drift_one_ves[i][axis]
                        + kappa_one * q_ves[i][axis];
            }
        }
        applyParticleState(
                k_newer_ves, Agent::Vesicle_agent, runtime_diagnostics);
        applyParticleState(
                k_newer_syn, Agent::Synapsin_agent, runtime_diagnostics);
        rebuildParticleMappings();

        ParticleState k_older_syn = x_zero_syn;
        ParticleState k_older_ves = x_zero_ves;
        for (int stage = 2; stage <= stages; ++stage) {
            agentInteraction(false);
            ++runtime_diagnostics.sk_rock_force_evaluations;
            const ParticleState drift_syn =
                    currentDrift(Agent::Synapsin_agent);
            const ParticleState drift_ves =
                    currentDrift(Agent::Vesicle_agent);

            const double denominator =
                    chebyshev_t[static_cast<std::size_t>(stage)];
            const double mu = 2.0 * omega_one
                              * chebyshev_t[static_cast<std::size_t>(stage - 1)]
                              / denominator;
            const double nu = 2.0 * omega_zero
                              * chebyshev_t[static_cast<std::size_t>(stage - 1)]
                              / denominator;
            const double kappa = -chebyshev_t[
                    static_cast<std::size_t>(stage - 2)] / denominator;

            ParticleState k_next_syn = affineSkRockStage(
                    k_older_syn, k_newer_syn, drift_syn,
                    kappa, nu, mu * macro_step);
            ParticleState k_next_ves = affineSkRockStage(
                    k_older_ves, k_newer_ves, drift_ves,
                    kappa, nu, mu * macro_step);
            applyParticleState(
                    k_next_ves, Agent::Vesicle_agent, runtime_diagnostics);
            applyParticleState(
                    k_next_syn, Agent::Synapsin_agent, runtime_diagnostics);
            rebuildParticleMappings();

            k_older_syn = std::move(k_newer_syn);
            k_older_ves = std::move(k_newer_ves);
            k_newer_syn = std::move(k_next_syn);
            k_newer_ves = std::move(k_next_ves);
        }

        ++runtime_diagnostics.sk_rock_macro_steps;
        Synapsin::drift_scale.assign(Synapsin::locations.size(), 1.0);
        Vesicles::drift_scale.assign(Vesicles::locations.size(), 1.0);
        GF::boundary_diagnostics_enabled = original_boundary_diagnostics;
    } catch (...) {
        Synapsin::locations = original_synapsin_locations;
        Synapsin::positions = original_synapsin_positions;
        Vesicles::locations = original_vesicle_locations;
        Vesicles::positions = original_vesicle_positions;
        GP::gen = original_generator;
        runtime_diagnostics = original_runtime;
        GF::boundary_diagnostics_enabled = original_boundary_diagnostics;
        rebuildParticleMappings();
        throw;
    }
}

void Interaction::advanceSrock2MacroStep() {
    if (GP::shape != 0) {
        throw std::invalid_argument(
                "The experimental S-ROCK2 integrator currently supports only the cubic reflecting domain");
    }

    const auto table = srock2_coefficients::tableForDegree(
            GP::srock2_degree);
    const std::size_t expected_coefficients =
            static_cast<std::size_t>(2 * table.degree - 1);
    if (table.recurrence_size != expected_coefficients) {
        throw std::runtime_error("Invalid S-ROCK2 recurrence table");
    }

    const double macro_step = static_cast<double>(GP::deltaT);
    const double alpha = table.alpha;
    const double sigma_alpha =
            0.5 * (1.0 - alpha) + alpha * table.sigma;
    const double tau_alpha =
            0.5 * (1.0 - alpha) * (1.0 - alpha)
            + 2.0 * alpha * (1.0 - alpha) * table.sigma
            + alpha * alpha * table.sigma
              * (table.sigma + table.tau);
    if (!std::isfinite(sigma_alpha) || !std::isfinite(tau_alpha)) {
        throw std::runtime_error("Invalid S-ROCK2 finishing coefficients");
    }

    const auto original_synapsin_locations = Synapsin::locations;
    const auto original_synapsin_positions = Synapsin::positions;
    const auto original_vesicle_locations = Vesicles::locations;
    const auto original_vesicle_positions = Vesicles::positions;
    const auto original_generator = GP::gen;
    const auto original_runtime = runtime_diagnostics;
    const bool original_boundary_diagnostics =
            GF::boundary_diagnostics_enabled;

    try {
        GF::boundary_diagnostics_enabled = false;

        const ParticleState x_zero_syn =
                captureParticleState(Agent::Synapsin_agent);
        const ParticleState x_zero_ves =
                captureParticleState(Agent::Vesicle_agent);

        std::normal_distribution<float> normal_distribution(0.0F, 1.0F);

        const ParticleState q_ves = brownianIncrement(
                Agent::Vesicle_agent, macro_step, normal_distribution, true);
        const ParticleState q_syn = brownianIncrement(
                Agent::Synapsin_agent, macro_step, normal_distribution, true);

        if (GP::f_off) {
            ParticleState exact_ves = x_zero_ves;
            ParticleState exact_syn = x_zero_syn;
            for (std::size_t i = 0; i < exact_ves.size(); ++i) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    exact_ves[i][axis] += q_ves[i][axis];
                }
            }
            for (std::size_t i = 0; i < exact_syn.size(); ++i) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    exact_syn[i][axis] += q_syn[i][axis];
                }
            }
            applyParticleState(
                    exact_ves, Agent::Vesicle_agent,
                    runtime_diagnostics, true);
            applyParticleState(
                    exact_syn, Agent::Synapsin_agent,
                    runtime_diagnostics, true);
            rebuildParticleMappings();
            ++runtime_diagnostics.srock2_macro_steps;
            Synapsin::drift_scale.assign(Synapsin::locations.size(), 1.0);
            Vesicles::drift_scale.assign(Vesicles::locations.size(), 1.0);
            GF::boundary_diagnostics_enabled = original_boundary_diagnostics;
            return;
        }

        agentInteraction(false);
        ++runtime_diagnostics.srock2_force_evaluations;
        const ParticleState first_drift_syn =
                currentDrift(Agent::Synapsin_agent);
        const ParticleState first_drift_ves =
                currentDrift(Agent::Vesicle_agent);

        const double mu_one = table.recurrence[0];
        ParticleState newer_syn = affineSkRockStage(
                x_zero_syn, x_zero_syn, first_drift_syn,
                0.0, 1.0, alpha * macro_step * mu_one);
        ParticleState newer_ves = affineSkRockStage(
                x_zero_ves, x_zero_ves, first_drift_ves,
                0.0, 1.0, alpha * macro_step * mu_one);
        applyParticleState(
                newer_ves, Agent::Vesicle_agent,
                runtime_diagnostics, true);
        applyParticleState(
                newer_syn, Agent::Synapsin_agent,
                runtime_diagnostics, true);
        rebuildParticleMappings();

        ParticleState older_syn = x_zero_syn;
        ParticleState older_ves = x_zero_ves;
        for (int stage = 2; stage <= table.degree; ++stage) {
            agentInteraction(false);
            ++runtime_diagnostics.srock2_force_evaluations;
            const ParticleState drift_syn =
                    currentDrift(Agent::Synapsin_agent);
            const ParticleState drift_ves =
                    currentDrift(Agent::Vesicle_agent);

            const std::size_t coefficient_index =
                    static_cast<std::size_t>(1 + 2 * (stage - 2));
            const double mu = table.recurrence[coefficient_index];
            const double kappa = table.recurrence[coefficient_index + 1];
            const double nu = 1.0 + kappa;
            ParticleState next_syn = affineSkRockStage(
                    older_syn, newer_syn, drift_syn,
                    -kappa, nu, alpha * macro_step * mu);
            ParticleState next_ves = affineSkRockStage(
                    older_ves, newer_ves, drift_ves,
                    -kappa, nu, alpha * macro_step * mu);
            applyParticleState(
                    next_ves, Agent::Vesicle_agent,
                    runtime_diagnostics, true);
            applyParticleState(
                    next_syn, Agent::Synapsin_agent,
                    runtime_diagnostics, true);
            rebuildParticleMappings();

            older_syn = std::move(newer_syn);
            older_ves = std::move(newer_ves);
            newer_syn = std::move(next_syn);
            newer_ves = std::move(next_ves);
        }

        agentInteraction(false);
        ++runtime_diagnostics.srock2_force_evaluations;
        const ParticleState base_drift_syn =
                currentDrift(Agent::Synapsin_agent);
        const ParticleState base_drift_ves =
                currentDrift(Agent::Vesicle_agent);

        ParticleState noisy_star_syn = newer_syn;
        ParticleState noisy_star_ves = newer_ves;
        ParticleState result_syn = newer_syn;
        ParticleState result_ves = newer_ves;
        for (std::size_t i = 0; i < newer_syn.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                noisy_star_syn[i][axis] +=
                        2.0 * macro_step * tau_alpha
                        * base_drift_syn[i][axis] + q_syn[i][axis];
                result_syn[i][axis] +=
                        (2.0 * sigma_alpha - 0.5) * macro_step
                        * base_drift_syn[i][axis] + q_syn[i][axis];
            }
        }
        for (std::size_t i = 0; i < newer_ves.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                noisy_star_ves[i][axis] +=
                        2.0 * macro_step * tau_alpha
                        * base_drift_ves[i][axis] + q_ves[i][axis];
                result_ves[i][axis] +=
                        (2.0 * sigma_alpha - 0.5) * macro_step
                        * base_drift_ves[i][axis] + q_ves[i][axis];
            }
        }

        applyParticleState(
                noisy_star_ves, Agent::Vesicle_agent,
                runtime_diagnostics, true);
        applyParticleState(
                noisy_star_syn, Agent::Synapsin_agent,
                runtime_diagnostics, true);
        rebuildParticleMappings();
        agentInteraction(false);
        ++runtime_diagnostics.srock2_force_evaluations;
        const ParticleState finishing_drift_syn =
                currentDrift(Agent::Synapsin_agent);
        const ParticleState finishing_drift_ves =
                currentDrift(Agent::Vesicle_agent);
        for (std::size_t i = 0; i < result_syn.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result_syn[i][axis] +=
                        0.5 * macro_step * finishing_drift_syn[i][axis];
            }
        }
        for (std::size_t i = 0; i < result_ves.size(); ++i) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                result_ves[i][axis] +=
                        0.5 * macro_step * finishing_drift_ves[i][axis];
            }
        }
        applyParticleState(
                result_ves, Agent::Vesicle_agent,
                runtime_diagnostics, true);
        applyParticleState(
                result_syn, Agent::Synapsin_agent,
                runtime_diagnostics, true);
        rebuildParticleMappings();

        ++runtime_diagnostics.srock2_macro_steps;
        Synapsin::drift_scale.assign(Synapsin::locations.size(), 1.0);
        Vesicles::drift_scale.assign(Vesicles::locations.size(), 1.0);
        GF::boundary_diagnostics_enabled = original_boundary_diagnostics;
    } catch (...) {
        Synapsin::locations = original_synapsin_locations;
        Synapsin::positions = original_synapsin_positions;
        Vesicles::locations = original_vesicle_locations;
        Vesicles::positions = original_vesicle_positions;
        GP::gen = original_generator;
        runtime_diagnostics = original_runtime;
        GF::boundary_diagnostics_enabled = original_boundary_diagnostics;
        rebuildParticleMappings();
        throw;
    }
}

void Interaction::minimizerAgents() {
    const int in = static_cast<int>(GP::index_inside.size());
    const auto &inside = GP::index_inside;
    const auto &nei = GP::neighborhoods;
    const auto &grid = GP::grid;
    const auto &pro_where = Synapsin::which_synapsin_where;
    const auto &ves_where = Vesicles::which_vesicle_where;

    std::vector<std::array<double, 3>> protein_forces(
            Synapsin::locations.size(), {0.0, 0.0, 0.0});
    std::vector<std::array<double, 3>> vesicle_forces(
            Vesicles::locations.size(), {0.0, 0.0, 0.0});

    for (int i = 0; i < in; ++i) {
        const int xi = inside[i][0];
        const int yi = inside[i][1];
        const int zi = inside[i][2];
        const int home = grid[xi][yi][zi];

        const int home_pro = static_cast<int>(pro_where[home].size());
        const int home_ves = static_cast<int>(ves_where[home].size());

        if (home_pro > 0 || home_ves > 0) {
            std::vector<int> agents;
            std::vector<int> vess;

            for (const auto &loc_xyz: nei[home]) {
                const int loc = grid[loc_xyz[0]][loc_xyz[1]][loc_xyz[2]];
                agents.insert(agents.end(), pro_where[loc].begin(), pro_where[loc].end());
                vess.insert(vess.end(), ves_where[loc].begin(), ves_where[loc].end());
            }

            for (int k = 0; k < home_pro; ++k) {
                const int id_protein = pro_where[home][k];
                auto &F = protein_forces[id_protein];

                for (const auto &p: agents) {
                    if (id_protein != p) {
                        const auto repulsion = repulsiveForce(
                                id_protein, p,
                                Agent::Synapsin_agent, Agent::Synapsin_agent);
                        F[0] += repulsion[0];
                        F[1] += repulsion[1];
                        F[2] += repulsion[2];
                    }
                }
                for (const auto &v: vess) {
                    const auto repulsion = repulsiveForce(
                            id_protein, v,
                            Agent::Synapsin_agent, Agent::Vesicle_agent);
                    F[0] += repulsion[0];
                    F[1] += repulsion[1];
                    F[2] += repulsion[2];
                }
            }

            for (int k = 0; k < home_ves; ++k) {
                const int id_vesicle = ves_where[home][k];
                auto &F = vesicle_forces[id_vesicle];

                for (const auto &v: vess) {
                    if (id_vesicle != v) {
                        const auto repulsion = repulsiveForce(
                                id_vesicle, v,
                                Agent::Vesicle_agent, Agent::Vesicle_agent);
                        F[0] += repulsion[0];
                        F[1] += repulsion[1];
                        F[2] += repulsion[2];
                    }
                }
                for (const auto &p: agents) {
                    const auto repulsion = repulsiveForce(
                            id_vesicle, p,
                            Agent::Vesicle_agent, Agent::Synapsin_agent);
                    F[0] += repulsion[0];
                    F[1] += repulsion[1];
                    F[2] += repulsion[2];
                }
            }
        }
    }

    const auto forceToDisplacement = [](
            const std::array<double, 3> &force,
            int id,
            Agent object_type) {
        const double drag = std::max(GF::getDrag(object_type, id), 1e-12);
        const double mobility_step = GP::deltaT / drag;

        std::array<double, 3> displacement = {
                force[0] * mobility_step,
                force[1] * mobility_step,
                force[2] * mobility_step
        };

        const double displacement_norm = MyUtility::euclideanSize(displacement);
        const double radius = GF::getRadius(object_type, id);

        const double contact_step = std::max(
                2.0 * GP::core_allowed_overlap_nm, 1e-6);
        const double max_displacement = GP::adjusted_max
                                        * std::min(radius / 4.0, contact_step);

        if (displacement_norm > max_displacement && displacement_norm > 0.0) {
            const double scale = max_displacement / displacement_norm;
            displacement[0] *= scale;
            displacement[1] *= scale;
            displacement[2] *= scale;
        }

        return displacement;
    };

    for (int id = 0; id < static_cast<int>(protein_forces.size()); ++id) {
        const auto displacement = forceToDisplacement(
                protein_forces[id], id, Agent::Synapsin_agent);
        GF::updateParticlePosition(id, displacement, Agent::Synapsin_agent);
    }

    for (int id = 0; id < static_cast<int>(vesicle_forces.size()); ++id) {
        const auto displacement = forceToDisplacement(
                vesicle_forces[id], id, Agent::Vesicle_agent);
        GF::updateParticlePosition(id, displacement, Agent::Vesicle_agent);
    }

    Vesicles::whichVesiclesWhere();
    Synapsin::whichSynapsinWhere();
}

Interaction::OverlapDiagnostics Interaction::measureOverlaps() {
    OverlapDiagnostics diagnostics;

    std::vector<unsigned char> overlapping_synapsins(
            Synapsin::locations.size(), 0);
    std::vector<unsigned char> overlapping_vesicles(
            Vesicles::locations.size(), 0);

    enum class PairCategory {
        SynapsinSynapsin,
        SynapsinVesicle,
        VesicleVesicle
    };

    const auto recordPair = [&diagnostics,
                             &overlapping_synapsins,
                             &overlapping_vesicles](
            int i,
            int j,
            Agent object_type_i,
            Agent object_type_j,
            PairCategory category) {
        const auto location_i = GF::getLocation(object_type_i, i);
        const auto location_j = GF::getLocation(object_type_j, j);

        const double dx = static_cast<double>(location_i[0])
                          - static_cast<double>(location_j[0]);
        const double dy = static_cast<double>(location_i[1])
                          - static_cast<double>(location_j[1]);
        const double dz = static_cast<double>(location_i[2])
                          - static_cast<double>(location_j[2]);
        const double distance = MyUtility::euclideanSize(dx, dy, dz);
        const double radius_sum =
                static_cast<double>(GF::getRadius(object_type_i, i))
                + static_cast<double>(GF::getRadius(object_type_j, j));
        const double outer_overlap = radius_sum - distance;
        const double excess_core_overlap = outer_overlap
                - allowedOuterOverlapNm(object_type_i, object_type_j);

        if (excess_core_overlap <= 0.0) {
            return;
        }

        ++diagnostics.overlapping_pair_count;
        diagnostics.overlap_sum_nm += excess_core_overlap;
        diagnostics.overlap_max_nm = std::max(
                diagnostics.overlap_max_nm, excess_core_overlap);

        switch (category) {
            case PairCategory::SynapsinSynapsin:
                ++diagnostics.synapsin_synapsin_pair_count;
                overlapping_synapsins[i] = 1;
                overlapping_synapsins[j] = 1;
                break;
            case PairCategory::SynapsinVesicle:
                ++diagnostics.synapsin_vesicle_pair_count;
                if (object_type_i == Agent::Synapsin_agent) {
                    overlapping_synapsins[i] = 1;
                    overlapping_vesicles[j] = 1;
                } else {
                    overlapping_vesicles[i] = 1;
                    overlapping_synapsins[j] = 1;
                }
                break;
            case PairCategory::VesicleVesicle:
                ++diagnostics.vesicle_vesicle_pair_count;
                overlapping_vesicles[i] = 1;
                overlapping_vesicles[j] = 1;
                break;
        }
    };

    for (int i = 0; i < static_cast<int>(Synapsin::locations.size()); ++i) {
        const auto &neighborhood = GF::neiborhoodList(
                i, Agent::Synapsin_agent);

        for (const auto &xyz: neighborhood) {
            const int cell = GP::grid[xyz[0]][xyz[1]][xyz[2]];
            for (const int j: Synapsin::which_synapsin_where[cell]) {
                if (j > i) {
                    recordPair(
                            i, j,
                            Agent::Synapsin_agent,
                            Agent::Synapsin_agent,
                            PairCategory::SynapsinSynapsin);
                }
            }
        }
    }

    for (int i = 0; i < static_cast<int>(Vesicles::locations.size()); ++i) {
        const auto &neighborhood = GF::neiborhoodList(
                i, Agent::Vesicle_agent);

        for (const auto &xyz: neighborhood) {
            const int cell = GP::grid[xyz[0]][xyz[1]][xyz[2]];

            for (const int j: Vesicles::which_vesicle_where[cell]) {
                if (j > i) {
                    recordPair(
                            i, j,
                            Agent::Vesicle_agent,
                            Agent::Vesicle_agent,
                            PairCategory::VesicleVesicle);
                }
            }

            for (const int j: Synapsin::which_synapsin_where[cell]) {
                recordPair(
                        i, j,
                        Agent::Vesicle_agent,
                        Agent::Synapsin_agent,
                        PairCategory::SynapsinVesicle);
            }
        }
    }

    diagnostics.overlapping_synapsin_count =
            static_cast<std::int64_t>(std::count(
                    overlapping_synapsins.begin(),
                    overlapping_synapsins.end(),
                    static_cast<unsigned char>(1)));
    diagnostics.overlapping_vesicle_count =
            static_cast<std::int64_t>(std::count(
                    overlapping_vesicles.begin(),
                    overlapping_vesicles.end(),
                    static_cast<unsigned char>(1)));
    diagnostics.overlapping_particle_count =
            diagnostics.overlapping_synapsin_count
            + diagnostics.overlapping_vesicle_count;

    if (diagnostics.overlapping_pair_count > 0) {
        diagnostics.overlap_mean_nm =
                diagnostics.overlap_sum_nm
                / static_cast<double>(diagnostics.overlapping_pair_count);
    }

    return diagnostics;
}

double Interaction::allowedOuterOverlapNm(
        Agent objectType_i, Agent objectType_j) {
    double allowed = std::max(GP::core_allowed_overlap_nm, 0.0);
    if (objectType_i == Agent::Vesicle_agent) {
        allowed += std::max(GP::vesicle_corona_thickness_nm, 0.0);
    }
    if (objectType_j == Agent::Vesicle_agent) {
        allowed += std::max(GP::vesicle_corona_thickness_nm, 0.0);
    }
    return allowed;
}

Interaction::OverlapProjectionDiagnostics
Interaction::resolveDynamicOverlaps() {
    OverlapProjectionDiagnostics diagnostics;
    constexpr double numerical_tolerance_nm = 1e-6;
    const double residual_tolerance_nm = std::max(
            GP::overlap_projection_residual_tolerance_nm,
            numerical_tolerance_nm);
    diagnostics.residual_tolerance_nm = residual_tolerance_nm;

    const double maximum_vesicle_radius_nm = Vesicles::radii.empty()
            ? 0.0
            : static_cast<double>(*std::max_element(
                    Vesicles::radii.begin(), Vesicles::radii.end()));
    const double vesicle_center_cutoff_nm = std::max(
            0.0,
            2.0 * maximum_vesicle_radius_nm
            - allowedOuterOverlapNm(
                    Agent::Vesicle_agent, Agent::Vesicle_agent));
    const int vesicle_neighbor_cell_reach =
            ProjectionNeighborhood::requiredCellReach(
                    vesicle_center_cutoff_nm, GP::steps);
    diagnostics.vesicle_center_cutoff_nm = vesicle_center_cutoff_nm;
    diagnostics.vesicle_neighbor_cell_reach =
            vesicle_neighbor_cell_reach;
    if (GP::vesicle_corona_thickness_nm > 0.0
        && vesicle_neighbor_cell_reach > 2) {
        throw std::runtime_error(
                "Configured cells do not cover vesicle core-overlap projection within neighbor depth two");
    }

    std::vector<unsigned char> corrected_synapsins(
            Synapsin::locations.size(), 0);
    std::vector<unsigned char> corrected_vesicles(
            Vesicles::locations.size(), 0);

    const auto visitPairs = [vesicle_neighbor_cell_reach](
            const auto &visitor) {
        const std::array<int, 3> grid_size = {
                GP::x_size, GP::y_size, GP::z_size
        };
#pragma omp parallel
        {
#pragma omp for schedule(static)
        for (int i = 0;
             i < static_cast<int>(Synapsin::locations.size()); ++i) {
            const auto &neighborhood = GF::neiborhoodList(
                    i, Agent::Synapsin_agent);
            for (const auto &xyz: neighborhood) {
                const int cell = GP::grid[xyz[0]][xyz[1]][xyz[2]];
                for (const int j: Synapsin::which_synapsin_where[cell]) {
                    if (j > i) {
                        visitor(i, j,
                                Agent::Synapsin_agent,
                                Agent::Synapsin_agent);
                    }
                }
            }
        }

#pragma omp for schedule(static)
        for (int i = 0;
             i < static_cast<int>(Vesicles::locations.size()); ++i) {
            const auto &neighborhood = GF::neiborhoodList(
                    i, Agent::Vesicle_agent);
            for (const auto &xyz: neighborhood) {
                const int cell = GP::grid[xyz[0]][xyz[1]][xyz[2]];
                for (const int j: Synapsin::which_synapsin_where[cell]) {
                    visitor(i, j,
                            Agent::Vesicle_agent,
                            Agent::Synapsin_agent);
                }
            }
        }

#pragma omp for schedule(static)
        for (int i = 0;
             i < static_cast<int>(Vesicles::locations.size()); ++i) {
            const auto &home = Vesicles::positions[i];
            ProjectionNeighborhood::visitCellCube(
                    home, vesicle_neighbor_cell_reach, grid_size,
                    [&visitor, i](const std::array<int, 3> &xyz) {
                        const int cell = GP::grid[xyz[0]][xyz[1]][xyz[2]];
                        if (cell < 0) return;
                        for (const int j: Vesicles::which_vesicle_where[cell]) {
                            if (j > i) {
                                visitor(i, j,
                                        Agent::Vesicle_agent,
                                        Agent::Vesicle_agent);
                            }
                        }
                    });
        }
        }
    };

    if (GP::enable_overlap_projection) {
        std::vector<std::array<double, 3>> synapsin_corrections(
                Synapsin::locations.size(), {0.0, 0.0, 0.0});
        std::vector<std::array<double, 3>> vesicle_corrections(
                Vesicles::locations.size(), {0.0, 0.0, 0.0});
        std::vector<int> synapsin_contact_count(
                Synapsin::locations.size(), 0);
        std::vector<int> vesicle_contact_count(
                Vesicles::locations.size(), 0);

        for (int iteration = 0;
             iteration < GP::overlap_projection_max_iterations;
             ++iteration) {
            std::fill(
                    synapsin_corrections.begin(),
                    synapsin_corrections.end(),
                    std::array<double, 3>{0.0, 0.0, 0.0});
            std::fill(
                    vesicle_corrections.begin(),
                    vesicle_corrections.end(),
                    std::array<double, 3>{0.0, 0.0, 0.0});
            std::fill(
                    synapsin_contact_count.begin(),
                    synapsin_contact_count.end(), 0);
            std::fill(
                    vesicle_contact_count.begin(),
                    vesicle_contact_count.end(), 0);
            std::int64_t iteration_pair_count = 0;

            const auto accumulateCorrection = [&] (
                    int i, int j,
                    Agent object_type_i, Agent object_type_j) {
                const auto location_i = GF::getLocation(object_type_i, i);
                const auto location_j = GF::getLocation(object_type_j, j);
                double dx = static_cast<double>(location_i[0])
                            - static_cast<double>(location_j[0]);
                double dy = static_cast<double>(location_i[1])
                            - static_cast<double>(location_j[1]);
                double dz = static_cast<double>(location_i[2])
                            - static_cast<double>(location_j[2]);
                const double distance = MyUtility::euclideanSize(dx, dy, dz);
                const double radius_sum =
                        static_cast<double>(GF::getRadius(object_type_i, i))
                        + static_cast<double>(GF::getRadius(object_type_j, j));
                const double excess_overlap =
                        radius_sum - distance
                        - allowedOuterOverlapNm(
                                object_type_i, object_type_j);
                if (excess_overlap <= residual_tolerance_nm) {
                    return;
                }

                if (distance > 1e-9) {
                    dx /= distance;
                    dy /= distance;
                    dz /= distance;
                } else {
                    const auto direction = deterministicSeparationDirection(
                            i, j, object_type_i, object_type_j);
                    dx = direction[0];
                    dy = direction[1];
                    dz = direction[2];
                }

                const double mobility_i = 1.0 / std::max(
                        GF::getDrag(object_type_i, i), 1e-12);
                const double mobility_j = 1.0 / std::max(
                        GF::getDrag(object_type_j, j), 1e-12);
                const double mobility_sum = mobility_i + mobility_j;
                const double correction_i =
                        excess_overlap * mobility_i / mobility_sum;
                const double correction_j =
                        excess_overlap * mobility_j / mobility_sum;

                auto &corrections_i =
                        object_type_i == Agent::Synapsin_agent
                        ? synapsin_corrections : vesicle_corrections;
                auto &corrections_j =
                        object_type_j == Agent::Synapsin_agent
                        ? synapsin_corrections : vesicle_corrections;
                auto &counts_i =
                        object_type_i == Agent::Synapsin_agent
                        ? synapsin_contact_count : vesicle_contact_count;
                auto &counts_j =
                        object_type_j == Agent::Synapsin_agent
                        ? synapsin_contact_count : vesicle_contact_count;

                const double correction_i_x = correction_i * dx;
                const double correction_i_y = correction_i * dy;
                const double correction_i_z = correction_i * dz;
                const double correction_j_x = correction_j * dx;
                const double correction_j_y = correction_j * dy;
                const double correction_j_z = correction_j * dz;
#pragma omp atomic update
                corrections_i[i][0] += correction_i_x;
#pragma omp atomic update
                corrections_i[i][1] += correction_i_y;
#pragma omp atomic update
                corrections_i[i][2] += correction_i_z;
#pragma omp atomic update
                corrections_j[j][0] -= correction_j_x;
#pragma omp atomic update
                corrections_j[j][1] -= correction_j_y;
#pragma omp atomic update
                corrections_j[j][2] -= correction_j_z;
#pragma omp atomic update
                ++counts_i[i];
#pragma omp atomic update
                ++counts_j[j];

#pragma omp atomic update
                ++iteration_pair_count;
                if (iteration == 0) {
#pragma omp atomic update
                    ++diagnostics.pair_count;
#pragma omp critical(projection_max_excess_update)
                    {
                        diagnostics.max_excess_overlap_nm = std::max(
                                diagnostics.max_excess_overlap_nm,
                                excess_overlap);
                    }
                }
            };

            visitPairs(accumulateCorrection);
            if (iteration_pair_count == 0) {
                break;
            }

            for (int i = 0;
                 i < static_cast<int>(synapsin_corrections.size()); ++i) {
                if (synapsin_contact_count[i] == 0) continue;
                const double inv_contacts =
                        1.0 / static_cast<double>(synapsin_contact_count[i]);
                for (double &component: synapsin_corrections[i]) {
                    component *= inv_contacts;
                }
                const auto before = GF::getLocation(
                        Agent::Synapsin_agent, i);
                GF::updateParticlePosition(
                        i, synapsin_corrections[i],
                        Agent::Synapsin_agent);
                const auto after = GF::getLocation(
                        Agent::Synapsin_agent, i);
                if (MyUtility::euclideanSize(
                        static_cast<double>(after[0] - before[0]),
                        static_cast<double>(after[1] - before[1]),
                        static_cast<double>(after[2] - before[2]))
                    > numerical_tolerance_nm) {
                    corrected_synapsins[i] = 1;
                }
            }
            for (int i = 0;
                 i < static_cast<int>(vesicle_corrections.size()); ++i) {
                if (vesicle_contact_count[i] == 0) continue;
                const double inv_contacts =
                        1.0 / static_cast<double>(vesicle_contact_count[i]);
                for (double &component: vesicle_corrections[i]) {
                    component *= inv_contacts;
                }
                const auto before = GF::getLocation(
                        Agent::Vesicle_agent, i);
                GF::updateParticlePosition(
                        i, vesicle_corrections[i],
                        Agent::Vesicle_agent);
                const auto after = GF::getLocation(
                        Agent::Vesicle_agent, i);
                if (MyUtility::euclideanSize(
                        static_cast<double>(after[0] - before[0]),
                        static_cast<double>(after[1] - before[1]),
                        static_cast<double>(after[2] - before[2]))
                    > numerical_tolerance_nm) {
                    corrected_vesicles[i] = 1;
                }
            }

            Synapsin::whichSynapsinWhere();
            Vesicles::whichVesiclesWhere();
            diagnostics.iterations_completed = iteration + 1;
        }
    }

    diagnostics.corrected_particle_count =
            static_cast<std::int64_t>(std::count(
                    corrected_synapsins.begin(),
                    corrected_synapsins.end(),
                    static_cast<unsigned char>(1)))
            + static_cast<std::int64_t>(std::count(
                    corrected_vesicles.begin(),
                    corrected_vesicles.end(),
                    static_cast<unsigned char>(1)));

    const auto recordResidual = [&] (
            int i, int j, Agent object_type_i, Agent object_type_j) {
        const auto location_i = GF::getLocation(object_type_i, i);
        const auto location_j = GF::getLocation(object_type_j, j);
        const double distance = MyUtility::euclideanSize(
                static_cast<double>(location_i[0] - location_j[0]),
                static_cast<double>(location_i[1] - location_j[1]),
                static_cast<double>(location_i[2] - location_j[2]));
        const double excess_overlap =
                static_cast<double>(GF::getRadius(object_type_i, i))
                + static_cast<double>(GF::getRadius(object_type_j, j))
                - distance
                - allowedOuterOverlapNm(object_type_i, object_type_j);
        if (excess_overlap > numerical_tolerance_nm) {
#pragma omp atomic update
            ++diagnostics.residual_pair_count;
#pragma omp critical(projection_residual_max_update)
            {
                diagnostics.residual_max_excess_overlap_nm = std::max(
                        diagnostics.residual_max_excess_overlap_nm,
                        excess_overlap);
            }
        }
    };

    visitPairs(recordResidual);

    diagnostics.converged =
            diagnostics.residual_max_excess_overlap_nm
            <= residual_tolerance_nm;
    diagnostics.hit_iteration_limit =
            GP::enable_overlap_projection
            && !diagnostics.converged
            && GP::overlap_projection_max_iterations > 0
            && diagnostics.iterations_completed
               >= GP::overlap_projection_max_iterations;

    last_overlap_projection_diagnostics = diagnostics;
    return diagnostics;
}

void Interaction::agentInteraction(bool capture_frame_analysis) {

    const bool capture_force_components =
            (capture_frame_analysis && GP::save_forces)
            || GP::enable_drift_component_diagnostics;
    if (capture_force_components) {
        synapsin_force_components.assign(
                Synapsin::locations.size(), ParticleForceComponents{});
        vesicle_force_components.assign(
                Vesicles::locations.size(), ParticleForceComponents{});
    }

    if (GP::number_of_SynapsinI) {

        GF::zeroFill(Synapsin::synapsin_forces);
        calculateForcesSelves(
                Agent::Synapsin_agent, capture_frame_analysis);

    }

    if (GP::number_of_vesicles) {

        GF::zeroFill(Vesicles::vesicles_forces);
        calculateForcesSelves(
                Agent::Vesicle_agent, capture_frame_analysis);

    }

    if (GP::number_of_SynapsinI && GP::number_of_vesicles) {
        calculateForcesBetween(
                Agent::Vesicle_agent, Agent::Synapsin_agent,
                capture_frame_analysis);

    }

}

std::array<double, 3> Interaction::repulsiveForce(int i, int j, Agent objectType_i, Agent objectType_j) {

    float r1 = GF::getRadius(objectType_i, i);
    float r2 = GF::getRadius(objectType_j, j);

    std::array<float, 3> location_i = GF::getLocation(objectType_i, i);
    std::array<float, 3> location_j = GF::getLocation(objectType_j, j);

    float Rx = location_i[0] - location_j[0];
    float Ry = location_i[1] - location_j[1];
    float Rz = location_i[2] - location_j[2];

    const double raw_distance = MyUtility::euclideanSize(Rx, Ry, Rz);

    if (raw_distance > 1e-9) {
        Rx /= raw_distance;
        Ry /= raw_distance;
        Rz /= raw_distance;
    } else {
        const auto direction = deterministicSeparationDirection(
                i, j, objectType_i, objectType_j);
        Rx = static_cast<float>(direction[0]);
        Ry = static_cast<float>(direction[1]);
        Rz = static_cast<float>(direction[2]);
    }

    const double surface_distance = raw_distance
                                    - static_cast<double>(r1 + r2);
    const double excess_core_overlap = -surface_distance
            - allowedOuterOverlapNm(objectType_i, objectType_j);
    if (excess_core_overlap <= 0.0) {
        return {0.0, 0.0, 0.0};
    }

    const double scaled_force = minimizerRepulsionScale(
            i, j, objectType_i, objectType_j);
    return {
            scaled_force * Rx,
            scaled_force * Ry,
            scaled_force * Rz
    };
}

void Interaction::calculateForcesBetween(
        Agent objectType_i, Agent objectType_j,
        bool capture_frame_analysis) {

    const auto &grid = GP::grid;

    const std::vector<std::vector<int>> &which_where_j = GF::getWhichWhere(objectType_j);

    std::vector<std::array<double, 3>> &forces_i = GF::getForces(objectType_i);
    std::vector<std::array<double, 3>> &forces_j = GF::getForces(objectType_j);
    auto &force_components_i = forceComponentStorage(objectType_i);
    auto &force_components_j = forceComponentStorage(objectType_j);

    int number_i = GF::getNumberAgent(objectType_i);
    int number_j = GF::getNumberAgent(objectType_j);

    const bool capture_force_components =
            (capture_frame_analysis && GP::save_forces)
            || GP::enable_drift_component_diagnostics;

    std::vector<std::vector<float>> force_between_i(
            capture_frame_analysis ? number_i : 0);
    std::vector<std::vector<float>> force_between_j(
            capture_frame_analysis ? number_j : 0);
    std::vector<std::vector<float>> distance_between_i(
            capture_frame_analysis ? number_i : 0);
    std::vector<std::vector<float>> distance_between_j(
            capture_frame_analysis ? number_j : 0);
    std::vector<std::vector<float>> overlap_between_i(
            capture_frame_analysis ? number_i : 0);
    std::vector<std::vector<float>> overlap_between_j(
            capture_frame_analysis ? number_j : 0);

    std::vector<std::array<double, 3>> shared_forces_j(number_j, {0.0, 0.0, 0.0});

#pragma omp parallel
    {

        std::vector<std::array<double, 3>> local_forces_i(number_i, {0.0, 0.0, 0.0});
        std::vector<ParticleForceComponents> local_components_i(
                capture_force_components ? number_i : 0);
        std::vector<ParticleForceComponents> local_components_j(
                capture_force_components ? number_j : 0);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < number_i; ++i) {

            const auto &neihood = GF::neiborhoodList(i, objectType_i);

            for (const auto &xyz: neihood) {

                int loc = grid[xyz[0]][xyz[1]][xyz[2]];

                for (int j: which_where_j[loc]) {
                    const auto force_result = calculateTotalForces(
                            i, j, objectType_i, objectType_j);
                    const auto &dF = force_result.total_force;
                    const double distance = force_result.distance;
                    const double overlap = force_result.overlap;

                    if (capture_force_components) {
                        addForceComponents(
                                local_components_i[i],
                                force_result.components,
                                1.0);
                        addForceComponents(
                                local_components_j[j],
                                force_result.components,
                                -1.0);
                    }

                    if (capture_frame_analysis) {
                        const float force_magnitude = MyUtility::euclideanSize(
                                dF[0], dF[1], dF[2]);

                        force_between_i[i].push_back(force_magnitude);
                        distance_between_i[i].push_back(static_cast<float>(distance));
                        overlap_between_i[i].push_back(static_cast<float>(overlap));

#pragma omp critical(j_stats_update)
                        {
                            force_between_j[j].push_back(force_magnitude);
                            distance_between_j[j].push_back(static_cast<float>(distance));
                            overlap_between_j[j].push_back(static_cast<float>(overlap));
                        }
                    }

                    local_forces_i[i][0] += dF[0];
                    local_forces_i[i][1] += dF[1];
                    local_forces_i[i][2] += dF[2];

#pragma omp atomic
                    shared_forces_j[j][0] -= dF[0];
#pragma omp atomic
                    shared_forces_j[j][1] -= dF[1];
#pragma omp atomic
                    shared_forces_j[j][2] -= dF[2];
                }
            }
        }

#pragma omp critical(forces_i_accumulation)
        {
            for (int i = 0; i < number_i; ++i) {
                forces_i[i][0] += local_forces_i[i][0];
                forces_i[i][1] += local_forces_i[i][1];
                forces_i[i][2] += local_forces_i[i][2];

                if (capture_force_components) {
                    addForceComponents(
                            force_components_i[i], local_components_i[i]);
                }
            }

            if (capture_force_components) {
                for (int j = 0; j < number_j; ++j) {
                    addForceComponents(
                            force_components_j[j], local_components_j[j]);
                }
            }
        }
    }

    for (int j = 0; j < number_j; ++j) {
        forces_j[j][0] += shared_forces_j[j][0];
        forces_j[j][1] += shared_forces_j[j][1];
        forces_j[j][2] += shared_forces_j[j][2];
    }
    if (capture_frame_analysis) {

        for (int i = 0; i < number_i; ++i) {
            int actual_neighbor_count = static_cast<int>(force_between_i[i].size());
            Vesicles::neighborhood_cross[i] = actual_neighbor_count;
            if (actual_neighbor_count > 0) {

                auto sum_f = std::accumulate(force_between_i[i].begin(),
                                             force_between_i[i].end(), 0.0);

                auto mean_f = sum_f / actual_neighbor_count;

                double diff_f = 0.0;
                for (const auto &f: force_between_i[i]) {
                    diff_f += std::pow(f - mean_f, 2);
                }

                const double std_f = actual_neighbor_count > 1
                                     ? std::sqrt(diff_f / (actual_neighbor_count - 1))
                                     : 0.0;

                Vesicles::cross_Nforces[i] = GP::StochasticParameters(
                        sum_f, mean_f, std_f,
                        *std::min_element(force_between_i[i].begin(), force_between_i[i].end()),
                        *std::max_element(force_between_i[i].begin(), force_between_i[i].end())
                );

                auto sum_d = std::accumulate(distance_between_i[i].begin(), distance_between_i[i].end(), 0.0);
                auto mean_d = sum_d / actual_neighbor_count;
                double diff_d = 0.0;
                for (const auto &d: distance_between_i[i]) {
                    diff_d += std::pow(d - mean_d, 2);
                }
                const double std_d = actual_neighbor_count > 1
                                     ? std::sqrt(diff_d / (actual_neighbor_count - 1))
                                     : 0.0;

                Vesicles::cross_Ndistance[i] = GP::StochasticParameters(
                        sum_d, mean_d, std_d,
                        *std::min_element(distance_between_i[i].begin(), distance_between_i[i].end()),
                        *std::max_element(distance_between_i[i].begin(), distance_between_i[i].end())
                );

                auto sum_o = std::accumulate(overlap_between_i[i].begin(), overlap_between_i[i].end(), 0.0);
                auto mean_o = sum_o / actual_neighbor_count;
                double diff_o = 0.0;
                for (const auto &o: overlap_between_i[i]) {
                    diff_o += std::pow(o - mean_o, 2);
                }
                const double std_o = actual_neighbor_count > 1
                                     ? std::sqrt(diff_o / (actual_neighbor_count - 1))
                                     : 0.0;

                Vesicles::cross_Noverlaps[i] = GP::StochasticParameters(
                        sum_o, mean_o, std_o,
                        *std::min_element(overlap_between_i[i].begin(), overlap_between_i[i].end()),
                        *std::max_element(overlap_between_i[i].begin(), overlap_between_i[i].end())
                );
            } else {

                Vesicles::cross_Nforces[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                Vesicles::cross_Ndistance[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                Vesicles::cross_Noverlaps[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
            }
        }

        for (int j = 0; j < number_j; ++j) {
            int actual_neighbor_count = static_cast<int>(force_between_j[j].size());
            Synapsin::neighborhood_cross[j] = actual_neighbor_count;
            if (actual_neighbor_count > 0) {

                auto sum_f = std::accumulate(force_between_j[j].begin(), force_between_j[j].end(), 0.0);
                auto mean_f = sum_f / actual_neighbor_count;
                double diff_f = 0.0;
                for (const auto &f: force_between_j[j]) {
                    diff_f += std::pow(f - mean_f, 2);
                }
                const double std_f = actual_neighbor_count > 1
                                     ? std::sqrt(diff_f / (actual_neighbor_count - 1))
                                     : 0.0;

                Synapsin::cross_Nforces[j] = GP::StochasticParameters(
                        sum_f, mean_f, std_f,
                        *std::min_element(force_between_j[j].begin(), force_between_j[j].end()),
                        *std::max_element(force_between_j[j].begin(), force_between_j[j].end())
                );

                auto sum_d = std::accumulate(distance_between_j[j].begin(), distance_between_j[j].end(), 0.0);
                auto mean_d = sum_d / actual_neighbor_count;
                double diff_d = 0.0;
                for (const auto &d: distance_between_j[j]) {
                    diff_d += std::pow(d - mean_d, 2);
                }
                const double std_d = actual_neighbor_count > 1
                                     ? std::sqrt(diff_d / (actual_neighbor_count - 1))
                                     : 0.0;

                Synapsin::cross_Ndistance[j] = GP::StochasticParameters(
                        sum_d, mean_d, std_d,
                        *std::min_element(distance_between_j[j].begin(), distance_between_j[j].end()),
                        *std::max_element(distance_between_j[j].begin(), distance_between_j[j].end())
                );

                auto sum_o = std::accumulate(overlap_between_j[j].begin(), overlap_between_j[j].end(), 0.0);
                auto mean_o = sum_o / actual_neighbor_count;
                double diff_o = 0.0;
                for (const auto &o: overlap_between_j[j]) {
                    diff_o += std::pow(o - mean_o, 2);
                }
                const double std_o = actual_neighbor_count > 1
                                     ? std::sqrt(diff_o / (actual_neighbor_count - 1))
                                     : 0.0;

                Synapsin::cross_Noverlaps[j] = GP::StochasticParameters(
                        sum_o, mean_o, std_o,
                        *std::min_element(overlap_between_j[j].begin(), overlap_between_j[j].end()),
                        *std::max_element(overlap_between_j[j].begin(), overlap_between_j[j].end())
                );
            } else {

                Synapsin::cross_Nforces[j] = {0.0, 0.0, 0.0, 0.0, 0.0};
                Synapsin::cross_Ndistance[j] = {0.0, 0.0, 0.0, 0.0, 0.0};
                Synapsin::cross_Noverlaps[j] = {0.0, 0.0, 0.0, 0.0, 0.0};
            }
        }
    }
}

void Interaction::calculateForcesSelves(
        Agent objectType_i, bool capture_frame_analysis) {
    const auto &grid = GP::grid;
    std::vector<std::array<double, 3>> &forces = GF::getForces(objectType_i);
    auto &force_components = forceComponentStorage(objectType_i);
    int number = GF::getNumberAgent(objectType_i);
    const std::vector<std::vector<int>> &which_where = GF::getWhichWhere(objectType_i);

    const bool capture_force_components =
            (capture_frame_analysis && GP::save_forces)
            || GP::enable_drift_component_diagnostics;
    std::vector<std::vector<float>> force_self(
            capture_frame_analysis ? number : 0);
    std::vector<std::vector<float>> distance_self(
            capture_frame_analysis ? number : 0);
    std::vector<std::vector<float>> overlap_self(
            capture_frame_analysis ? number : 0);

    std::vector<std::array<double, 3>> shared_forces(number, {0.0, 0.0, 0.0});

#pragma omp parallel
    {
        std::vector<std::array<double, 3>> local_forces(number, {0.0, 0.0, 0.0});
        std::vector<ParticleForceComponents> local_components(
                capture_force_components ? number : 0);

        std::vector<std::vector<float>> local_force_self(
                capture_frame_analysis ? number : 0);
        std::vector<std::vector<float>> local_distance_self(
                capture_frame_analysis ? number : 0);
        std::vector<std::vector<float>> local_overlap_self(
                capture_frame_analysis ? number : 0);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < number; ++i) {
            const auto &neihood = GF::neiborhoodList(i, objectType_i);

            for (const auto &xyz: neihood) {
                int loc = grid[xyz[0]][xyz[1]][xyz[2]];
                const auto &agents = which_where[loc];

                for (const auto &j: agents) {
                    if (j > i) {
                        const auto force_result = calculateTotalForces(
                                i, j, objectType_i, objectType_i);
                        const auto &dF = force_result.total_force;
                        const double distance = force_result.distance;
                        const double overlap = force_result.overlap;

                        if (capture_force_components) {
                            addForceComponents(
                                    local_components[i],
                                    force_result.components,
                                    1.0);
                            addForceComponents(
                                    local_components[j],
                                    force_result.components,
                                    -1.0);
                        }

                        if (capture_frame_analysis) {
                            const float force_magnitude = MyUtility::euclideanSize(
                                    dF[0], dF[1], dF[2]);

                            local_force_self[i].push_back(force_magnitude);
                            local_distance_self[i].push_back((float) distance);
                            local_overlap_self[i].push_back((float) overlap);
                            local_force_self[j].push_back(force_magnitude);
                            local_distance_self[j].push_back((float) distance);
                            local_overlap_self[j].push_back((float) overlap);
                        }

                        local_forces[i][0] += dF[0];
                        local_forces[i][1] += dF[1];
                        local_forces[i][2] += dF[2];

#pragma omp atomic
                        shared_forces[j][0] -= dF[0];
#pragma omp atomic
                        shared_forces[j][1] -= dF[1];
#pragma omp atomic
                        shared_forces[j][2] -= dF[2];
                    }
                }
            }
        }

#pragma omp critical(merge_all)
        {
            for (int i = 0; i < number; ++i) {

                forces[i][0] += local_forces[i][0];
                forces[i][1] += local_forces[i][1];
                forces[i][2] += local_forces[i][2];

                if (capture_force_components) {
                    addForceComponents(
                            force_components[i], local_components[i]);
                }

                if (capture_frame_analysis) {

                    force_self[i].insert(force_self[i].end(),
                                         local_force_self[i].begin(), local_force_self[i].end());
                    distance_self[i].insert(distance_self[i].end(),
                                            local_distance_self[i].begin(), local_distance_self[i].end());
                    overlap_self[i].insert(overlap_self[i].end(),
                                           local_overlap_self[i].begin(), local_overlap_self[i].end());
                }
            }
        }
    }

    for (int j = 0; j < number; ++j) {
        forces[j][0] += shared_forces[j][0];
        forces[j][1] += shared_forces[j][1];
        forces[j][2] += shared_forces[j][2];
    }

    if (capture_frame_analysis) {

        for (int i = 0; i < number; ++i) {

            int actual_neighbor_count = static_cast<int>(force_self[i].size());

            if (objectType_i == Agent::Synapsin_agent) {

                Synapsin::neighborhood_self[i] = actual_neighbor_count;

            } else {

                Vesicles::neighborhood_self[i] = actual_neighbor_count;

            }

            if (actual_neighbor_count > 0) {

                auto sum_f = std::accumulate(force_self[i].begin(), force_self[i].end(), 0.0);
                auto mean_f = sum_f / actual_neighbor_count;
                double diff_f = 0.0;
                for (const auto &f: force_self[i]) {
                    diff_f += std::pow(f - mean_f, 2);
                }
                const double std_f = actual_neighbor_count > 1
                                     ? std::sqrt(diff_f / (actual_neighbor_count - 1))
                                     : 0.0;

                GP::StochasticParameters force_params(
                        sum_f, mean_f, std_f,
                        *std::min_element(force_self[i].begin(), force_self[i].end()),
                        *std::max_element(force_self[i].begin(), force_self[i].end())
                );

                auto sum_d = std::accumulate(distance_self[i].begin(), distance_self[i].end(), 0.0);
                auto mean_d = sum_d / actual_neighbor_count;
                double diff_d = 0.0;
                for (const auto &d: distance_self[i]) {
                    diff_d += std::pow(d - mean_d, 2);
                }
                const double std_d = actual_neighbor_count > 1
                                     ? std::sqrt(diff_d / (actual_neighbor_count - 1))
                                     : 0.0;

                GP::StochasticParameters distance_params(
                        sum_d, mean_d, std_d,
                        *std::min_element(distance_self[i].begin(), distance_self[i].end()),
                        *std::max_element(distance_self[i].begin(), distance_self[i].end())
                );

                auto sum_o = std::accumulate(overlap_self[i].begin(), overlap_self[i].end(), 0.0);
                auto mean_o = sum_o / actual_neighbor_count;
                double diff_o = 0.0;
                for (const auto &o: overlap_self[i]) {
                    diff_o += std::pow(o - mean_o, 2);
                }
                const double std_o = actual_neighbor_count > 1
                                     ? std::sqrt(diff_o / (actual_neighbor_count - 1))
                                     : 0.0;

                GP::StochasticParameters overlap_params(
                        sum_o, mean_o, std_o,
                        *std::min_element(overlap_self[i].begin(), overlap_self[i].end()),
                        *std::max_element(overlap_self[i].begin(), overlap_self[i].end())
                );

                if (objectType_i == Agent::Synapsin_agent) {
                    Synapsin::self_Nforces[i] = force_params;
                    Synapsin::self_Ndistance[i] = distance_params;
                    Synapsin::self_Noverlaps[i] = overlap_params;
                } else {
                    Vesicles::self_Nforces[i] = force_params;
                    Vesicles::self_Ndistance[i] = distance_params;
                    Vesicles::self_Noverlaps[i] = overlap_params;
                }

            } else {

                if (objectType_i == Agent::Synapsin_agent) {
                    Synapsin::self_Nforces[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    Synapsin::self_Ndistance[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    Synapsin::self_Noverlaps[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                } else {
                    Vesicles::self_Nforces[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    Vesicles::self_Ndistance[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    Vesicles::self_Noverlaps[i] = {0.0, 0.0, 0.0, 0.0, 0.0};
                }

            }
        }

    }
}

Interaction::PairForceResult
Interaction::calculateTotalForces(int i, int j, Agent objectType_i, Agent objectType_j) {

    double r1 = GF::getRadius(objectType_i, i);
    double r2 = GF::getRadius(objectType_j, j);
    std::array<float, 3> loc_i = GF::getLocation(objectType_i, i);
    std::array<float, 3> loc_j = GF::getLocation(objectType_j, j);

    double Rx = loc_i[0] - loc_j[0];
    double Ry = loc_i[1] - loc_j[1];
    double Rz = loc_i[2] - loc_j[2];
    const double raw_distance = MyUtility::euclideanSize(Rx, Ry, Rz);
    const double distance = std::max(raw_distance, 1e-9);

    const double surface_distance = raw_distance - (r1 + r2);
    const double overlap = std::max(0.0, -surface_distance);
    PairForceResult result;
    result.distance = raw_distance;
    result.overlap = overlap;

    if (raw_distance > 1e-9) {
        const double inv_dist = 1.0 / raw_distance;
        Rx *= inv_dist;
        Ry *= inv_dist;
        Rz *= inv_dist;
    } else {
        const auto direction = deterministicSeparationDirection(
                i, j, objectType_i, objectType_j);
        Rx = direction[0];
        Ry = direction[1];
        Rz = direction[2];
    }

    double core_scalar = 0.0;
    if (surface_distance < 0.0) {
        const double excess_core_overlap = std::max(
                0.0,
                overlap - allowedOuterOverlapNm(
                        objectType_i, objectType_j));
        core_scalar = GP::enable_dynamic_steric_penalty
                ? stericCoreForce(excess_core_overlap)
                : 0.0;
        result.components.steric_core = {
                core_scalar * Rx,
                core_scalar * Ry,
                core_scalar * Rz
        };
        if (!GP::continuous_contact_forces) {
            result.total_force = result.components.steric_core;
            return result;
        }
    }

    const double surface_force_distance =
            GP::continuous_contact_forces
            ? std::max(distance, r1 + r2)
            : distance;

    const double electrostatic_scalar =
            GP::enable_electrostatic
            ? electricForce(
                    i, j, surface_force_distance,
                    objectType_i, objectType_j)
            : 0.0;
    const double van_der_waals_scalar =
            GP::enable_van_der_waals
            ? vanDerWaalsForce(
                    distance,
                    static_cast<float>(
                            objectType_i == Agent::Vesicle_agent
                            ? std::max(
                                    r1 - GP::vesicle_corona_thickness_nm,
                                    1e-9)
                            : r1),
                    static_cast<float>(
                            objectType_j == Agent::Vesicle_agent
                            ? std::max(
                                    r2 - GP::vesicle_corona_thickness_nm,
                                    1e-9)
                            : r2))
            : 0.0;
    const double hydrophobic_scalar =
            GP::enable_hydrophobic
            ? hydrophobicForce(
                    i, j, surface_force_distance,
                    objectType_i, objectType_j)
            : 0.0;

    const double raw_total_scalar = core_scalar + electrostatic_scalar
                                    + van_der_waals_scalar
                                    + hydrophobic_scalar;

    const auto radial_vector = [Rx, Ry, Rz](double scalar) {
        return std::array<double, 3>{
                scalar * Rx,
                scalar * Ry,
                scalar * Rz
        };
    };

    result.components.electrostatic = radial_vector(electrostatic_scalar);
    result.components.van_der_waals = radial_vector(van_der_waals_scalar);
    result.components.hydrophobic = radial_vector(hydrophobic_scalar);
    result.total_force = {
            raw_total_scalar * Rx,
            raw_total_scalar * Ry,
            raw_total_scalar * Rz
    };
    return result;
}

double Interaction::stericCoreForce(double excess_core_overlap_nm) {
    if (excess_core_overlap_nm <= 0.0) return 0.0;

    const double allowed_overlap = std::max(
            GP::core_allowed_overlap_nm, 1e-9);
    const double thermal_energy =
            static_cast<double>(GP::kb) * static_cast<double>(GP::T);
    const double spring_constant =
            2.0 * GP::core_energy_kbt * thermal_energy
            / (allowed_overlap * allowed_overlap);
    return spring_constant * excess_core_overlap_nm;
}

double Interaction::electricForce(int i, int j, double distance, Agent objectType_i, Agent objectType_j) {
    const double r1 = GF::getRadius(objectType_i, i);
    const double r2 = GF::getRadius(objectType_j, j);
    const double h = std::max(distance - (r1 + r2), 0.0);

    const auto surfacePotentialMv = [](Agent type) {
        return type == Agent::Synapsin_agent
               ? GP::synapsin_surface_potential_mV
               : GP::vesicle_surface_potential_mV;
    };
    const double psi_i_mV = surfacePotentialMv(objectType_i);
    const double psi_j_mV = surfacePotentialMv(objectType_j);

    constexpr double boltzmann_j_per_k = 1.380649e-23;
    constexpr double elementary_charge_c = 1.602176634e-19;
    const double thermal_voltage_mV =
            1000.0 * boltzmann_j_per_k * static_cast<double>(GP::T)
            / elementary_charge_c;
    const double reduced_psi_i = psi_i_mV / thermal_voltage_mV;
    const double reduced_psi_j = psi_j_mV / thermal_voltage_mV;

    const double thermal_energy =
            static_cast<double>(GP::kb) * static_cast<double>(GP::T);
    const double bjerrum_length_nm =
            static_cast<double>(GP::eCharge) * GP::eCharge
            / (4.0 * std::numbers::pi_v<double>
               * static_cast<double>(GP::e0)
               * static_cast<double>(GP::eSolvent) * thermal_energy);
    const double prefactor =
            thermal_energy * r1 * r2 / bjerrum_length_nm
            * reduced_psi_i * reduced_psi_j;

    const double inv_r = 1.0 / std::max(distance, 1e-9);
    return prefactor * std::exp(-GP::kapa * h)
           * (inv_r * inv_r + GP::kapa * inv_r);
}

double Interaction::hydrophobicForce(int i, int j, double distance, Agent objectType_i, Agent objectType_j) {
    const double r1 = GF::getRadius(objectType_i, i);
    const double r2 = GF::getRadius(objectType_j, j);
    const double h = std::max(distance - (r1 + r2), 0.0);

    double contact_energy_kbt = 0.0;
    if (objectType_i == Agent::Synapsin_agent
        && objectType_j == Agent::Synapsin_agent) {
        contact_energy_kbt = GP::hydrophobic_energy_syn_syn_kbt;
    } else if (objectType_i == Agent::Vesicle_agent
               && objectType_j == Agent::Vesicle_agent) {
        contact_energy_kbt = GP::hydrophobic_energy_ves_ves_kbt;
    } else {
        contact_energy_kbt = GP::hydrophobic_energy_syn_ves_kbt;
    }

    const double decay_length = std::max(
            GP::hydrophobic_decay_length_nm, 1e-9);
    const double thermal_energy =
            static_cast<double>(GP::kb) * static_cast<double>(GP::T);

    return -contact_energy_kbt * thermal_energy / decay_length
           * std::exp(-h / decay_length);
}

double Interaction::vanDerWaalsForce(double R, float r1, float r2) {

    const double contact_distance = std::max(
            GP::minimum_surface_separation_nm, 1e-9);
    const double surface_distance = std::max(
            R - static_cast<double>(r1 + r2), contact_distance);

    const double reduced_radius =
            static_cast<double>(r1) * r2 / (r1 + r2);
    return -GP::A * reduced_radius /
           (6.0 * surface_distance * surface_distance);
}

double Interaction::minimizerRepulsionScale(
        int i, int j, Agent objectType_i, Agent objectType_j) {
    const double time_step = std::max(
            static_cast<double>(GP::deltaT), 1e-12);

    const double radius_i = GF::getRadius(objectType_i, i);
    const double radius_j = GF::getRadius(objectType_j, j);
    const double drag_i = std::max(
            GF::getDrag(objectType_i, i), 1e-12);
    const double drag_j = std::max(
            GF::getDrag(objectType_j, j), 1e-12);

    const double force_limit_i = GP::adjusted_max
                                 * (radius_i / 4.0) * drag_i / time_step;
    const double force_limit_j = GP::adjusted_max
                                 * (radius_j / 4.0) * drag_j / time_step;

    return std::min(force_limit_i, force_limit_j);
}
