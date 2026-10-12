#ifndef CODE_MYDISSERATION_Interaction_H
#define CODE_MYDISSERATION_Interaction_H

#include "Synapsin.h"
#include <fstream>
#include <random>
#include <boost/multi_array.hpp>
#include <cstdlib>
#include <ctime>
#include <array>
#include <tuple>
#include <vector>
#include <cstdint>
#include "GP.h"
#include "MyUtility.h"
#include <algorithm>
#include "Vesicles.h"

enum class Agent {
    Synapsin_agent,
    Vesicle_agent
};

class Interaction {
public:
    struct ParticleForceComponents {
        std::array<double, 3> electrostatic{0.0, 0.0, 0.0};
        std::array<double, 3> van_der_waals{0.0, 0.0, 0.0};
        std::array<double, 3> hydrophobic{0.0, 0.0, 0.0};
        std::array<double, 3> steric_core{0.0, 0.0, 0.0};
        int clamped_pair_count = 0;
    };

    struct PairForceResult {
        std::array<double, 3> total_force{0.0, 0.0, 0.0};
        ParticleForceComponents components{};
        double distance = 0.0;
        double overlap = 0.0;
    };

    struct OverlapDiagnostics {
        std::int64_t overlapping_pair_count = 0;
        std::int64_t synapsin_synapsin_pair_count = 0;
        std::int64_t synapsin_vesicle_pair_count = 0;
        std::int64_t vesicle_vesicle_pair_count = 0;
        std::int64_t overlapping_particle_count = 0;
        std::int64_t overlapping_synapsin_count = 0;
        std::int64_t overlapping_vesicle_count = 0;
        double overlap_sum_nm = 0.0;
        double overlap_mean_nm = 0.0;
        double overlap_max_nm = 0.0;
    };

    struct OverlapProjectionDiagnostics {
        std::int64_t pair_count = 0;
        std::int64_t corrected_particle_count = 0;
        std::int64_t residual_pair_count = 0;
        int iterations_completed = 0;
        int vesicle_neighbor_cell_reach = 0;
        double max_excess_overlap_nm = 0.0;
        double residual_max_excess_overlap_nm = 0.0;
        double residual_tolerance_nm = 0.0;
        double vesicle_center_cutoff_nm = 0.0;
        bool converged = true;
        bool hit_iteration_limit = false;
    };

    struct RuntimeDiagnostics {
        std::int64_t dynamic_step_count = 0;
        std::int64_t drift_limited_particle_events = 0;
        std::int64_t drift_limited_synapsin_events = 0;
        std::int64_t drift_limited_vesicle_events = 0;
        std::int64_t drift_limiter_active_steps = 0;
        double sum_drift_limited_fraction = 0.0;
        double max_drift_limited_fraction = 0.0;
        std::int64_t drift_component_diagnostic_events = 0;
        std::int64_t drift_component_unresolved_events = 0;
        std::int64_t drift_dominant_electrostatic_events = 0;
        std::int64_t drift_dominant_van_der_waals_events = 0;
        std::int64_t drift_dominant_hydrophobic_events = 0;
        std::int64_t drift_dominant_steric_core_events = 0;
        double sum_limited_raw_to_cap_ratio = 0.0;
        double max_limited_raw_to_cap_ratio = 0.0;
        double sum_limited_electrostatic_aligned_share = 0.0;
        double sum_limited_van_der_waals_aligned_share = 0.0;
        double sum_limited_hydrophobic_aligned_share = 0.0;
        double sum_limited_steric_core_aligned_share = 0.0;
        std::int64_t euler_particle_proposal_count = 0;
        std::int64_t nonfinite_euler_proposal_count = 0;
        double maximum_raw_drift_displacement_nm = 0.0;
        double maximum_used_drift_displacement_nm = 0.0;
        double maximum_used_drift_fraction_of_radius = 0.0;
        double maximum_brownian_displacement_nm = 0.0;
        double maximum_brownian_displacement_sigma_units = 0.0;
        double maximum_total_proposal_displacement_nm = 0.0;
        double maximum_total_proposal_fraction_of_radius = 0.0;
        std::int64_t projection_active_steps = 0;
        std::int64_t projection_converged_steps = 0;
        std::int64_t projection_nonconverged_steps = 0;
        std::int64_t projection_iteration_limit_steps = 0;
        std::int64_t projection_pair_events = 0;
        std::int64_t projection_corrected_particle_events = 0;
        std::int64_t projection_iteration_count = 0;
        int max_projection_iterations_used = 0;
        int max_projection_vesicle_neighbor_cell_reach = 0;
        double max_projection_excess_overlap_nm = 0.0;
        std::int64_t projection_residual_pair_events = 0;
        double max_projection_residual_excess_nm = 0.0;
        double max_projection_vesicle_center_cutoff_nm = 0.0;
        std::int64_t sk_rock_macro_steps = 0;
        std::int64_t sk_rock_force_evaluations = 0;
        std::int64_t sk_rock_internal_reflected_updates = 0;
        double sk_rock_max_internal_boundary_correction_nm = 0.0;
        std::int64_t srock2_macro_steps = 0;
        std::int64_t srock2_force_evaluations = 0;
        std::int64_t srock2_internal_reflected_updates = 0;
        double srock2_max_internal_boundary_correction_nm = 0.0;
    };

    Interaction();

    static void agentInteraction(bool capture_frame_analysis = false);

    static void calculateForcesBetween(
            Agent objectType_i, Agent objectType_j,
            bool capture_frame_analysis = false);

    static void calculateForcesSelves(
            Agent objectType_i,
            bool capture_frame_analysis = false);

    static PairForceResult calculateTotalForces(
            int i, int j, Agent objectType_i, Agent objectType_j);

    static double allowedOuterOverlapNm(
            Agent objectType_i, Agent objectType_j);

    static void minimizerAgents();

    static OverlapDiagnostics measureOverlaps();

    static OverlapProjectionDiagnostics resolveDynamicOverlaps();
    static void resetRuntimeDiagnostics();
    static void recordRuntimeStep();

    static void recordEulerParticleProposal(
            Agent object_type,
            const std::array<double, 3> &raw_drift,
            const std::array<double, 3> &used_drift,
            const std::array<double, 3> &brownian,
            const std::array<double, 3> &total_proposal,
            double radius_nm,
            double brownian_sigma_nm);

    static void advanceSkRockMacroStep();

    static void advanceSrock2MacroStep();

    static std::array<double, 3> repulsiveForce(int i, int j, Agent objectType_i, Agent objectType_j);

    static double electricForce(int i, int j, double distance, Agent objectType_i, Agent objectType_j);

    static double hydrophobicForce(int i, int j, double distance, Agent objectType_i, Agent objectType_j);

    static double minimizerRepulsionScale(
            int i, int j, Agent objectType_i, Agent objectType_j);

    static double stericCoreForce(double excess_core_overlap_nm);

    static double vanDerWaalsForce(double distance, float r1, float r2);

    static std::vector<ParticleForceComponents> synapsin_force_components;
    static std::vector<ParticleForceComponents> vesicle_force_components;
    static OverlapProjectionDiagnostics last_overlap_projection_diagnostics;
    static RuntimeDiagnostics runtime_diagnostics;

};

#endif
