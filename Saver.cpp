#include "Saver.h"
#include "MyUtility.h"
#include "Vesicles.h"
#include "GP.h"
#include "ProjectManager.h"
#include "Synapsin.h"
#include "Interaction.h"
#include "Converter.h"
#include "GF.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <highfive/H5File.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <system_error>
#include <limits>
#include <cctype>

static constexpr std::size_t CHUNK_FLUSH_BYTES = 1 << 20;
static constexpr int LINE_BUF = 256;

using json = nlohmann::json;

namespace {

namespace fs = std::filesystem;

constexpr int RESUME_CHECKPOINT_SCHEMA_VERSION = 2;
constexpr std::size_t RESUME_CHECKPOINTS_TO_KEEP = 2;
const std::string CHECKPOINT_PREFIX = "resume_checkpoint_step_";
const std::string CHECKPOINT_SUFFIX = ".json";

std::int64_t expectedHdf5FrameCount(
        int segment_start_state_step,
        bool resumed_run) {
    if (GP::load <= 0) {
        throw std::runtime_error(
                "Save interval must be positive before HDF5 initialization");
    }
    if (GP::Total_time < 1) {
        throw std::runtime_error(
                "Total simulation steps must be at least 1");
    }
    if (GP::f_off) {
        return 0;
    }

    const std::int64_t final_state = GP::Total_time - 1;
    if (!resumed_run) {
        return final_state / static_cast<std::int64_t>(GP::load);
    }
    if (segment_start_state_step < 0
        || segment_start_state_step >= final_state) {
        throw std::runtime_error(
                "A resumed run must extend beyond its checkpoint step");
    }

    return 1
           + final_state / static_cast<std::int64_t>(GP::load)
           - static_cast<std::int64_t>(segment_start_state_step)
             / static_cast<std::int64_t>(GP::load);
}

json currentResumeConfiguration() {
    return json{
            {"state_compatibility", {
                    {"synapsin_count", GP::number_of_SynapsinI},
                    {"vesicle_count", GP::number_of_vesicles},
                    {"environment_shape", GP::shape},
                    {"environment_length_nm", GP::length},
                    {"environment_width_nm", GP::width},
                    {"environment_height_nm", GP::height},
                    {"environment_steps_nm", GP::steps},
                    {"environment_out", GP::out},
                    {"environment_slide", GP::slide},
                    {"environment_scale", GP::scale},
                    {"environment_box_fraction", GP::box_fract},
                    {"environment_volume", GP::volume},
                    {"environment_boundary_labels", GP::labels_out_bound},
                    {"delta_t_ms", GP::deltaT},
                    {"boltzmann", GP::kb},
                    {"temperature", GP::T},
                    {"dimension", GP::dim},
                    {"viscosity", GP::eta},
                    {"vesicle_D_multiplier", GP::D_ves},
                    {"vesicle_drag_multiplier", GP::drag_ves},
                    {"vesicle_mean_radius_nm", GP::mean_radius},
                    {"vesicle_radius_std_nm", GP::std_radius},
                    {"vesicle_core_radius_min_nm", GP::vesicle_core_radius_min_nm},
                    {"vesicle_core_radius_max_nm", GP::vesicle_core_radius_max_nm},
                    {"vesicle_corona_thickness_nm", GP::vesicle_corona_thickness_nm},
                    {"synapsin_D_multiplier", GP::D_syn},
                    {"synapsin_drag_multiplier", GP::drag_syn},
                    {"synapsin_radius_nm", GP::synapsinI_radius}
            }},
            {"protocol_parameters", {
                    {"force_model_schema_version", 4},
                    {"electrostatic_kernel", "linear_pb_sphere_yukawa"},
                    {"hydrophobic_kernel", "exponential_contact_pmf"},
                    {"vdw_kernel", "hamaker_derjaguin"},
                    {"steric_kernel", "pair_specific_corona_constraint"},
                    {"forces_disabled", GP::f_off},
                    {"centering", GP::centering},
                    {"neighbor_type", GP::nei},
                    {"neighbor_level", GP::neiLev},
                    {"connectivity_shell_nm", GP::cut_off},
                    {"adjusted_max", GP::adjusted_max},
                    {"dynamic_drift_limit_fraction_of_radius",
                     GP::dynamic_drift_limit_fraction_of_radius},
                    {"enable_overlap_projection", GP::enable_overlap_projection},
                    {"overlap_projection_max_iterations", GP::overlap_projection_max_iterations},
                    {"overlap_projection_residual_tolerance_nm",
                     GP::overlap_projection_residual_tolerance_nm},
                    {"enable_drift_component_diagnostics",
                     GP::enable_drift_component_diagnostics},
                    {"enable_electrostatic", GP::enable_electrostatic},
                    {"enable_van_der_waals", GP::enable_van_der_waals},
                    {"enable_hydrophobic", GP::enable_hydrophobic},
                    {"core_allowed_overlap_nm", GP::core_allowed_overlap_nm},
                    {"core_energy_kbt", GP::core_energy_kbt},
                    {"enable_dynamic_steric_penalty",
                     GP::enable_dynamic_steric_penalty},
                    {"minimum_surface_separation_nm", GP::minimum_surface_separation_nm},
                    {"hydrophobic_decay_length_nm", GP::hydrophobic_decay_length_nm},
                    {"hydrophobic_energy_syn_syn_kbt", GP::hydrophobic_energy_syn_syn_kbt},
                    {"hydrophobic_energy_syn_ves_kbt", GP::hydrophobic_energy_syn_ves_kbt},
                    {"hydrophobic_energy_ves_ves_kbt", GP::hydrophobic_energy_ves_ves_kbt},
                    {"electronic_charge", GP::eCharge},
                    {"permittivity", GP::e0},
                    {"hamaker", GP::A},
                    {"solvent_relative_permittivity", GP::eSolvent},
                    {"ionic_strength_molar", GP::ionic_strength_molar},
                    {"kappa_nm_inverse", GP::kapa},
                    {"vesicle_surface_potential_mV", GP::vesicle_surface_potential_mV},
                    {"synapsin_surface_potential_mV", GP::synapsin_surface_potential_mV}
            }}
    };
}

bool jsonNumbersClose(const json &left, const json &right) {
    if (left.type() != right.type()) {
        if (!(left.is_number() && right.is_number())) {
            return false;
        }
    }
    if (left.is_number() && right.is_number()) {
        const double a = left.get<double>();
        const double b = right.get<double>();
        const double scale = std::max(std::fabs(a), std::fabs(b));

        return std::fabs(a - b) <= 1e-12 + 1e-7 * scale;
    }
    if (left.is_array() && right.is_array()) {
        if (left.size() != right.size()) return false;
        for (std::size_t i = 0; i < left.size(); ++i) {
            if (!jsonNumbersClose(left[i], right[i])) return false;
        }
        return true;
    }
    if (left.is_object() && right.is_object()) {
        if (left.size() != right.size()) return false;
        for (auto it = left.begin(); it != left.end(); ++it) {
            if (!right.contains(it.key())
                || !jsonNumbersClose(it.value(), right.at(it.key()))) {
                return false;
            }
        }
        return true;
    }
    return left == right;
}

int checkpointStepFromFilename(const fs::path &path) {
    const std::string name = path.filename().string();
    if (name.size() <= CHECKPOINT_PREFIX.size() + CHECKPOINT_SUFFIX.size()
        || name.compare(0, CHECKPOINT_PREFIX.size(), CHECKPOINT_PREFIX) != 0
        || name.compare(
                name.size() - CHECKPOINT_SUFFIX.size(),
                CHECKPOINT_SUFFIX.size(),
                CHECKPOINT_SUFFIX) != 0) {
        return -1;
    }
    const std::string step_text = name.substr(
            CHECKPOINT_PREFIX.size(),
            name.size() - CHECKPOINT_PREFIX.size()
            - CHECKPOINT_SUFFIX.size());
    if (step_text.empty()
        || !std::all_of(
                step_text.begin(), step_text.end(), [](unsigned char c) {
                    return std::isdigit(c) != 0;
                })) {
        return -1;
    }
    try {
        const long long step = std::stoll(step_text);
        if (step < 0 || step > std::numeric_limits<int>::max()) return -1;
        return static_cast<int>(step);
    } catch (...) {
        return -1;
    }
}

std::vector<std::pair<int, fs::path>> checkpointFiles(
        const fs::path &directory) {
    std::vector<std::pair<int, fs::path>> files;
    if (!fs::is_directory(directory)) return files;
    for (const auto &entry: fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const int step = checkpointStepFromFilename(entry.path());
        if (step >= 0) files.emplace_back(step, entry.path());
    }
    std::sort(files.begin(), files.end(), [](const auto &left, const auto &right) {
        return left.first > right.first;
    });
    return files;
}

}

void Saver::initializeRunMetadataToHDF5(
        const std::string &filename,
        int segment_start_state_step,
        int resume_mode_code,
        bool resume_parameters_changed) {
    using namespace HighFive;

    if (Synapsin::locations.size()
        != static_cast<std::size_t>(Synapsin::number)
        || Vesicles::locations.size()
           != static_cast<std::size_t>(Vesicles::number)) {
        throw std::runtime_error(
                "Run metadata particle counts do not match location arrays");
    }

    File file(filename, File::OpenOrCreate);
    Group group = file.createGroup("run_metadata");

    DataSetCreateProps props_scalar;
    props_scalar.add(Chunking(std::vector<hsize_t>{1}));
    props_scalar.add(Shuffle());
    props_scalar.add(Deflate(6));

    const auto int64Value = [](std::int64_t value) {
        return std::vector<std::int64_t>{value};
    };
    const auto uint64Value = [](std::uint64_t value) {
        return std::vector<std::uint64_t>{value};
    };
    const auto doubleValue = [](double value) {
        return std::vector<double>{value};
    };

    const bool resumed_run = resume_mode_code != 0;
    const std::int64_t expected_frames = expectedHdf5FrameCount(
            segment_start_state_step, resumed_run);

    const std::int64_t expected_frame_dataset_count =
            GP::f_off ? 0 : (GP::save_forces ? 59 : 55);
    const std::int64_t particle_count =
            static_cast<std::int64_t>(Synapsin::number)
            + static_cast<std::int64_t>(Vesicles::number);

    group.createDataSet(
            "hdf5_schema_version", int64Value(8), props_scalar);
    group.createDataSet(
            "frame_schema_version", int64Value(7), props_scalar);
    group.createDataSet(
            "expected_frame_dataset_count",
            int64Value(expected_frame_dataset_count),
            props_scalar);

    group.createDataSet(
            "actual_seed",
            uint64Value(static_cast<std::uint64_t>(GP::seed)),
            props_scalar);
    group.createDataSet(
            "use_seed", int64Value(GP::use_seed ? 1 : 0), props_scalar);
    group.createDataSet(
            "save_forces", int64Value(GP::save_forces ? 1 : 0), props_scalar);
    group.createDataSet(
            "forces_enabled", int64Value(GP::f_off ? 0 : 1), props_scalar);
    group.createDataSet(
            "data_analysis", int64Value(GP::dataAnalysis ? 1 : 0), props_scalar);
    group.createDataSet(
            "resumed_run", int64Value(resumed_run ? 1 : 0), props_scalar);
    group.createDataSet(
            "resume_mode_code", int64Value(resume_mode_code), props_scalar);
    group.createDataSet(
            "resume_parameters_changed",
            int64Value(resume_parameters_changed ? 1 : 0),
            props_scalar);
    group.createDataSet(
            "segment_start_state_step",
            int64Value(segment_start_state_step),
            props_scalar);
    group.createDataSet(
            "segment_start_time_ms",
            doubleValue(
                    static_cast<double>(segment_start_state_step)
                    * GP::deltaT),
            props_scalar);

    group.createDataSet(
            "synapsin_count", int64Value(Synapsin::number), props_scalar);
    group.createDataSet(
            "vesicle_count", int64Value(Vesicles::number), props_scalar);
    group.createDataSet(
            "particle_count", int64Value(particle_count), props_scalar);

    group.createDataSet(
            "total_time_steps", int64Value(GP::Total_time), props_scalar);
    group.createDataSet(
            "expected_final_state_step",
            int64Value(GP::Total_time - 1),
            props_scalar);
    group.createDataSet(
            "save_interval_steps", int64Value(GP::load), props_scalar);
    group.createDataSet(
            "expected_saved_frame_count",
            int64Value(expected_frames),
            props_scalar);
    group.createDataSet(
            "delta_t_ms", doubleValue(GP::deltaT), props_scalar);
    group.createDataSet(
            "frame_interval_ms",
            doubleValue(static_cast<double>(GP::load) * GP::deltaT),
            props_scalar);
    group.createDataSet(
            "intended_duration_ms",
            doubleValue(static_cast<double>(GP::Total_time) * GP::deltaT),
            props_scalar);
    group.createDataSet(
            "last_integrated_time_ms",
            doubleValue(static_cast<double>(GP::Total_time - 1) * GP::deltaT),
            props_scalar);

    group.createDataSet(
            "environment_shape", int64Value(GP::shape), props_scalar);
    group.createDataSet(
            "environment_length_nm", doubleValue(GP::length), props_scalar);
    group.createDataSet(
            "environment_width_nm", doubleValue(GP::width), props_scalar);
    group.createDataSet(
            "environment_height_nm", doubleValue(GP::height), props_scalar);

    group.createDataSet(
            "temperature_kelvin", doubleValue(GP::T), props_scalar);
    group.createDataSet(
            "boltzmann_simulation_units", doubleValue(GP::kb), props_scalar);
    group.createDataSet(
            "thermal_energy_simulation_units",
            doubleValue(static_cast<double>(GP::kb) * GP::T), props_scalar);
    group.createDataSet(
            "simulation_force_units_per_kbt_per_nm",
            doubleValue(static_cast<double>(GP::kb) * GP::T), props_scalar);
    group.createDataSet(
            "viscosity_simulation_units", doubleValue(GP::eta), props_scalar);
    group.createDataSet(
            "elementary_charge_simulation_units",
            doubleValue(GP::eCharge), props_scalar);
    group.createDataSet(
            "vacuum_permittivity_simulation_units",
            doubleValue(GP::e0), props_scalar);
    group.createDataSet(
            "synapsin_radius_nm",
            doubleValue(GP::synapsinI_radius), props_scalar);
    group.createDataSet(
            "vesicle_mean_radius_nm",
            doubleValue(GP::mean_radius), props_scalar);
    group.createDataSet(
            "vesicle_radius_std_nm",
            doubleValue(GP::std_radius), props_scalar);
    group.createDataSet(
            "vesicle_core_radius_min_nm",
            doubleValue(GP::vesicle_core_radius_min_nm), props_scalar);
    group.createDataSet(
            "vesicle_core_radius_max_nm",
            doubleValue(GP::vesicle_core_radius_max_nm), props_scalar);
    group.createDataSet(
            "vesicle_corona_thickness_nm",
            doubleValue(GP::vesicle_corona_thickness_nm), props_scalar);

    group.createDataSet(
            "minimizer_max_iterations",
            int64Value(GP::loop_minimiz),
            props_scalar);
    group.createDataSet(
            "minimizer_overlap_tolerance_nm",
            doubleValue(GP::minimizer_overlap_tolerance_nm),
            props_scalar);
    group.createDataSet(
            "minimizer_consecutive_checks",
            int64Value(GP::minimizer_consecutive_checks),
            props_scalar);

    group.createDataSet(
            "connectivity_shell_nm", doubleValue(GP::cut_off), props_scalar);
    group.createDataSet(
            "adjusted_max", doubleValue(GP::adjusted_max), props_scalar);
    group.createDataSet(
            "deterministic_drift_limit_fraction_of_radius",
            doubleValue(GP::dynamic_drift_limit_fraction_of_radius),
            props_scalar);
    group.createDataSet(
            "enable_overlap_projection",
            int64Value(GP::enable_overlap_projection ? 1 : 0), props_scalar);
    group.createDataSet(
            "overlap_projection_max_iterations",
            int64Value(GP::overlap_projection_max_iterations), props_scalar);
    group.createDataSet(
            "enable_electrostatic",
            int64Value(GP::enable_electrostatic ? 1 : 0), props_scalar);
    group.createDataSet(
            "enable_van_der_waals",
            int64Value(GP::enable_van_der_waals ? 1 : 0), props_scalar);
    group.createDataSet(
            "enable_hydrophobic",
            int64Value(GP::enable_hydrophobic ? 1 : 0), props_scalar);
    group.createDataSet(
            "core_allowed_overlap_nm",
            doubleValue(GP::core_allowed_overlap_nm), props_scalar);
    group.createDataSet(
            "core_energy_kbt",
            doubleValue(GP::core_energy_kbt), props_scalar);
    group.createDataSet(
            "dynamic_steric_penalty_enabled",
            int64Value(GP::enable_dynamic_steric_penalty ? 1 : 0),
            props_scalar);
    group.createDataSet(
            "minimum_surface_separation_nm",
            doubleValue(GP::minimum_surface_separation_nm), props_scalar);
    group.createDataSet(
            "hydrophobic_decay_length_nm",
            doubleValue(GP::hydrophobic_decay_length_nm), props_scalar);
    group.createDataSet(
            "hydrophobic_energy_syn_syn_kbt",
            doubleValue(GP::hydrophobic_energy_syn_syn_kbt), props_scalar);
    group.createDataSet(
            "hydrophobic_energy_syn_ves_kbt",
            doubleValue(GP::hydrophobic_energy_syn_ves_kbt), props_scalar);
    group.createDataSet(
            "hydrophobic_energy_ves_ves_kbt",
            doubleValue(GP::hydrophobic_energy_ves_ves_kbt), props_scalar);
    group.createDataSet(
            "ionic_strength_molar",
            doubleValue(GP::ionic_strength_molar), props_scalar);
    group.createDataSet(
            "debye_length_nm",
            doubleValue(GP::debye_length_nm), props_scalar);
    group.createDataSet(
            "kappa_nm_inverse",
            doubleValue(GP::kapa), props_scalar);
    group.createDataSet(
            "solvent_relative_permittivity",
            doubleValue(GP::eSolvent), props_scalar);
    group.createDataSet(
            "synapsin_surface_potential_mV",
            doubleValue(GP::synapsin_surface_potential_mV), props_scalar);
    group.createDataSet(
            "vesicle_surface_potential_mV",
            doubleValue(GP::vesicle_surface_potential_mV), props_scalar);
    group.createDataSet(
            "effective_hamaker_simulation_units",
            doubleValue(GP::A), props_scalar);

    group.createDataSet(
            "force_model_schema_version", int64Value(4), props_scalar);
    group.createDataSet(
            "electrostatic_kernel_linear_pb_sphere_yukawa",
            int64Value(1), props_scalar);
    group.createDataSet(
            "hydrophobic_kernel_exponential_contact_pmf",
            int64Value(1), props_scalar);
    group.createDataSet(
            "vdw_kernel_hamaker_derjaguin", int64Value(1), props_scalar);
    group.createDataSet(
            "steric_kernel_pair_specific_corona_constraint",
            int64Value(1), props_scalar);
    group.createDataSet(
            "scientific_force_terms_independent",
            int64Value(1), props_scalar);
    group.createDataSet(
            "saved_force_components_are_raw",
            int64Value(1), props_scalar);
    group.createDataSet(
            "pair_force_clamp_enabled",
            int64Value(0), props_scalar);
    group.createDataSet(
            "particle_drift_limiter_enabled",
            int64Value(1), props_scalar);
    group.createDataSet(
            "brownian_displacement_is_not_limited",
            int64Value(1), props_scalar);
    group.createDataSet(
            "overlap_projection_mobility_weighted_jacobi",
            int64Value(1), props_scalar);
    group.createDataSet(
            "overlap_projection_allows_soft_core_overlap",
            int64Value(1), props_scalar);
    group.createDataSet(
            "total_force_magnitude_from_same_frame_vector",
            int64Value(1), props_scalar);

    if (ProjectManager::getRunProvenance().is_null()
        || ProjectManager::getRunProvenance().empty()) {
        ProjectManager::captureRunProvenance();
    }
    const json &provenance = ProjectManager::getRunProvenance();
    const json &source_snapshot = provenance.at("source_snapshot");
    const json &input_dashboard = provenance.at("input_dashboard");
    const json &git = provenance.at("git");
    const json &build = provenance.at("build");
    const json &runtime = provenance.at("runtime");

    Group provenance_group = file.createGroup("provenance");
    provenance_group.createDataSet(
            "provenance_schema_version",
            int64Value(provenance.at("provenance_schema_version").get<int>()),
            props_scalar);
    provenance_group.createDataSet(
            "source_snapshot_file_count",
            int64Value(source_snapshot.at("file_count").get<std::int64_t>()),
            props_scalar);
    provenance_group.createDataSet(
            "source_snapshot_fnv1a64",
            uint64Value(
                    source_snapshot.at("aggregate_fnv1a64")
                            .get<std::uint64_t>()),
            props_scalar);
    provenance_group.createDataSet(
            "input_dashboard_fnv1a64",
            uint64Value(
                    input_dashboard.at("fnv1a64").get<std::uint64_t>()),
            props_scalar);
    provenance_group.createDataSet(
            "input_dashboard_size_bytes",
            uint64Value(
                    input_dashboard.at("size_bytes").get<std::uint64_t>()),
            props_scalar);
    provenance_group.createDataSet(
            "git_repository_detected",
            int64Value(git.value("repository_detected", false) ? 1 : 0),
            props_scalar);
    provenance_group.createDataSet(
            "git_dirty_known",
            int64Value(git.value("dirty_known", false) ? 1 : 0),
            props_scalar);
    provenance_group.createDataSet(
            "git_dirty",
            int64Value(git.value("dirty", false) ? 1 : 0),
            props_scalar);
    provenance_group.createDataSet(
            "git_status_entry_count",
            int64Value(git.value("status_entry_count", 0)),
            props_scalar);
    provenance_group.createDataSet(
            "cplusplus",
            int64Value(build.value("cplusplus", 0LL)),
            props_scalar);
    provenance_group.createDataSet(
            "release_build",
            int64Value(build.value("release_build", false) ? 1 : 0),
            props_scalar);
    provenance_group.createDataSet(
            "openmp_enabled",
            int64Value(runtime.value("openmp_enabled", false) ? 1 : 0),
            props_scalar);
    provenance_group.createDataSet(
            "logical_cores",
            int64Value(runtime.value("logical_cores", 1)),
            props_scalar);
    provenance_group.createDataSet(
            "openmp_available_processors",
            int64Value(runtime.value("openmp_available_processors", 1)),
            props_scalar);
    provenance_group.createDataSet(
            "detected_physical_cores",
            int64Value(runtime.value("detected_physical_cores", 1)),
            props_scalar);
    provenance_group.createDataSet(
            "reserved_cores",
            int64Value(runtime.value("reserved_cores", 0)),
            props_scalar);
    provenance_group.createDataSet(
            "openmp_selected_threads",
            int64Value(runtime.value("openmp_selected_threads", 1)),
            props_scalar);
    provenance_group.createDataSet(
            "explicit_thread_override",
            int64Value(
                    runtime.value("explicit_thread_override", false) ? 1 : 0),
            props_scalar);
    provenance_group.createDataSet(
            "executable_sha256_known",
            int64Value(
                    runtime.value("executable_sha256_known", false) ? 1 : 0),
            props_scalar);

    const auto writeAsciiBytes = [&provenance_group](
            const std::string &dataset_name,
            std::string text_value) {
        using namespace HighFive;
        if (text_value.empty()) text_value = "unknown";
        const std::vector<std::uint8_t> bytes(
                text_value.begin(), text_value.end());
        DataSetCreateProps props_ascii;
        props_ascii.add(Chunking(
                std::vector<hsize_t>{bytes.size()}));
        props_ascii.add(Shuffle());
        props_ascii.add(Deflate(6));
        provenance_group.createDataSet(
                dataset_name, bytes, props_ascii);
    };

    writeAsciiBytes(
            "git_commit_ascii",
            git.value("commit", std::string("unknown")));
    writeAsciiBytes(
            "git_branch_ascii",
            git.value("branch", std::string("unknown")));
    writeAsciiBytes(
            "compiler_id_ascii",
            build.value("compiler_id", std::string("unknown")));
    writeAsciiBytes(
            "compiler_version_ascii",
            build.value("compiler_version", std::string("unknown")));
    writeAsciiBytes(
            "thread_policy_ascii",
            runtime.value("thread_policy", std::string("unknown")));
    writeAsciiBytes(
            "executable_sha256_ascii",
            runtime.value("executable_sha256", std::string("unknown")));
    writeAsciiBytes("run_provenance_json_ascii", provenance.dump());

    file.flush();
}

void Saver::finalizeRunMetadataToHDF5(
        const std::string &filename,
        int final_state_step,
        int saved_hdf5_frame_count,
        int segment_start_state_step,
        bool resumed_run) {
    using namespace HighFive;

    const std::int64_t expected_frames = expectedHdf5FrameCount(
            segment_start_state_step, resumed_run);
    const std::int64_t expected_final_state = GP::Total_time - 1;
    const std::int64_t particle_count =
            static_cast<std::int64_t>(Synapsin::locations.size())
            + static_cast<std::int64_t>(Vesicles::locations.size());

    if (final_state_step != expected_final_state
        || saved_hdf5_frame_count != expected_frames
        || particle_count
           != static_cast<std::int64_t>(Synapsin::number)
              + static_cast<std::int64_t>(Vesicles::number)) {
        throw std::runtime_error(
                "Run completion checks failed; HDF5 was not marked complete");
    }

    File file(filename, File::OpenOrCreate);
    Group group = file.createGroup("run_completion");

    DataSetCreateProps props_scalar;
    props_scalar.add(Chunking(std::vector<hsize_t>{1}));
    props_scalar.add(Shuffle());
    props_scalar.add(Deflate(6));

    const auto int64Value = [](std::int64_t value) {
        return std::vector<std::int64_t>{value};
    };
    const auto doubleValue = [](double value) {
        return std::vector<double>{value};
    };

    group.createDataSet(
            "completion_schema_version", int64Value(3), props_scalar);
    group.createDataSet(
            "run_completed", int64Value(1), props_scalar);
    group.createDataSet(
            "final_state_step", int64Value(final_state_step), props_scalar);
    group.createDataSet(
            "final_physical_time_ms",
            doubleValue(static_cast<double>(final_state_step) * GP::deltaT),
            props_scalar);
    group.createDataSet(
            "saved_hdf5_frame_count",
            int64Value(saved_hdf5_frame_count),
            props_scalar);
    group.createDataSet(
            "expected_saved_frame_count",
            int64Value(expected_frames),
            props_scalar);
    group.createDataSet(
            "all_expected_frames_saved", int64Value(1), props_scalar);
    group.createDataSet(
            "particle_count_at_completion",
            int64Value(particle_count),
            props_scalar);
    group.createDataSet(
            "integrator_code",
            int64Value(GP::integrator == "sk_rock" ? 1
                       : (GP::integrator == "s_rock2" ? 2 : 0)),
            props_scalar);
    group.createDataSet(
            "sk_rock_configured_stages",
            int64Value(GP::sk_rock_stages),
            props_scalar);
    group.createDataSet(
            "sk_rock_configured_damping",
            doubleValue(GP::sk_rock_damping),
            props_scalar);
    group.createDataSet(
            "srock2_configured_degree",
            int64Value(GP::srock2_degree),
            props_scalar);
    group.createDataSet(
            "continuous_contact_forces_enabled",
            int64Value(GP::continuous_contact_forces ? 1 : 0),
            props_scalar);
    group.createDataSet(
            "dynamic_steric_penalty_enabled_in_dynamics",
            int64Value(GP::enable_dynamic_steric_penalty ? 1 : 0),
            props_scalar);
    group.createDataSet(
            "overlap_projection_residual_tolerance_nm",
            doubleValue(GP::overlap_projection_residual_tolerance_nm),
            props_scalar);
    group.createDataSet(
            "drift_component_diagnostics_enabled",
            int64Value(GP::enable_drift_component_diagnostics ? 1 : 0),
            props_scalar);

    const auto &runtime = Interaction::runtime_diagnostics;
    const auto &boundary = GF::boundary_diagnostics;
    const double mean_limited_fraction = runtime.dynamic_step_count > 0
            ? runtime.sum_drift_limited_fraction
              / static_cast<double>(runtime.dynamic_step_count)
            : 0.0;
    const double mean_raw_to_cap_ratio =
            runtime.drift_component_diagnostic_events > 0
            ? runtime.sum_limited_raw_to_cap_ratio
              / static_cast<double>(runtime.drift_component_diagnostic_events)
            : 0.0;
    const std::int64_t resolved_component_events =
            runtime.drift_component_diagnostic_events
            - runtime.drift_component_unresolved_events;
    const auto meanResolvedShare = [resolved_component_events](double sum) {
        return resolved_component_events > 0
                ? sum / static_cast<double>(resolved_component_events)
                : 0.0;
    };
    group.createDataSet("dynamic_steps_recorded",
                        int64Value(runtime.dynamic_step_count), props_scalar);
    group.createDataSet("drift_limited_particle_events",
                        int64Value(runtime.drift_limited_particle_events), props_scalar);
    group.createDataSet("drift_limited_synapsin_events",
                        int64Value(runtime.drift_limited_synapsin_events), props_scalar);
    group.createDataSet("drift_limited_vesicle_events",
                        int64Value(runtime.drift_limited_vesicle_events), props_scalar);
    group.createDataSet("drift_limiter_active_steps",
                        int64Value(runtime.drift_limiter_active_steps), props_scalar);
    group.createDataSet("mean_drift_limited_fraction",
                        doubleValue(mean_limited_fraction), props_scalar);
    group.createDataSet("maximum_drift_limited_fraction",
                        doubleValue(runtime.max_drift_limited_fraction), props_scalar);
    group.createDataSet("drift_component_diagnostic_events",
                        int64Value(runtime.drift_component_diagnostic_events), props_scalar);
    group.createDataSet("drift_component_unresolved_events",
                        int64Value(runtime.drift_component_unresolved_events), props_scalar);
    group.createDataSet("drift_dominant_electrostatic_events",
                        int64Value(runtime.drift_dominant_electrostatic_events), props_scalar);
    group.createDataSet("drift_dominant_van_der_waals_events",
                        int64Value(runtime.drift_dominant_van_der_waals_events), props_scalar);
    group.createDataSet("drift_dominant_hydrophobic_events",
                        int64Value(runtime.drift_dominant_hydrophobic_events), props_scalar);
    group.createDataSet("drift_dominant_steric_core_events",
                        int64Value(runtime.drift_dominant_steric_core_events), props_scalar);
    group.createDataSet("mean_limited_raw_to_cap_ratio",
                        doubleValue(mean_raw_to_cap_ratio), props_scalar);
    group.createDataSet("maximum_limited_raw_to_cap_ratio",
                        doubleValue(runtime.max_limited_raw_to_cap_ratio), props_scalar);
    group.createDataSet("mean_limited_electrostatic_aligned_share",
                        doubleValue(meanResolvedShare(
                                runtime.sum_limited_electrostatic_aligned_share)),
                        props_scalar);
    group.createDataSet("mean_limited_van_der_waals_aligned_share",
                        doubleValue(meanResolvedShare(
                                runtime.sum_limited_van_der_waals_aligned_share)),
                        props_scalar);
    group.createDataSet("mean_limited_hydrophobic_aligned_share",
                        doubleValue(meanResolvedShare(
                                runtime.sum_limited_hydrophobic_aligned_share)),
                        props_scalar);
    group.createDataSet("mean_limited_steric_core_aligned_share",
                        doubleValue(meanResolvedShare(
                                runtime.sum_limited_steric_core_aligned_share)),
                        props_scalar);
    group.createDataSet("euler_particle_proposal_count",
                        int64Value(runtime.euler_particle_proposal_count), props_scalar);
    group.createDataSet("nonfinite_euler_proposal_count",
                        int64Value(runtime.nonfinite_euler_proposal_count), props_scalar);
    group.createDataSet("maximum_raw_drift_displacement_nm",
                        doubleValue(runtime.maximum_raw_drift_displacement_nm), props_scalar);
    group.createDataSet("maximum_used_drift_displacement_nm",
                        doubleValue(runtime.maximum_used_drift_displacement_nm), props_scalar);
    group.createDataSet("maximum_used_drift_fraction_of_radius",
                        doubleValue(runtime.maximum_used_drift_fraction_of_radius), props_scalar);
    group.createDataSet("maximum_brownian_displacement_nm",
                        doubleValue(runtime.maximum_brownian_displacement_nm), props_scalar);
    group.createDataSet("maximum_brownian_displacement_sigma_units",
                        doubleValue(runtime.maximum_brownian_displacement_sigma_units), props_scalar);
    group.createDataSet("maximum_total_proposal_displacement_nm",
                        doubleValue(runtime.maximum_total_proposal_displacement_nm), props_scalar);
    group.createDataSet("maximum_total_proposal_fraction_of_radius",
                        doubleValue(runtime.maximum_total_proposal_fraction_of_radius), props_scalar);
    group.createDataSet("projection_active_steps",
                        int64Value(runtime.projection_active_steps), props_scalar);
    group.createDataSet("projection_converged_steps",
                        int64Value(runtime.projection_converged_steps), props_scalar);
    group.createDataSet("projection_nonconverged_steps",
                        int64Value(runtime.projection_nonconverged_steps), props_scalar);
    group.createDataSet("projection_iteration_limit_steps",
                        int64Value(runtime.projection_iteration_limit_steps), props_scalar);
    group.createDataSet("projection_pair_events",
                        int64Value(runtime.projection_pair_events), props_scalar);
    group.createDataSet("projection_corrected_particle_events",
                        int64Value(runtime.projection_corrected_particle_events), props_scalar);
    group.createDataSet("projection_iteration_count",
                        int64Value(runtime.projection_iteration_count), props_scalar);
    group.createDataSet("maximum_projection_iterations_used",
                        int64Value(runtime.max_projection_iterations_used), props_scalar);
    group.createDataSet("maximum_projection_vesicle_neighbor_cell_reach",
                        int64Value(
                                runtime.max_projection_vesicle_neighbor_cell_reach),
                        props_scalar);
    group.createDataSet("maximum_projection_excess_overlap_nm",
                        doubleValue(runtime.max_projection_excess_overlap_nm), props_scalar);
    group.createDataSet("projection_residual_pair_events",
                        int64Value(runtime.projection_residual_pair_events), props_scalar);
    group.createDataSet("maximum_projection_residual_excess_nm",
                        doubleValue(runtime.max_projection_residual_excess_nm), props_scalar);
    group.createDataSet("maximum_projection_vesicle_center_cutoff_nm",
                        doubleValue(
                                runtime.max_projection_vesicle_center_cutoff_nm),
                        props_scalar);
    group.createDataSet("boundary_position_update_calls",
                        int64Value(boundary.position_update_calls), props_scalar);
    group.createDataSet("boundary_reflected_update_count",
                        int64Value(boundary.reflected_update_count), props_scalar);
    group.createDataSet("boundary_reflected_axis_count",
                        int64Value(boundary.reflected_axis_count), props_scalar);
    group.createDataSet("boundary_rejected_update_count",
                        int64Value(boundary.rejected_update_count), props_scalar);
    group.createDataSet("nonfinite_position_update_count",
                        int64Value(boundary.nonfinite_position_update_count), props_scalar);
    group.createDataSet("maximum_requested_displacement_nm",
                        doubleValue(boundary.maximum_requested_displacement_nm), props_scalar);
    group.createDataSet("maximum_boundary_proposal_correction_nm",
                        doubleValue(boundary.maximum_proposal_correction_nm), props_scalar);
    group.createDataSet("sk_rock_macro_steps",
                        int64Value(runtime.sk_rock_macro_steps), props_scalar);
    group.createDataSet("sk_rock_force_evaluations",
                        int64Value(runtime.sk_rock_force_evaluations), props_scalar);
    group.createDataSet("sk_rock_internal_reflected_updates",
                        int64Value(runtime.sk_rock_internal_reflected_updates), props_scalar);
    group.createDataSet("sk_rock_max_internal_boundary_correction_nm",
                        doubleValue(runtime.sk_rock_max_internal_boundary_correction_nm), props_scalar);
    group.createDataSet("srock2_macro_steps",
                        int64Value(runtime.srock2_macro_steps), props_scalar);
    group.createDataSet("srock2_force_evaluations",
                        int64Value(runtime.srock2_force_evaluations), props_scalar);
    group.createDataSet("srock2_internal_reflected_updates",
                        int64Value(runtime.srock2_internal_reflected_updates), props_scalar);
    group.createDataSet("srock2_max_internal_boundary_correction_nm",
                        doubleValue(runtime.srock2_max_internal_boundary_correction_nm), props_scalar);

    file.flush();

    const json numerical_diagnostics = {
            {"schema_version", 5},
            {"integrator", GP::integrator},
            {"sk_rock_configured_stages", GP::sk_rock_stages},
            {"sk_rock_configured_damping", GP::sk_rock_damping},
            {"srock2_configured_degree", GP::srock2_degree},
            {"continuous_contact_forces_enabled",
             GP::continuous_contact_forces},
            {"dynamic_steric_penalty_enabled",
             GP::enable_dynamic_steric_penalty},
            {"overlap_projection_residual_tolerance_nm",
             GP::overlap_projection_residual_tolerance_nm},
            {"drift_component_diagnostics_enabled",
             GP::enable_drift_component_diagnostics},
            {"dynamic_steps_recorded", runtime.dynamic_step_count},
            {"drift_limited_particle_events", runtime.drift_limited_particle_events},
            {"drift_limited_synapsin_events", runtime.drift_limited_synapsin_events},
            {"drift_limited_vesicle_events", runtime.drift_limited_vesicle_events},
            {"drift_limiter_active_steps", runtime.drift_limiter_active_steps},
            {"mean_drift_limited_fraction", mean_limited_fraction},
            {"maximum_drift_limited_fraction", runtime.max_drift_limited_fraction},
            {"drift_component_diagnostic_events",
             runtime.drift_component_diagnostic_events},
            {"drift_component_unresolved_events",
             runtime.drift_component_unresolved_events},
            {"drift_dominant_electrostatic_events",
             runtime.drift_dominant_electrostatic_events},
            {"drift_dominant_van_der_waals_events",
             runtime.drift_dominant_van_der_waals_events},
            {"drift_dominant_hydrophobic_events",
             runtime.drift_dominant_hydrophobic_events},
            {"drift_dominant_steric_core_events",
             runtime.drift_dominant_steric_core_events},
            {"mean_limited_raw_to_cap_ratio", mean_raw_to_cap_ratio},
            {"maximum_limited_raw_to_cap_ratio",
             runtime.max_limited_raw_to_cap_ratio},
            {"mean_limited_electrostatic_aligned_share",
             meanResolvedShare(runtime.sum_limited_electrostatic_aligned_share)},
            {"mean_limited_van_der_waals_aligned_share",
             meanResolvedShare(runtime.sum_limited_van_der_waals_aligned_share)},
            {"mean_limited_hydrophobic_aligned_share",
             meanResolvedShare(runtime.sum_limited_hydrophobic_aligned_share)},
            {"mean_limited_steric_core_aligned_share",
             meanResolvedShare(runtime.sum_limited_steric_core_aligned_share)},
            {"euler_particle_proposal_count",
             runtime.euler_particle_proposal_count},
            {"nonfinite_euler_proposal_count",
             runtime.nonfinite_euler_proposal_count},
            {"maximum_raw_drift_displacement_nm",
             runtime.maximum_raw_drift_displacement_nm},
            {"maximum_used_drift_displacement_nm",
             runtime.maximum_used_drift_displacement_nm},
            {"maximum_used_drift_fraction_of_radius",
             runtime.maximum_used_drift_fraction_of_radius},
            {"maximum_brownian_displacement_nm",
             runtime.maximum_brownian_displacement_nm},
            {"maximum_brownian_displacement_sigma_units",
             runtime.maximum_brownian_displacement_sigma_units},
            {"maximum_total_proposal_displacement_nm",
             runtime.maximum_total_proposal_displacement_nm},
            {"maximum_total_proposal_fraction_of_radius",
             runtime.maximum_total_proposal_fraction_of_radius},
            {"projection_active_steps", runtime.projection_active_steps},
            {"projection_converged_steps", runtime.projection_converged_steps},
            {"projection_nonconverged_steps",
             runtime.projection_nonconverged_steps},
            {"projection_iteration_limit_steps",
             runtime.projection_iteration_limit_steps},
            {"projection_pair_events", runtime.projection_pair_events},
            {"projection_corrected_particle_events",
             runtime.projection_corrected_particle_events},
            {"projection_iteration_count", runtime.projection_iteration_count},
            {"maximum_projection_iterations_used",
             runtime.max_projection_iterations_used},
            {"maximum_projection_vesicle_neighbor_cell_reach",
             runtime.max_projection_vesicle_neighbor_cell_reach},
            {"maximum_projection_excess_overlap_nm",
             runtime.max_projection_excess_overlap_nm},
            {"projection_residual_pair_events",
             runtime.projection_residual_pair_events},
            {"maximum_projection_residual_excess_nm",
             runtime.max_projection_residual_excess_nm},
            {"maximum_projection_vesicle_center_cutoff_nm",
             runtime.max_projection_vesicle_center_cutoff_nm},
            {"boundary_position_update_calls", boundary.position_update_calls},
            {"boundary_reflected_update_count", boundary.reflected_update_count},
            {"boundary_reflected_axis_count", boundary.reflected_axis_count},
            {"boundary_rejected_update_count", boundary.rejected_update_count},
            {"nonfinite_position_update_count",
             boundary.nonfinite_position_update_count},
            {"maximum_requested_displacement_nm",
             boundary.maximum_requested_displacement_nm},
            {"maximum_boundary_proposal_correction_nm",
             boundary.maximum_proposal_correction_nm},
            {"sk_rock_macro_steps", runtime.sk_rock_macro_steps},
            {"sk_rock_force_evaluations",
             runtime.sk_rock_force_evaluations},
            {"sk_rock_internal_reflected_updates",
             runtime.sk_rock_internal_reflected_updates},
            {"sk_rock_max_internal_boundary_correction_nm",
             runtime.sk_rock_max_internal_boundary_correction_nm},
            {"srock2_macro_steps", runtime.srock2_macro_steps},
            {"srock2_force_evaluations",
             runtime.srock2_force_evaluations},
            {"srock2_internal_reflected_updates",
             runtime.srock2_internal_reflected_updates},
            {"srock2_max_internal_boundary_correction_nm",
             runtime.srock2_max_internal_boundary_correction_nm}
    };
    const fs::path diagnostics_path =
            fs::path(filename).parent_path() / "numerical_diagnostics.json";
    std::ofstream diagnostics_output(diagnostics_path, std::ios::trunc);
    if (!diagnostics_output.is_open()) {
        throw std::runtime_error(
                "Could not write numerical diagnostics sidecar");
    }
    diagnostics_output << numerical_diagnostics.dump(2) << '\n';
    diagnostics_output.flush();
    if (!diagnostics_output.good()) {
        throw std::runtime_error(
                "Could not finalize numerical diagnostics sidecar");
    }
}

std::filesystem::path Saver::resolveResumeSource(
        const std::string &configured_source,
        const std::filesystem::path &project_root) {
    if (configured_source.empty()) {
        throw std::invalid_argument(
                "total.resume_source is empty while total.continue is true");
    }

    std::string portable_source = configured_source;
    std::replace(portable_source.begin(), portable_source.end(), '\\', '/');

    const fs::path configured_path =
            fs::path(portable_source).lexically_normal();

    std::vector<fs::path> candidates;
    if (configured_path.is_absolute()) {
        candidates.push_back(configured_path);
    } else {
        candidates.push_back(project_root / configured_path);
        candidates.push_back(project_root / "builds" / configured_path);

        fs::path ancestor = fs::current_path();
        while (!ancestor.empty()) {
            candidates.push_back(ancestor / configured_path);
            const fs::path parent = ancestor.parent_path();
            if (parent.empty() || parent == ancestor) break;
            ancestor = parent;
        }
    }

    for (const fs::path &candidate: candidates) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && !ec) {
            return fs::weakly_canonical(candidate, ec);
        }
    }

    throw std::runtime_error(
            "Resume source was not found. Use a repository-relative path "
            "such as builds/run-name so the Dashboard is portable.");
}

Saver::ResumeCheckpoint Saver::loadLatestResumeCheckpoint(
        const std::filesystem::path &source) {
    std::vector<std::pair<int, fs::path>> files;
    if (fs::is_regular_file(source)) {
        const int step = checkpointStepFromFilename(source);
        if (step < 0) {
            throw std::runtime_error(
                    "resume_source file is not a resume_checkpoint_step_*.json file");
        }
        files.emplace_back(step, source);
    } else if (fs::is_directory(source)) {
        files = checkpointFiles(source);
    } else {
        throw std::runtime_error(
                "resume_source is neither a checkpoint file nor a directory");
    }

    if (files.empty()) {
        throw std::runtime_error(
                "No complete resume checkpoint was found in resume_source. "
                "Runs created before the resume edit cannot be resumed exactly.");
    }

    std::string last_error;
    for (const auto &[filename_step, path]: files) {
        try {
            std::ifstream input(path, std::ios::binary);
            if (!input.is_open()) {
                throw std::runtime_error("checkpoint could not be opened");
            }
            json j;
            input >> j;

            const int checkpoint_schema_version =
                    j.at("checkpoint_schema_version").get<int>();
            if ((checkpoint_schema_version != 1
                 && checkpoint_schema_version
                    != RESUME_CHECKPOINT_SCHEMA_VERSION)
                || !j.at("checkpoint_write_complete").get<bool>()) {
                throw std::runtime_error(
                        "checkpoint schema or completion marker is invalid");
            }

            ResumeCheckpoint checkpoint;
            checkpoint.simulation_step = j.at("simulation_step").get<int>();
            checkpoint.physical_time_ms =
                    j.at("physical_time_ms").get<double>();
            checkpoint.actual_seed = j.at("actual_seed").get<std::uint32_t>();
            checkpoint.checkpoint_file = path;
            checkpoint.configuration = j.at("configuration");
            checkpoint.rng_state = j.at("rng_state_mt19937").get<std::string>();

            if (checkpoint.simulation_step != filename_step
                || checkpoint.simulation_step < 0
                || checkpoint.rng_state.empty()) {
                throw std::runtime_error(
                        "checkpoint step or random-generator state is invalid");
            }

            const json &syn = j.at("synapsin");
            checkpoint.synapsin_locations =
                    syn.at("locations").get<
                            std::vector<std::array<float, 3>>>();
            checkpoint.synapsin_positions =
                    syn.at("positions").get<
                            std::vector<std::array<int, 3>>>();
            checkpoint.which_synapsin_where =
                    syn.at("which_synapsin_where").get<
                            std::vector<std::vector<int>>>();
            checkpoint.synapsin_radius = syn.at("radius").get<float>();
            checkpoint.synapsin_diffusion_coefficient =
                    syn.at("diffusion_coefficient").get<float>();
            checkpoint.synapsin_drag = syn.at("drag").get<float>();

            const json &ves = j.at("vesicles");
            checkpoint.vesicle_locations =
                    ves.at("locations").get<
                            std::vector<std::array<float, 3>>>();
            checkpoint.vesicle_positions =
                    ves.at("positions").get<
                            std::vector<std::array<int, 3>>>();
            checkpoint.which_vesicle_where =
                    ves.at("which_vesicle_where").get<
                            std::vector<std::vector<int>>>();
            checkpoint.vesicle_radii =
                    ves.at("radii").get<std::vector<float>>();
            checkpoint.vesicle_core_radii = ves.contains("core_radii")
                    ? ves.at("core_radii").get<std::vector<float>>()
                    : checkpoint.vesicle_radii;
            checkpoint.vesicle_diffusion_coefficient =
                    ves.at("diffusion_coefficient").get<std::vector<float>>();
            checkpoint.vesicle_drag =
                    ves.at("drag").get<std::vector<float>>();
            checkpoint.vesicle_volume =
                    ves.at("volume").get<std::vector<float>>();
            checkpoint.vesicle_active =
                    ves.at("active").get<std::vector<bool>>();

            return checkpoint;
        } catch (const std::exception &error) {
            last_error = path.filename().string() + ": " + error.what();
        }
    }

    throw std::runtime_error(
            "All resume checkpoints were invalid; newest error: " + last_error);
}

bool Saver::validateResumeCompatibility(
        const ResumeCheckpoint &checkpoint,
        const std::string &resume_mode) {
    const json current = currentResumeConfiguration();
    if (!checkpoint.configuration.contains("state_compatibility")
        || !checkpoint.configuration.contains("protocol_parameters")) {
        throw std::runtime_error(
                "Checkpoint configuration snapshot is incomplete");
    }

    if (!jsonNumbersClose(
            checkpoint.configuration.at("state_compatibility"),
            current.at("state_compatibility"))) {
        throw std::runtime_error(
                "Resume rejected: particle counts, geometry, time step, or "
                "transport parameters changed. These fields define the saved "
                "state and cannot be changed mid-trajectory.");
    }

    const bool parameters_changed = !jsonNumbersClose(
            checkpoint.configuration.at("protocol_parameters"),
            current.at("protocol_parameters"));
    if (resume_mode != "new_parameters" && parameters_changed) {
        throw std::runtime_error(
                "Resume rejected: protocol/force parameters changed but "
                "resume_mode is not new_parameters");
    }
    return parameters_changed;
}

void Saver::restoreResumeCheckpoint(
        const ResumeCheckpoint &checkpoint) {
    const std::size_t num_syn = checkpoint.synapsin_locations.size();
    const std::size_t num_ves = checkpoint.vesicle_locations.size();
    const auto sameSize = [](std::size_t expected, std::size_t actual) {
        return expected == actual;
    };

    if (num_syn != static_cast<std::size_t>(Synapsin::number)
        || !sameSize(num_syn, checkpoint.synapsin_positions.size())
        || num_ves != static_cast<std::size_t>(Vesicles::number)
        || !sameSize(num_ves, checkpoint.vesicle_positions.size())
        || !sameSize(num_ves, checkpoint.vesicle_core_radii.size())
        || !sameSize(num_ves, checkpoint.vesicle_radii.size())
        || !sameSize(num_ves, checkpoint.vesicle_diffusion_coefficient.size())
        || !sameSize(num_ves, checkpoint.vesicle_drag.size())
        || !sameSize(num_ves, checkpoint.vesicle_volume.size())
        || !sameSize(num_ves, checkpoint.vesicle_active.size())) {
        throw std::runtime_error(
                "Checkpoint particle arrays do not match configured counts");
    }

    Synapsin::locations = checkpoint.synapsin_locations;
    Synapsin::positions = checkpoint.synapsin_positions;
    Synapsin::which_synapsin_where = checkpoint.which_synapsin_where;
    Synapsin::radius = checkpoint.synapsin_radius;
    Synapsin::diffusion_coefficient =
            checkpoint.synapsin_diffusion_coefficient;
    Synapsin::drag = checkpoint.synapsin_drag;
    Synapsin::drift_scale.clear();

    Vesicles::locations = checkpoint.vesicle_locations;
    Vesicles::positions = checkpoint.vesicle_positions;
    Vesicles::which_vesicle_where = checkpoint.which_vesicle_where;
    Vesicles::core_radii = checkpoint.vesicle_core_radii;
    Vesicles::radii = checkpoint.vesicle_radii;
    Vesicles::diffusion_coefficient =
            checkpoint.vesicle_diffusion_coefficient;
    Vesicles::drag = checkpoint.vesicle_drag;
    Vesicles::volume = checkpoint.vesicle_volume;
    Vesicles::active = checkpoint.vesicle_active;
    Vesicles::drift_scale.clear();
    Interaction::last_overlap_projection_diagnostics = {};

    std::istringstream rng_input(checkpoint.rng_state);
    rng_input >> GP::gen;
    if (rng_input.fail()) {
        throw std::runtime_error(
                "Could not restore the mt19937 state from checkpoint");
    }
    GP::seed = checkpoint.actual_seed;
}

void Saver::saveResumeCheckpoint(
        int simulation_step,
        const std::string &output_filename) {
    if (simulation_step < 0) {
        throw std::invalid_argument(
                "Resume checkpoint step must be non-negative");
    }

    std::ostringstream rng_output;
    rng_output << GP::gen;
    if (rng_output.fail()) {
        throw std::runtime_error(
                "Could not serialize the mt19937 state");
    }

    json checkpoint = {
            {"checkpoint_schema_version",
             RESUME_CHECKPOINT_SCHEMA_VERSION},
            {"checkpoint_write_complete", true},
            {"simulation_step", simulation_step},
            {"physical_time_ms",
             static_cast<double>(simulation_step) * GP::deltaT},
            {"source_output_file",
             fs::path(output_filename).filename().generic_string()},
            {"source_run_directory", ProjectManager::folder_name},
            {"actual_seed", GP::seed},
            {"rng_engine", "std::mt19937"},
            {"rng_state_mt19937", rng_output.str()},
            {"configuration", currentResumeConfiguration()},
            {"synapsin", {
                    {"locations", Synapsin::locations},
                    {"positions", Synapsin::positions},
                    {"which_synapsin_where", Synapsin::which_synapsin_where},
                    {"radius", Synapsin::radius},
                    {"diffusion_coefficient", Synapsin::diffusion_coefficient},
                    {"drag", Synapsin::drag}
            }},
            {"vesicles", {
                    {"locations", Vesicles::locations},
                    {"positions", Vesicles::positions},
                    {"which_vesicle_where", Vesicles::which_vesicle_where},
                    {"core_radii", Vesicles::core_radii},
                    {"radii", Vesicles::radii},
                    {"diffusion_coefficient", Vesicles::diffusion_coefficient},
                    {"drag", Vesicles::drag},
                    {"volume", Vesicles::volume},
                    {"active", Vesicles::active}
            }}
    };

    const fs::path final_path = fs::current_path()
            / (CHECKPOINT_PREFIX + std::to_string(simulation_step)
               + CHECKPOINT_SUFFIX);
    fs::path temporary_path = final_path;
    temporary_path += ".tmp";

    {
        std::ofstream output(temporary_path,
                             std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            throw std::runtime_error(
                    "Could not create temporary resume checkpoint");
        }
        output << checkpoint.dump();
        output.flush();
        if (!output.good()) {
            throw std::runtime_error(
                    "Could not finish writing temporary resume checkpoint");
        }
    }

    std::error_code rename_error;
    fs::rename(temporary_path, final_path, rename_error);
    if (rename_error) {
        throw std::runtime_error(
                "Could not publish resume checkpoint: "
                + rename_error.message());
    }

    const auto files = checkpointFiles(fs::current_path());
    for (std::size_t i = RESUME_CHECKPOINTS_TO_KEEP; i < files.size(); ++i) {
        std::error_code remove_error;
        fs::remove(files[i].second, remove_error);
        if (remove_error) {
            std::cerr << "Warning: old resume checkpoint was not removed: "
                      << files[i].second << " ("
                      << remove_error.message() << ")\n";
        }
    }
}

void Saver::saveMinimizerDiagnosticsToHDF5(
        const std::string &filename,
        const std::vector<std::array<float, 3>> &synapsin_before,
        const std::vector<std::array<float, 3>> &vesicle_before,
        const std::vector<std::array<float, 3>> &synapsin_after,
        const std::vector<std::array<float, 3>> &vesicle_after,
        const Interaction::OverlapDiagnostics &before,
        const Interaction::OverlapDiagnostics &after,
        const std::vector<Interaction::OverlapDiagnostics> &history,
        int iterations_requested,
        int iterations_completed,
        double overlap_tolerance_nm,
        int consecutive_checks_required,
        int consecutive_checks_achieved,
        int stop_reason_code,
        bool converged) {
    using namespace HighFive;

    if (synapsin_before.size() != synapsin_after.size()
        || vesicle_before.size() != vesicle_after.size()
        || vesicle_before.size() != Vesicles::radii.size()) {
        throw std::runtime_error(
                "Minimizer diagnostic particle counts are inconsistent");
    }
    if (iterations_requested < 0
        || iterations_completed < 0
        || iterations_completed > iterations_requested
        || history.size()
           != static_cast<std::size_t>(iterations_completed) + 1) {
        throw std::runtime_error(
                "Minimizer diagnostic iteration history is inconsistent");
    }
    if (!std::isfinite(overlap_tolerance_nm)
        || overlap_tolerance_nm < 0.0
        || consecutive_checks_required < 1
        || consecutive_checks_achieved < 0) {
        throw std::runtime_error(
                "Minimizer convergence settings are invalid");
    }
    if (stop_reason_code < 0 || stop_reason_code > 2
        || converged != (stop_reason_code != 0)) {
        throw std::runtime_error(
                "Minimizer convergence status is inconsistent");
    }

    const auto sameDiagnostics = [](
            const Interaction::OverlapDiagnostics &left,
            const Interaction::OverlapDiagnostics &right) {
        return left.overlapping_pair_count == right.overlapping_pair_count
               && left.synapsin_synapsin_pair_count
                  == right.synapsin_synapsin_pair_count
               && left.synapsin_vesicle_pair_count
                  == right.synapsin_vesicle_pair_count
               && left.vesicle_vesicle_pair_count
                  == right.vesicle_vesicle_pair_count
               && left.overlapping_particle_count
                  == right.overlapping_particle_count
               && left.overlapping_synapsin_count
                  == right.overlapping_synapsin_count
               && left.overlapping_vesicle_count
                  == right.overlapping_vesicle_count
               && left.overlap_sum_nm == right.overlap_sum_nm
               && left.overlap_mean_nm == right.overlap_mean_nm
               && left.overlap_max_nm == right.overlap_max_nm;
    };

    if (history.empty()
        || !sameDiagnostics(before, history.front())
        || !sameDiagnostics(after, history.back())) {
        throw std::runtime_error(
                "Minimizer before/after diagnostics do not match history");
    }

    File file(filename, File::OpenOrCreate);
    Group group = file.createGroup("minimizer_diagnostics");

    const std::size_t num_synapsin = synapsin_before.size();
    const std::size_t num_vesicle = vesicle_before.size();
    const std::size_t total = num_synapsin + num_vesicle;

    DataSetCreateProps props_1d;
    props_1d.add(Chunking(std::vector<hsize_t>{total}));
    props_1d.add(Shuffle());
    props_1d.add(Deflate(6));

    DataSetCreateProps props_2d;
    props_2d.add(Chunking(std::vector<hsize_t>{total, 3}));
    props_2d.add(Shuffle());
    props_2d.add(Deflate(6));

    DataSetCreateProps props_scalar;
    props_scalar.add(Chunking(std::vector<hsize_t>{1}));
    props_scalar.add(Shuffle());
    props_scalar.add(Deflate(6));

    DataSetCreateProps props_history;
    props_history.add(Chunking(
            std::vector<hsize_t>{history.size()}));
    props_history.add(Shuffle());
    props_history.add(Deflate(6));

    std::vector<int> ids(total), types(total);
    std::vector<float> radii(total), core_radii(total);
    std::vector<std::vector<float>> positions_before(
            total, std::vector<float>(3));
    std::vector<std::vector<float>> positions_after(
            total, std::vector<float>(3));

    std::vector<std::int64_t> history_iteration(history.size());
    std::vector<std::int64_t> history_overlapping_pair_count(history.size());
    std::vector<std::int64_t> history_overlapping_particle_count(history.size());
    std::vector<std::int64_t> history_overlapping_synapsin_count(history.size());
    std::vector<std::int64_t> history_overlapping_vesicle_count(history.size());
    std::vector<std::int64_t> history_ss_pair_count(history.size());
    std::vector<std::int64_t> history_sv_pair_count(history.size());
    std::vector<std::int64_t> history_vv_pair_count(history.size());
    std::vector<double> history_overlap_sum_nm(history.size());
    std::vector<double> history_overlap_mean_nm(history.size());
    std::vector<double> history_overlap_max_nm(history.size());

    for (std::size_t iteration = 0; iteration < history.size(); ++iteration) {
        const auto &diagnostics = history[iteration];
        history_iteration[iteration] =
                static_cast<std::int64_t>(iteration);
        history_overlapping_pair_count[iteration] =
                diagnostics.overlapping_pair_count;
        history_overlapping_particle_count[iteration] =
                diagnostics.overlapping_particle_count;
        history_overlapping_synapsin_count[iteration] =
                diagnostics.overlapping_synapsin_count;
        history_overlapping_vesicle_count[iteration] =
                diagnostics.overlapping_vesicle_count;
        history_ss_pair_count[iteration] =
                diagnostics.synapsin_synapsin_pair_count;
        history_sv_pair_count[iteration] =
                diagnostics.synapsin_vesicle_pair_count;
        history_vv_pair_count[iteration] =
                diagnostics.vesicle_vesicle_pair_count;
        history_overlap_sum_nm[iteration] = diagnostics.overlap_sum_nm;
        history_overlap_mean_nm[iteration] = diagnostics.overlap_mean_nm;
        history_overlap_max_nm[iteration] = diagnostics.overlap_max_nm;
    }

    for (std::size_t i = 0; i < num_synapsin; ++i) {
        ids[i] = static_cast<int>(i) + 1;
        types[i] = 1;
        radii[i] = Synapsin::radius;
        core_radii[i] = Synapsin::radius;
        positions_before[i] = {
                synapsin_before[i][0],
                synapsin_before[i][1],
                synapsin_before[i][2]
        };
        positions_after[i] = {
                synapsin_after[i][0],
                synapsin_after[i][1],
                synapsin_after[i][2]
        };
    }

    for (std::size_t i = 0; i < num_vesicle; ++i) {
        const std::size_t index = num_synapsin + i;
        ids[index] = static_cast<int>(index) + 1;
        types[index] = 2;
        radii[index] = Vesicles::radii[i];
        core_radii[index] = Vesicles::core_radii[i];
        positions_before[index] = {
                vesicle_before[i][0],
                vesicle_before[i][1],
                vesicle_before[i][2]
        };
        positions_after[index] = {
                vesicle_after[i][0],
                vesicle_after[i][1],
                vesicle_after[i][2]
        };
    }

    const auto int64Value = [](std::int64_t value) {
        return std::vector<std::int64_t>{value};
    };
    const auto doubleValue = [](double value) {
        return std::vector<double>{value};
    };

    group.createDataSet("ids", ids, props_1d);
    group.createDataSet("types", types, props_1d);
    group.createDataSet("radii", radii, props_1d);
    group.createDataSet("core_radii", core_radii, props_1d);
    group.createDataSet("positions_before", positions_before, props_2d);
    group.createDataSet("positions_after", positions_after, props_2d);

    group.createDataSet(
            "iterations_requested",
            int64Value(iterations_requested),
            props_scalar);
    group.createDataSet(
            "iterations_completed",
            int64Value(iterations_completed),
            props_scalar);
    group.createDataSet(
            "overlap_tolerance_nm",
            doubleValue(overlap_tolerance_nm),
            props_scalar);
    group.createDataSet(
            "consecutive_checks_required",
            int64Value(consecutive_checks_required),
            props_scalar);
    group.createDataSet(
            "consecutive_checks_achieved",
            int64Value(consecutive_checks_achieved),
            props_scalar);
    group.createDataSet(
            "stop_reason_code",
            int64Value(stop_reason_code),
            props_scalar);
    group.createDataSet(
            "converged",
            int64Value(converged ? 1 : 0),
            props_scalar);

    group.createDataSet(
            "overlapping_pair_count_before",
            int64Value(before.overlapping_pair_count),
            props_scalar);
    group.createDataSet(
            "overlapping_pair_count_after",
            int64Value(after.overlapping_pair_count),
            props_scalar);
    group.createDataSet(
            "overlapping_particle_count_before",
            int64Value(before.overlapping_particle_count),
            props_scalar);
    group.createDataSet(
            "overlapping_particle_count_after",
            int64Value(after.overlapping_particle_count),
            props_scalar);
    group.createDataSet(
            "overlapping_synapsin_count_before",
            int64Value(before.overlapping_synapsin_count),
            props_scalar);
    group.createDataSet(
            "overlapping_synapsin_count_after",
            int64Value(after.overlapping_synapsin_count),
            props_scalar);
    group.createDataSet(
            "overlapping_vesicle_count_before",
            int64Value(before.overlapping_vesicle_count),
            props_scalar);
    group.createDataSet(
            "overlapping_vesicle_count_after",
            int64Value(after.overlapping_vesicle_count),
            props_scalar);

    group.createDataSet(
            "ss_pair_count_before",
            int64Value(before.synapsin_synapsin_pair_count),
            props_scalar);
    group.createDataSet(
            "ss_pair_count_after",
            int64Value(after.synapsin_synapsin_pair_count),
            props_scalar);
    group.createDataSet(
            "sv_pair_count_before",
            int64Value(before.synapsin_vesicle_pair_count),
            props_scalar);
    group.createDataSet(
            "sv_pair_count_after",
            int64Value(after.synapsin_vesicle_pair_count),
            props_scalar);
    group.createDataSet(
            "vv_pair_count_before",
            int64Value(before.vesicle_vesicle_pair_count),
            props_scalar);
    group.createDataSet(
            "vv_pair_count_after",
            int64Value(after.vesicle_vesicle_pair_count),
            props_scalar);

    group.createDataSet(
            "overlap_sum_nm_before",
            doubleValue(before.overlap_sum_nm),
            props_scalar);
    group.createDataSet(
            "overlap_sum_nm_after",
            doubleValue(after.overlap_sum_nm),
            props_scalar);
    group.createDataSet(
            "overlap_mean_nm_before",
            doubleValue(before.overlap_mean_nm),
            props_scalar);
    group.createDataSet(
            "overlap_mean_nm_after",
            doubleValue(after.overlap_mean_nm),
            props_scalar);
    group.createDataSet(
            "overlap_max_nm_before",
            doubleValue(before.overlap_max_nm),
            props_scalar);
    group.createDataSet(
            "overlap_max_nm_after",
            doubleValue(after.overlap_max_nm),
            props_scalar);

    group.createDataSet(
            "history_iteration",
            history_iteration,
            props_history);
    group.createDataSet(
            "history_overlapping_pair_count",
            history_overlapping_pair_count,
            props_history);
    group.createDataSet(
            "history_overlapping_particle_count",
            history_overlapping_particle_count,
            props_history);
    group.createDataSet(
            "history_overlapping_synapsin_count",
            history_overlapping_synapsin_count,
            props_history);
    group.createDataSet(
            "history_overlapping_vesicle_count",
            history_overlapping_vesicle_count,
            props_history);
    group.createDataSet(
            "history_ss_pair_count",
            history_ss_pair_count,
            props_history);
    group.createDataSet(
            "history_sv_pair_count",
            history_sv_pair_count,
            props_history);
    group.createDataSet(
            "history_vv_pair_count",
            history_vv_pair_count,
            props_history);
    group.createDataSet(
            "history_overlap_sum_nm",
            history_overlap_sum_nm,
            props_history);
    group.createDataSet(
            "history_overlap_mean_nm",
            history_overlap_mean_nm,
            props_history);
    group.createDataSet(
            "history_overlap_max_nm",
            history_overlap_max_nm,
            props_history);
}

void Saver::saveRawDataToHDF5(const std::string &filename, int t) {
    using namespace HighFive;

    File file(filename, File::OpenOrCreate);

    std::string group_name = "step_" + std::to_string(t);
    Group step_group = file.createGroup(group_name);

    size_t numSyn = Synapsin::number;
    size_t numVes = Vesicles::number;
    size_t total = numSyn + numVes;

    DataSetCreateProps props_1d;
    props_1d.add(Chunking(std::vector<hsize_t>{total}));
    props_1d.add(Shuffle());
    props_1d.add(Deflate(6));

    DataSetCreateProps props_2d;
    props_2d.add(Chunking(std::vector<hsize_t>{total, 3}));
    props_2d.add(Shuffle());
    props_2d.add(Deflate(6));

    DataSetCreateProps props_scalar;
    props_scalar.add(Chunking(std::vector<hsize_t>{1}));
    props_scalar.add(Shuffle());
    props_scalar.add(Deflate(6));

    std::ostringstream rng_output;
    rng_output << GP::gen;
    if (rng_output.fail()) {
        throw std::runtime_error(
                "Could not serialize mt19937 state for HDF5 frame");
    }
    const std::string rng_state_text = rng_output.str();
    const std::vector<std::uint8_t> rng_state_bytes(
            rng_state_text.begin(), rng_state_text.end());
    DataSetCreateProps props_rng;
    props_rng.add(Chunking(
            std::vector<hsize_t>{rng_state_bytes.size()}));
    props_rng.add(Shuffle());
    props_rng.add(Deflate(6));

    const std::vector<std::int64_t> simulation_step{
            static_cast<std::int64_t>(t)
    };
    const std::vector<double> physical_time_ms{
            static_cast<double>(t) * static_cast<double>(GP::deltaT)
    };
    const std::vector<double> delta_t_ms{
            static_cast<double>(GP::deltaT)
    };
    const std::vector<double> frame_interval_ms{
            static_cast<double>(GP::load)
            * static_cast<double>(GP::deltaT)
    };

    std::vector<int> ids(total), types(total), n_self(total), n_cross(total);
    std::vector<std::vector<float>> positions(total, std::vector<float>(3));

    std::vector<std::vector<double>> total_force_vector(total, std::vector<double>(3));
    std::vector<float> radii(total), core_radii(total), total_f(total);
    std::vector<double> drift_scale(total, 1.0);

    if (Synapsin::drift_scale.size() != numSyn) {
        Synapsin::prepareDriftScales();
    }
    if (Vesicles::drift_scale.size() != numVes) {
        Vesicles::prepareDriftScales();
    }

    std::vector<std::vector<double>> electrostatic_force_vector;
    std::vector<std::vector<double>> van_der_waals_force_vector;
    std::vector<std::vector<double>> hydrophobic_force_vector;
    std::vector<std::vector<double>> steric_core_force_vector;

    if (GP::save_forces) {
        if (Interaction::synapsin_force_components.size() != numSyn
            || Interaction::vesicle_force_components.size() != numVes) {
            throw std::runtime_error(
                    "Force-component buffers do not match particle counts");
        }

        electrostatic_force_vector.assign(total, std::vector<double>(3));
        van_der_waals_force_vector.assign(
                total, std::vector<double>(3));
        hydrophobic_force_vector.assign(total, std::vector<double>(3));
        steric_core_force_vector.assign(total, std::vector<double>(3));
    }

    std::vector<float> s_dist_sum(total), s_dist_avg(total), s_dist_std(total), s_dist_min(total), s_dist_max(total);
    std::vector<float> c_dist_sum(total), c_dist_avg(total), c_dist_std(total), c_dist_min(total), c_dist_max(total);

    std::vector<float> s_ov_sum(total), s_ov_avg(total), s_ov_std(total), s_ov_min(total), s_ov_max(total);
    std::vector<float> c_ov_sum(total), c_ov_avg(total), c_ov_std(total), c_ov_min(total), c_ov_max(total);

    std::vector<float> s_f_sum(total), s_f_avg(total), s_f_std(total), s_f_min(total), s_f_max(total);
    std::vector<float> c_f_sum(total), c_f_avg(total), c_f_std(total), c_f_min(total), c_f_max(total);

    for (size_t i = 0; i < numSyn; ++i) {
        ids[i] = (int) i + 1;
        types[i] = 1;

        positions[i] = {Synapsin::locations[i][0], Synapsin::locations[i][1], Synapsin::locations[i][2]};
        radii[i] = Synapsin::radius;
        core_radii[i] = Synapsin::radius;
        n_self[i] = Synapsin::neighborhood_self[i];
        n_cross[i] = Synapsin::neighborhood_cross[i];
        total_force_vector[i] = {
                Synapsin::synapsin_forces[i][0],
                Synapsin::synapsin_forces[i][1],
                Synapsin::synapsin_forces[i][2]
        };
        total_f[i] = static_cast<float>(std::sqrt(
                total_force_vector[i][0] * total_force_vector[i][0]
                + total_force_vector[i][1] * total_force_vector[i][1]
                + total_force_vector[i][2] * total_force_vector[i][2]));
        drift_scale[i] = Synapsin::drift_scale[i];

        if (GP::save_forces) {
            const auto &components =
                    Interaction::synapsin_force_components[i];
            electrostatic_force_vector[i] = {
                    components.electrostatic[0],
                    components.electrostatic[1],
                    components.electrostatic[2]
            };
            van_der_waals_force_vector[i] = {
                    components.van_der_waals[0],
                    components.van_der_waals[1],
                    components.van_der_waals[2]
            };
            hydrophobic_force_vector[i] = {
                    components.hydrophobic[0],
                    components.hydrophobic[1],
                    components.hydrophobic[2]
            };
            steric_core_force_vector[i] = {
                    components.steric_core[0],
                    components.steric_core[1],
                    components.steric_core[2]
            };
        }

        s_dist_sum[i] = Synapsin::self_Ndistance[i].sum;
        s_dist_avg[i] = Synapsin::self_Ndistance[i].avg;
        s_dist_std[i] = Synapsin::self_Ndistance[i].std;
        s_dist_min[i] = Synapsin::self_Ndistance[i].min;
        s_dist_max[i] = Synapsin::self_Ndistance[i].max;

        c_dist_sum[i] = Synapsin::cross_Ndistance[i].sum;
        c_dist_avg[i] = Synapsin::cross_Ndistance[i].avg;
        c_dist_std[i] = Synapsin::cross_Ndistance[i].std;
        c_dist_min[i] = Synapsin::cross_Ndistance[i].min;
        c_dist_max[i] = Synapsin::cross_Ndistance[i].max;

        s_ov_sum[i] = Synapsin::self_Noverlaps[i].sum;
        s_ov_avg[i] = Synapsin::self_Noverlaps[i].avg;
        s_ov_std[i] = Synapsin::self_Noverlaps[i].std;
        s_ov_min[i] = Synapsin::self_Noverlaps[i].min;
        s_ov_max[i] = Synapsin::self_Noverlaps[i].max;

        c_ov_sum[i] = Synapsin::cross_Noverlaps[i].sum;
        c_ov_avg[i] = Synapsin::cross_Noverlaps[i].avg;
        c_ov_std[i] = Synapsin::cross_Noverlaps[i].std;
        c_ov_min[i] = Synapsin::cross_Noverlaps[i].min;
        c_ov_max[i] = Synapsin::cross_Noverlaps[i].max;

        s_f_sum[i] = Synapsin::self_Nforces[i].sum;
        s_f_avg[i] = Synapsin::self_Nforces[i].avg;
        s_f_std[i] = Synapsin::self_Nforces[i].std;
        s_f_min[i] = Synapsin::self_Nforces[i].min;
        s_f_max[i] = Synapsin::self_Nforces[i].max;

        c_f_sum[i] = Synapsin::cross_Nforces[i].sum;
        c_f_avg[i] = Synapsin::cross_Nforces[i].avg;
        c_f_std[i] = Synapsin::cross_Nforces[i].std;
        c_f_min[i] = Synapsin::cross_Nforces[i].min;
        c_f_max[i] = Synapsin::cross_Nforces[i].max;
    }

    for (size_t i = 0; i < numVes; ++i) {
        size_t idx = numSyn + i;

        ids[idx] = (int) idx + 1;
        types[idx] = 2;

        positions[idx] = {Vesicles::locations[i][0], Vesicles::locations[i][1], Vesicles::locations[i][2]};
        radii[idx] = Vesicles::radii[i];
        core_radii[idx] = Vesicles::core_radii[i];
        n_self[idx] = Vesicles::neighborhood_self[i];
        n_cross[idx] = Vesicles::neighborhood_cross[i];
        total_force_vector[idx] = {
                Vesicles::vesicles_forces[i][0],
                Vesicles::vesicles_forces[i][1],
                Vesicles::vesicles_forces[i][2]
        };
        total_f[idx] = static_cast<float>(std::sqrt(
                total_force_vector[idx][0] * total_force_vector[idx][0]
                + total_force_vector[idx][1] * total_force_vector[idx][1]
                + total_force_vector[idx][2] * total_force_vector[idx][2]));
        drift_scale[idx] = Vesicles::drift_scale[i];

        if (GP::save_forces) {
            const auto &components =
                    Interaction::vesicle_force_components[i];
            electrostatic_force_vector[idx] = {
                    components.electrostatic[0],
                    components.electrostatic[1],
                    components.electrostatic[2]
            };
            van_der_waals_force_vector[idx] = {
                    components.van_der_waals[0],
                    components.van_der_waals[1],
                    components.van_der_waals[2]
            };
            hydrophobic_force_vector[idx] = {
                    components.hydrophobic[0],
                    components.hydrophobic[1],
                    components.hydrophobic[2]
            };
            steric_core_force_vector[idx] = {
                    components.steric_core[0],
                    components.steric_core[1],
                    components.steric_core[2]
            };
        }

        s_dist_sum[idx] = Vesicles::self_Ndistance[i].sum;
        s_dist_avg[idx] = Vesicles::self_Ndistance[i].avg;
        s_dist_std[idx] = Vesicles::self_Ndistance[i].std;
        s_dist_min[idx] = Vesicles::self_Ndistance[i].min;
        s_dist_max[idx] = Vesicles::self_Ndistance[i].max;

        c_dist_sum[idx] = Vesicles::cross_Ndistance[i].sum;
        c_dist_avg[idx] = Vesicles::cross_Ndistance[i].avg;
        c_dist_std[idx] = Vesicles::cross_Ndistance[i].std;
        c_dist_min[idx] = Vesicles::cross_Ndistance[i].min;
        c_dist_max[idx] = Vesicles::cross_Ndistance[i].max;

        s_ov_sum[idx] = Vesicles::self_Noverlaps[i].sum;
        s_ov_avg[idx] = Vesicles::self_Noverlaps[i].avg;
        s_ov_std[idx] = Vesicles::self_Noverlaps[i].std;
        s_ov_min[idx] = Vesicles::self_Noverlaps[i].min;
        s_ov_max[idx] = Vesicles::self_Noverlaps[i].max;

        c_ov_sum[idx] = Vesicles::cross_Noverlaps[i].sum;
        c_ov_avg[idx] = Vesicles::cross_Noverlaps[i].avg;
        c_ov_std[idx] = Vesicles::cross_Noverlaps[i].std;
        c_ov_min[idx] = Vesicles::cross_Noverlaps[i].min;
        c_ov_max[idx] = Vesicles::cross_Noverlaps[i].max;

        s_f_sum[idx] = Vesicles::self_Nforces[i].sum;
        s_f_avg[idx] = Vesicles::self_Nforces[i].avg;
        s_f_std[idx] = Vesicles::self_Nforces[i].std;
        s_f_min[idx] = Vesicles::self_Nforces[i].min;
        s_f_max[idx] = Vesicles::self_Nforces[i].max;

        c_f_sum[idx] = Vesicles::cross_Nforces[i].sum;
        c_f_avg[idx] = Vesicles::cross_Nforces[i].avg;
        c_f_std[idx] = Vesicles::cross_Nforces[i].std;
        c_f_min[idx] = Vesicles::cross_Nforces[i].min;
        c_f_max[idx] = Vesicles::cross_Nforces[i].max;
    }

    step_group.createDataSet(
            "simulation_step", simulation_step, props_scalar);
    step_group.createDataSet(
            "physical_time_ms", physical_time_ms, props_scalar);
    step_group.createDataSet(
            "delta_t_ms", delta_t_ms, props_scalar);
    step_group.createDataSet(
            "frame_interval_ms", frame_interval_ms, props_scalar);

    step_group.createDataSet("ids", ids, props_1d);
    step_group.createDataSet("types", types, props_1d);
    step_group.createDataSet("positions", positions, props_2d);
    step_group.createDataSet("radii", radii, props_1d);
    step_group.createDataSet("core_radii", core_radii, props_1d);
    step_group.createDataSet("n_self", n_self, props_1d);
    step_group.createDataSet("n_cross", n_cross, props_1d);
    step_group.createDataSet("total_force_magnitude", total_f, props_1d);
    step_group.createDataSet("total_force_vector", total_force_vector, props_2d);
    step_group.createDataSet("drift_scale", drift_scale, props_1d);

    const auto drift_limited_particle_count =
            static_cast<std::int64_t>(std::count_if(
                    drift_scale.begin(), drift_scale.end(),
                    [](double scale) { return scale < 1.0 - 1e-12; }));
    const auto &projection =
            Interaction::last_overlap_projection_diagnostics;
    step_group.createDataSet(
            "drift_limited_particle_count",
            std::vector<std::int64_t>{drift_limited_particle_count},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_pair_count",
            std::vector<std::int64_t>{projection.pair_count},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_corrected_particle_count",
            std::vector<std::int64_t>{projection.corrected_particle_count},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_iterations",
            std::vector<std::int64_t>{projection.iterations_completed},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_vesicle_neighbor_cell_reach",
            std::vector<std::int64_t>{
                    projection.vesicle_neighbor_cell_reach},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_max_excess_nm",
            std::vector<double>{projection.max_excess_overlap_nm},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_residual_pair_count",
            std::vector<std::int64_t>{projection.residual_pair_count},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_residual_max_excess_nm",
            std::vector<double>{projection.residual_max_excess_overlap_nm},
            props_scalar);
    step_group.createDataSet(
            "previous_step_overlap_projection_vesicle_center_cutoff_nm",
            std::vector<double>{projection.vesicle_center_cutoff_nm},
            props_scalar);

    if (GP::save_forces) {
        step_group.createDataSet(
                "electrostatic_force_vector",
                electrostatic_force_vector,
                props_2d);
        step_group.createDataSet(
                "van_der_waals_force_vector",
                van_der_waals_force_vector,
                props_2d);
        step_group.createDataSet(
                "hydrophobic_force_vector",
                hydrophobic_force_vector,
                props_2d);
        step_group.createDataSet(
                "steric_core_force_vector",
                steric_core_force_vector,
                props_2d);
    }

    step_group.createDataSet("self_dist_sum", s_dist_sum, props_1d);
    step_group.createDataSet("cross_dist_sum", c_dist_sum, props_1d);
    step_group.createDataSet("self_dist_avg", s_dist_avg, props_1d);
    step_group.createDataSet("cross_dist_avg", c_dist_avg, props_1d);
    step_group.createDataSet("self_dist_std", s_dist_std, props_1d);
    step_group.createDataSet("cross_dist_std", c_dist_std, props_1d);
    step_group.createDataSet("self_dist_min", s_dist_min, props_1d);
    step_group.createDataSet("cross_dist_min", c_dist_min, props_1d);
    step_group.createDataSet("self_dist_max", s_dist_max, props_1d);
    step_group.createDataSet("cross_dist_max", c_dist_max, props_1d);

    step_group.createDataSet("self_overlap_sum", s_ov_sum, props_1d);
    step_group.createDataSet("cross_overlap_sum", c_ov_sum, props_1d);
    step_group.createDataSet("self_overlap_avg", s_ov_avg, props_1d);
    step_group.createDataSet("cross_overlap_avg", c_ov_avg, props_1d);
    step_group.createDataSet("self_overlap_std", s_ov_std, props_1d);
    step_group.createDataSet("cross_overlap_std", c_ov_std, props_1d);
    step_group.createDataSet("self_overlap_min", s_ov_min, props_1d);
    step_group.createDataSet("cross_overlap_min", c_ov_min, props_1d);
    step_group.createDataSet("self_overlap_max", s_ov_max, props_1d);
    step_group.createDataSet("cross_overlap_max", c_ov_max, props_1d);

    step_group.createDataSet("self_force_sum", s_f_sum, props_1d);
    step_group.createDataSet("cross_force_sum", c_f_sum, props_1d);
    step_group.createDataSet("self_force_avg", s_f_avg, props_1d);
    step_group.createDataSet("cross_force_avg", c_f_avg, props_1d);
    step_group.createDataSet("self_force_std", s_f_std, props_1d);
    step_group.createDataSet("cross_force_std", c_f_std, props_1d);
    step_group.createDataSet("self_force_min", s_f_min, props_1d);
    step_group.createDataSet("cross_force_min", c_f_min, props_1d);
    step_group.createDataSet("self_force_max", s_f_max, props_1d);
    step_group.createDataSet("cross_force_max", c_f_max, props_1d);

    step_group.createDataSet(
            "rng_state_mt19937_ascii",
            rng_state_bytes,
            props_rng);

    file.flush();
    step_group.createDataSet(
            "frame_write_complete",
            std::vector<std::int64_t>{1},
            props_scalar);
    file.flush();

    saveResumeCheckpoint(t, filename);
}

void Saver::saveAllDataToJson() {
    saveToJsonTotalData();
    saveGPDataToJson();
    saveVesicleDataToJson();
    saveSynapsinDataToJson();
    Saver::saveToJsonToEnvironment();
    Saver::saveToJsonToParameters();
    Saver::saveInteractionDataToJson();
}

void Saver::saveGPDataToJson() {

    Converter::setDashboardValue(
            "pre_test.f_off",
            GP::f_off
    );

    Converter::setDashboardValue(
            "pre_test.centering",
            GP::centering
    );

    Converter::setDashboardValue(
            "pre_test.dataAnalysis",
            GP::dataAnalysis
    );

    Converter::setDashboardValue(
            "total.actual_seed",
            GP::seed
    );

    Converter::setDashboardValue(
            "total.use_seed",
            GP::use_seed
    );

    Converter::setDashboardValue(
            "total.load",
            GP::load
    );

    Converter::setDashboardValue(
            "total.log",
            static_cast<bool>(GP::log)
    );

    Converter::setDashboardValue(
            "total.save_forces",
            GP::save_forces
    );

    Converter::setDashboardValue(
            "total.continue",
            GP::con
    );
}

void Saver::saveInteractionDataToJson() {

    Converter::setDashboardValue(
            "forces_parms.connectivity_shell_nm", GP::cut_off);
    Converter::setDashboardValue("forces_parms.adjusted_max", GP::adjusted_max);
    Converter::setDashboardValue(
            "forces_parms.dynamic_drift_limit_fraction_of_radius",
            GP::dynamic_drift_limit_fraction_of_radius);
    Converter::setDashboardValue(
            "forces_parms.enable_overlap_projection",
            GP::enable_overlap_projection);
    Converter::setDashboardValue(
            "forces_parms.overlap_projection_max_iterations",
            GP::overlap_projection_max_iterations);
    Converter::setDashboardValue(
            "forces_parms.overlap_projection_residual_tolerance_nm",
            GP::overlap_projection_residual_tolerance_nm);
    Converter::setDashboardValue(
            "forces_parms.enable_drift_component_diagnostics",
            GP::enable_drift_component_diagnostics);
    Converter::setDashboardValue(
            "forces_parms.enable_electrostatic", GP::enable_electrostatic);
    Converter::setDashboardValue(
            "forces_parms.enable_van_der_waals", GP::enable_van_der_waals);
    Converter::setDashboardValue(
            "forces_parms.enable_hydrophobic", GP::enable_hydrophobic);
    Converter::setDashboardValue(
            "forces_parms.continuous_contact_forces",
            GP::continuous_contact_forces);
    Converter::setDashboardValue(
            "forces_parms.core_allowed_overlap_nm",
            GP::core_allowed_overlap_nm);
    Converter::setDashboardValue(
            "forces_parms.core_energy_kbt", GP::core_energy_kbt);
    Converter::setDashboardValue(
            "forces_parms.enable_dynamic_steric_penalty",
            GP::enable_dynamic_steric_penalty);
    Converter::setDashboardValue(
            "forces_parms.minimum_surface_separation_nm",
            GP::minimum_surface_separation_nm);
    Converter::setDashboardValue(
            "forces_parms.hydrophobic_decay_length_nm",
            GP::hydrophobic_decay_length_nm);
    Converter::setDashboardValue(
            "forces_parms.hydrophobic_energy_syn_syn_kbt",
            GP::hydrophobic_energy_syn_syn_kbt);
    Converter::setDashboardValue(
            "forces_parms.hydrophobic_energy_syn_ves_kbt",
            GP::hydrophobic_energy_syn_ves_kbt);
    Converter::setDashboardValue(
            "forces_parms.hydrophobic_energy_ves_ves_kbt",
            GP::hydrophobic_energy_ves_ves_kbt);
}

void Saver::saveVesicleDataToJson() {
    Converter::setDashboardValue("vesicles.diffusion_coefficient", Vesicles::diffusion_coefficient);
    Converter::setDashboardValue("vesicles.drag", Vesicles::drag);
    Converter::setDashboardValue("vesicles.radii", Vesicles::radii);
    Converter::setDashboardValue("vesicles.core_radii", Vesicles::core_radii);
    Converter::setDashboardValue("vesicles.which_vesicle_where", Vesicles::which_vesicle_where);
    Converter::setDashboardValue("vesicles.std_radius", GP::std_radius);
    Converter::setDashboardValue("vesicles.mean_radius", GP::mean_radius);
    Converter::setDashboardValue(
            "vesicles.std_core_radius_nm", GP::std_radius);
    Converter::setDashboardValue(
            "vesicles.mean_core_radius_nm", GP::mean_radius);
    Converter::setDashboardValue(
            "vesicles.core_radius_min_nm", GP::vesicle_core_radius_min_nm);
    Converter::setDashboardValue(
            "vesicles.core_radius_max_nm", GP::vesicle_core_radius_max_nm);
    Converter::setDashboardValue(
            "vesicles.corona_thickness_nm", GP::vesicle_corona_thickness_nm);
    Converter::setDashboardValue("vesicles.num", GP::number_of_vesicles);
    Converter::setDashboardValue(
            "vesicles.surface_potential_mV",
            GP::vesicle_surface_potential_mV);
    Converter::setDashboardValue("vesicles.drag_ves", GP::drag_ves);
    Converter::setDashboardValue("vesicles.D_ves", GP::D_ves);
    Converter::setDashboardValue("vesicles.D_ves_C", Vesicles::D_ves_C);
    Converter::setDashboardValue("vesicles.drag_ves_C", Vesicles::drag_ves_C);

}

void Saver::saveSynapsinDataToJson() {
    Converter::setDashboardValue("synapsinI.diffusion_coefficient", Synapsin::diffusion_coefficient);
    Converter::setDashboardValue("synapsinI.drag", Synapsin::drag);
    Converter::setDashboardValue("synapsinI.which_vesicle_where", Synapsin::which_synapsin_where);
    Converter::setDashboardValue(
            "synapsinI.surface_potential_mV",
            GP::synapsin_surface_potential_mV);
    Converter::setDashboardValue("synapsinI.radius", Synapsin::radius);
    Converter::setDashboardValue("synapsinI.drag_syn", GP::drag_syn);
    Converter::setDashboardValue("synapsinI.D_syn", GP::D_syn);
    Converter::setDashboardValue("synapsinI.num", GP::number_of_SynapsinI);

}

void Saver::saveToJsonTotalData() {
    Converter::setDashboardValue("systems.build_folder_name", GP::build_folder_name);
    Converter::setDashboardValue("systems.GP::new_directory", GP::new_directory);
    Converter::setDashboardValue("systems.system_info", getSystemInfo());
    Converter::setDashboardValue("systems.execution_time", getCurrentTime());
    Converter::setDashboardValue("times.run_time", GP::total_seconds);
    Converter::setDashboardValue("times.hours", GP::hours);
    Converter::setDashboardValue("times.minutes", GP::minutes);
    Converter::setDashboardValue("times.seconds", GP::seconds);
    Converter::setDashboardValue("times.time_step", GP::deltaT);
    Converter::setDashboardValue("times.integrator", GP::integrator);
    Converter::setDashboardValue(
            "times.sk_rock_stages", GP::sk_rock_stages);
    Converter::setDashboardValue(
            "times.sk_rock_damping", GP::sk_rock_damping);
    Converter::setDashboardValue(
            "times.srock2_degree", GP::srock2_degree);
    Converter::setDashboardValue("times.Ttime", GP::Total_time);
    Converter::setDashboardValue("times.loop_minimiz", GP::loop_minimiz);
    Converter::setDashboardValue(
            "times.minimizer_overlap_tolerance_nm",
            GP::minimizer_overlap_tolerance_nm);
    Converter::setDashboardValue(
            "times.minimizer_consecutive_checks",
            GP::minimizer_consecutive_checks);
}

void Saver::saveToJsonToEnvironment() {
    Converter::setDashboardValue(
            "environment.shape",
            static_cast<int>(Environment::shape)
    );

    Converter::setDashboardValue(
            "environment.Neibor",
            static_cast<int>(Environment::nei)
    );

    Converter::setDashboardValue(
            "environment.levelNei",
            static_cast<int>(Environment::neiLev)
    );

    Converter::setDashboardValue(
            "environment.length",
            GP::length
    );

    Converter::setDashboardValue(
            "environment.width",
            GP::width
    );

    Converter::setDashboardValue(
            "environment.height",
            GP::height
    );

    Converter::setDashboardValue(
            "environment.volume",
            GP::volume
    );

    Converter::setDashboardValue(
            "environment.steps",
            GP::steps
    );

    Converter::setDashboardValue(
            "environment.out",
            GP::out
    );

    Converter::setDashboardValue(
            "environment.slide",
            GP::slide
    );

    Converter::setDashboardValue(
            "environment.scale",
            GP::scale
    );

    Converter::setDashboardValue(
            "environment.box_fract",
            GP::box_fract
    );

    Converter::setDashboardValue(
            "environment.size_Synapse",
            std::vector<int>{
                    GP::x_Synapse,
                    GP::y_Synapse,
                    GP::z_Synapse
            }
    );

    Converter::setDashboardValue(
            "environment.xyz_size",
            std::vector<int>{
                    GP::x_size,
                    GP::y_size,
                    GP::z_size
            }
    );
}

void Saver::saveToJsonToParameters() {
    Converter::setDashboardValue("parameters.temperature", GP::T);
    Converter::setDashboardValue("parameters.eta", GP::eta);
    Converter::setDashboardValue("parameters.hamaker", GP::A);
    Converter::setDashboardValue(
            "parameters.solvent_relative_permittivity", GP::eSolvent);
    Converter::setDashboardValue(
            "parameters.ionic_strength_molar", GP::ionic_strength_molar);
    Converter::setDashboardValue(
            "parameters.debye_length_nm", GP::debye_length_nm);
    Converter::setDashboardValue(
            "parameters.kappa_nm_inverse", GP::kapa);

}

nlohmann::json Saver::getSystemInfo() {
    nlohmann::json system_info;

#if defined(_WIN32)
    system_info["os"] = "Windows";
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    system_info["cores"] = si.dwNumberOfProcessors;
#elif defined(__linux__)
    struct utsname uname_data;
    uname(&uname_data);
    system_info["os"] = "Linux";
    system_info["node_name"] = uname_data.nodename;
    system_info["release"] = uname_data.release;
    system_info["version"] = uname_data.version;
    system_info["machine"] = uname_data.machine;
    system_info["cores"] = std::thread::hardware_concurrency();
#else
    system_info["os"] = "Unknown";
    system_info["cores"] = std::thread::hardware_concurrency();
#endif

    return system_info;
}

std::string Saver::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm buf;
#if defined(_WIN32)
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif

    char time_buffer[80];
    strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", &buf);
    return std::string(time_buffer);
}

void Saver::saveArrayToFile(const std::array<double, 3> &data, const std::string &filename) {
    std::ofstream outputFile(filename, std::ios::app);
    if (outputFile.is_open()) {
        for (const auto &value: data) {
            outputFile << value << " ";
        }
        outputFile << "\n";
        outputFile.close();
    } else {
        std::cerr << "Unable to open the file: " << filename << std::endl;
    }
}

void Saver::save_multi_array(const boost::multi_array<int, 3> &arr, const std::string &filename) {
    std::ofstream outfile(filename, std::ios::app);
    if (outfile.is_open()) {
        for (std::size_t i = 0; i < arr.shape()[0]; ++i) {
            for (std::size_t j = 0; j < arr.shape()[1]; ++j) {
                for (std::size_t k = 0; k < arr.shape()[2]; ++k) {
                    outfile << arr[i][j][k] << " ";
                }
                outfile << std::endl;
            }
            outfile << std::endl;
        }
        outfile.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void Saver::saveThreeDimensionalVector(const std::vector<std::vector<std::vector<int>>> &myVector,
                                       const std::string &filename) {
    std::ofstream outfile(filename, std::ios::app);
    if (outfile.is_open()) {
        for (const auto &row: myVector) {
            for (const auto &subVector: row) {
                for (const auto &element: subVector) {
                    outfile << element << " ";
                }
                outfile << "  ";
            }
            outfile << std::endl;
        }
        outfile.close();
        std::cout << "Saved to " << filename << std::endl;
    } else {
        std::cerr << "⚠ Error: Unable to open file " << filename << " for writing!" << std::endl;
    }
}

void Saver::saveVectorOfArraysToJson(const std::vector<std::array<int, 3>> &data, const std::string &filename) {
    json jsonData;
    for (const auto &arr: data) {
        jsonData.push_back({arr[0], arr[1], arr[2]});
    }

    std::ofstream outputFile(filename, std::ios::app);
    if (outputFile.is_open()) {
        outputFile << jsonData.dump(4);
        outputFile.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "⚠ Error: Unable to open file for writing!" << std::endl;
    }
}

void Saver::saveVectorToFile(const std::vector<std::vector<int>> &data, const std::string &filename) {

    std::ofstream outFile(filename, std::ios::app);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto &row: data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            MyUtility::print(row[i]);
            if (i < row.size() - 1) {
                outFile << " ";
            }
        }
        outFile << std::endl;
    }

    outFile.close();

    std::cout << "Data saved to " << filename << std::endl;
}

void Saver::saveVectorToJson(const std::vector<std::vector<int>> &data, const std::string &filename) {

    json jsonData = data;

    std::ofstream outFile(filename, std::ios::app);

    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    outFile << jsonData.dump(4);

    outFile.close();

    std::cout << "Data saved to " << filename << " in JSON format." << std::endl;
}

void Saver::saveLocationsToFile(const std::string &filename, int t) {

    std::ofstream outputFile(filename, std::ios_base::app);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing!" << std::endl;

        return;
    }

    outputFile << "ITEM: TIMESTEP\n";
    outputFile << t << "\n";
    outputFile << "ITEM: NUMBER OF ATOMS\n";
    outputFile << (Synapsin::number + Vesicles::number) << "\n";
    outputFile << "ITEM: BOX BOUNDS pp pp pp\n";
    outputFile << 150 << " " << GP::length + 150 << "\n";
    outputFile << 150 << " " << GP::width + 150 << "\n";
    outputFile << 150 << " " << GP::height + 150 << "\n";

    outputFile
            << "ITEM: ATOMS id type x y z radius color \n";

    for (int j = 0; j < Synapsin::number; ++j) {

        outputFile << j + 1 << "\t"
                   << "1\t"
                   << Synapsin::locations[j][0] << "\t"
                   << Synapsin::locations[j][1] << "\t"
                   << Synapsin::locations[j][2] << "\t"
                   << Synapsin::radius << "\t"
                   << Synapsin::color[j] << "\n";
    }

    for (int j = 0; j < Vesicles::number; ++j) {
        outputFile << j + Synapsin::number + 1 << "\t"
                   << "2\t"
                   << Vesicles::locations[j][0] << "\t"
                   << Vesicles::locations[j][1] << "\t"
                   << Vesicles::locations[j][2] << "\t"
                   << Vesicles::radii[j] << "\t"
                   << Vesicles::radii[j] << "\n";
    }

}

struct AgentData {
    int id;
    int type;

    double x, y, z;
    double radius;

    int neighborhood_self;
    int neighborhood_cross;

    float self_dist_sum, self_dist_avg, self_dist_std, self_dist_min, self_dist_max;
    float cross_dist_sum, cross_dist_avg, cross_dist_std, cross_dist_min, cross_dist_max;

    float self_overlap_sum, self_overlap_avg, self_overlap_std, self_overlap_min, self_overlap_max;
    float cross_overlap_sum, cross_overlap_avg, cross_overlap_std, cross_overlap_min, cross_overlap_max;

    float self_force_sum, self_force_avg, self_force_std, self_force_min, self_force_max;
    float cross_force_sum, cross_force_avg, cross_force_std, cross_force_min, cross_force_max;

    double total_forces;

};

static void writeStochastic(std::ofstream &f, const GP::StochasticParameters &s) {
    f.write(reinterpret_cast<const char *>(&s.sum), sizeof(s.sum));
    f.write(reinterpret_cast<const char *>(&s.avg), sizeof(s.avg));
    f.write(reinterpret_cast<const char *>(&s.std), sizeof(s.std));
    f.write(reinterpret_cast<const char *>(&s.min), sizeof(s.min));
    f.write(reinterpret_cast<const char *>(&s.max), sizeof(s.max));
}

void appendAgentData(std::ofstream &outputFile, const AgentData &d) {

    outputFile.write(reinterpret_cast<const char *>(&d.id), sizeof(d.id));
    outputFile.write(reinterpret_cast<const char *>(&d.type), sizeof(d.type));

    outputFile.write(reinterpret_cast<const char *>(&d.x), sizeof(d.x));
    outputFile.write(reinterpret_cast<const char *>(&d.y), sizeof(d.y));
    outputFile.write(reinterpret_cast<const char *>(&d.z), sizeof(d.z));
    outputFile.write(reinterpret_cast<const char *>(&d.radius), sizeof(d.radius));

    outputFile.write(reinterpret_cast<const char *>(&d.neighborhood_self), sizeof(d.neighborhood_self));
    outputFile.write(reinterpret_cast<const char *>(&d.neighborhood_cross), sizeof(d.neighborhood_cross));

    GP::StochasticParameters tmp;

    tmp = {d.self_dist_sum, d.self_dist_avg, d.self_dist_std, d.self_dist_min, d.self_dist_max};
    writeStochastic(outputFile, tmp);
    tmp = {d.cross_dist_sum, d.cross_dist_avg, d.cross_dist_std, d.cross_dist_min, d.cross_dist_max};
    writeStochastic(outputFile, tmp);

    tmp = {d.self_overlap_sum, d.self_overlap_avg, d.self_overlap_std, d.self_overlap_min, d.self_overlap_max};
    writeStochastic(outputFile, tmp);
    tmp = {d.cross_overlap_sum, d.cross_overlap_avg, d.cross_overlap_std, d.cross_overlap_min, d.cross_overlap_max};
    writeStochastic(outputFile, tmp);

    tmp = {d.self_force_sum, d.self_force_avg, d.self_force_std, d.self_force_min, d.self_force_max};
    writeStochastic(outputFile, tmp);
    tmp = {d.cross_force_sum, d.cross_force_avg, d.cross_force_std, d.cross_force_min, d.cross_force_max};
    writeStochastic(outputFile, tmp);

    outputFile.write(reinterpret_cast<const char *>(&d.total_forces), sizeof(d.total_forces));

}

static AgentData fillSynapsin(int j) {
    AgentData a{};
    a.id = j + 1;
    a.type = 1;

    a.x = Synapsin::locations[j][0];
    a.y = Synapsin::locations[j][1];
    a.z = Synapsin::locations[j][2];
    a.radius = Synapsin::radius;

    a.neighborhood_self = Synapsin::neighborhood_self[j];
    a.neighborhood_cross = Synapsin::neighborhood_cross[j];

    a.self_dist_sum = Synapsin::self_Ndistance[j].sum;
    a.self_dist_avg = Synapsin::self_Ndistance[j].avg;
    a.self_dist_std = Synapsin::self_Ndistance[j].std;
    a.self_dist_min = Synapsin::self_Ndistance[j].min;
    a.self_dist_max = Synapsin::self_Ndistance[j].max;

    a.cross_dist_sum = Synapsin::cross_Ndistance[j].sum;
    a.cross_dist_avg = Synapsin::cross_Ndistance[j].avg;
    a.cross_dist_std = Synapsin::cross_Ndistance[j].std;
    a.cross_dist_min = Synapsin::cross_Ndistance[j].min;
    a.cross_dist_max = Synapsin::cross_Ndistance[j].max;

    a.self_overlap_sum = Synapsin::self_Noverlaps[j].sum;
    a.self_overlap_avg = Synapsin::self_Noverlaps[j].avg;
    a.self_overlap_std = Synapsin::self_Noverlaps[j].std;
    a.self_overlap_min = Synapsin::self_Noverlaps[j].min;
    a.self_overlap_max = Synapsin::self_Noverlaps[j].max;

    a.cross_overlap_sum = Synapsin::cross_Noverlaps[j].sum;
    a.cross_overlap_avg = Synapsin::cross_Noverlaps[j].avg;
    a.cross_overlap_std = Synapsin::cross_Noverlaps[j].std;
    a.cross_overlap_min = Synapsin::cross_Noverlaps[j].min;
    a.cross_overlap_max = Synapsin::cross_Noverlaps[j].max;

    a.self_force_sum = Synapsin::self_Nforces[j].sum;
    a.self_force_avg = Synapsin::self_Nforces[j].avg;
    a.self_force_std = Synapsin::self_Nforces[j].std;
    a.self_force_min = Synapsin::self_Nforces[j].min;
    a.self_force_max = Synapsin::self_Nforces[j].max;

    a.cross_force_sum = Synapsin::cross_Nforces[j].sum;
    a.cross_force_avg = Synapsin::cross_Nforces[j].avg;
    a.cross_force_std = Synapsin::cross_Nforces[j].std;
    a.cross_force_min = Synapsin::cross_Nforces[j].min;
    a.cross_force_max = Synapsin::cross_Nforces[j].max;

    a.total_forces = Synapsin::total_forces[j];

    return a;
}

static AgentData fillVesicle(int j) {
    AgentData a{};
    a.id = j + Synapsin::number + 1;
    a.type = 2;

    a.x = Vesicles::locations[j][0];
    a.y = Vesicles::locations[j][1];
    a.z = Vesicles::locations[j][2];
    a.radius = Vesicles::radii[j];

    a.neighborhood_self = Vesicles::neighborhood_self[j];
    a.neighborhood_cross = Vesicles::neighborhood_cross[j];

    a.self_dist_sum = Vesicles::self_Ndistance[j].sum;
    a.self_dist_avg = Vesicles::self_Ndistance[j].avg;
    a.self_dist_std = Vesicles::self_Ndistance[j].std;
    a.self_dist_min = Vesicles::self_Ndistance[j].min;
    a.self_dist_max = Vesicles::self_Ndistance[j].max;

    a.cross_dist_sum = Vesicles::cross_Ndistance[j].sum;
    a.cross_dist_avg = Vesicles::cross_Ndistance[j].avg;
    a.cross_dist_std = Vesicles::cross_Ndistance[j].std;
    a.cross_dist_min = Vesicles::cross_Ndistance[j].min;
    a.cross_dist_max = Vesicles::cross_Ndistance[j].max;

    a.self_overlap_sum = Vesicles::self_Noverlaps[j].sum;
    a.self_overlap_avg = Vesicles::self_Noverlaps[j].avg;
    a.self_overlap_std = Vesicles::self_Noverlaps[j].std;
    a.self_overlap_min = Vesicles::self_Noverlaps[j].min;
    a.self_overlap_max = Vesicles::self_Noverlaps[j].max;

    a.cross_overlap_sum = Vesicles::cross_Noverlaps[j].sum;
    a.cross_overlap_avg = Vesicles::cross_Noverlaps[j].avg;
    a.cross_overlap_std = Vesicles::cross_Noverlaps[j].std;
    a.cross_overlap_min = Vesicles::cross_Noverlaps[j].min;
    a.cross_overlap_max = Vesicles::cross_Noverlaps[j].max;

    a.self_force_sum = Vesicles::self_Nforces[j].sum;
    a.self_force_avg = Vesicles::self_Nforces[j].avg;
    a.self_force_std = Vesicles::self_Nforces[j].std;
    a.self_force_min = Vesicles::self_Nforces[j].min;
    a.self_force_max = Vesicles::self_Nforces[j].max;

    a.cross_force_sum = Vesicles::cross_Nforces[j].sum;
    a.cross_force_avg = Vesicles::cross_Nforces[j].avg;
    a.cross_force_std = Vesicles::cross_Nforces[j].std;
    a.cross_force_min = Vesicles::cross_Nforces[j].min;
    a.cross_force_max = Vesicles::cross_Nforces[j].max;

    a.total_forces = Vesicles::total_forces[j];

    return a;
}

void Saver::saveRawDataToBinaryFile(const std::string &filename, int t) {
    std::ofstream outputFile(filename, std::ios_base::binary | std::ios_base::app);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open binary file: " << filename << std::endl;
        return;
    }

    int timestep = t;
    int totalAtoms = Synapsin::number + Vesicles::number;

    outputFile.write(reinterpret_cast<const char *>(&timestep), sizeof(timestep));
    outputFile.write(reinterpret_cast<const char *>(&totalAtoms), sizeof(totalAtoms));

    for (int j = 0; j < Synapsin::number; ++j)
        appendAgentData(outputFile, fillSynapsin(j));

    for (int j = 0; j < Vesicles::number; ++j)
        appendAgentData(outputFile, fillVesicle(j));

    outputFile.close();
}

void Saver::logOverlapEvent(int timeStep, int agent_i, int agent_j, int type_i, int type_j,
                            double distance, double interactionRadius) {

    std::ofstream logFile;
    logFile.open("overlap_log.csv", std::ios_base::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open file " << "overlap_log.csv" << std::endl;
        return;
    }
    if (logFile.tellp() == 0) {
        logFile << "Time,Agent_i,Agent_j,Type_i,Type_j,Distance,InteractionRadius,OverlapDepth,"
                << "\n";
    }
    double overlapDepth = interactionRadius - distance;

    logFile << timeStep << ","
            << agent_i << ","
            << agent_j << ","
            << type_i << ","
            << type_j << ","
            << distance << ","
            << interactionRadius << ","
            << overlapDepth << "\n";
    logFile.close();
}

void Saver::saveGridDebug() {
    std::ofstream f("grid_debug.txt");

    f << GP::x_size << " " << GP::y_size << " " << GP::z_size << "\n";
    for (int i = 0; i < GP::x_size; ++i)
        for (int j = 0; j < GP::y_size; ++j) {
            for (int k = 0; k < GP::z_size; ++k)
                f << GP::grid[i][j][k] << " ";
            f << "\n";
        }
}
