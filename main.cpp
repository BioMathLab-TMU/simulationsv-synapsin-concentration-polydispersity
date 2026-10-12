#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "nlohmann/json.hpp"

#include <boost/type_index.hpp>

#include "Environment.h"
#include "MyUtility.h"
#include "Vesicles.h"
#include "GP.h"
#include "GF.h"
#include "Interaction.h"
#include "ProjectManager.h"
#include "Synapsin.h"
#include "Converter.h"
#include "Saver.h"

using json = nlohmann::json;

namespace {

struct RuntimeThreadConfiguration {
    bool openmp_enabled = false;
    int available_processors = 1;
    int detected_physical_cores = 0;
    int reserved_cores = 0;
    int selected_threads = 1;
    bool explicit_override = false;
    std::string policy = "openmp_disabled";
    std::string executable_sha256 = "unknown";
};

int positiveEnvironmentInteger(
        const char *name,
        int fallback) {
    const char *raw_value = std::getenv(name);
    if (raw_value == nullptr || raw_value[0] == '\0') return fallback;

    char *end = nullptr;
    const long parsed = std::strtol(raw_value, &end, 10);
    if (end == raw_value || *end != '\0' || parsed <= 0
        || parsed > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

bool booleanEnvironmentValue(const char *name, bool fallback) {
    const char *raw_value = std::getenv(name);
    if (raw_value == nullptr) return fallback;
    const std::string value(raw_value);
    if (value == "1" || value == "true" || value == "TRUE") return true;
    if (value == "0" || value == "false" || value == "FALSE") return false;
    return fallback;
}

std::string environmentValue(
        const char *name,
        const std::string &fallback) {
    const char *raw_value = std::getenv(name);
    return raw_value == nullptr || raw_value[0] == '\0'
           ? fallback : std::string(raw_value);
}

void setProcessEnvironment(
        const std::string &name,
        const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

RuntimeThreadConfiguration configureRuntimeThreads() {
    RuntimeThreadConfiguration configuration;
#ifdef _OPENMP
    configuration.openmp_enabled = true;
    configuration.available_processors = std::max(1, omp_get_num_procs());
    configuration.detected_physical_cores = positiveEnvironmentInteger(
            "SIMULATION_DETECTED_PHYSICAL_CORES",
            0);
    if (configuration.detected_physical_cores > 0) {
        configuration.detected_physical_cores = std::min(
                configuration.detected_physical_cores,
                configuration.available_processors);
    }
    configuration.reserved_cores = positiveEnvironmentInteger(
            "SIMULATION_RESERVED_CORES", 2);

    const char *explicit_threads = std::getenv("OMP_NUM_THREADS");
    if (explicit_threads != nullptr && explicit_threads[0] != '\0') {
        const int requested_threads = positiveEnvironmentInteger(
                "OMP_NUM_THREADS", -1);
        if (requested_threads < 1) {
            throw std::invalid_argument(
                    "OMP_NUM_THREADS must be a single positive integer");
        }
        configuration.selected_threads = std::min(
                requested_threads, configuration.available_processors);
        configuration.explicit_override = booleanEnvironmentValue(
                "SIMULATION_OMP_EXPLICIT_OVERRIDE", true);
        configuration.policy = environmentValue(
                "SIMULATION_THREAD_POLICY", "explicit_OMP_NUM_THREADS");
    } else {
        const int core_basis = configuration.detected_physical_cores > 0
                               ? configuration.detected_physical_cores
                               : configuration.available_processors;
        configuration.selected_threads = std::max(
                1,
                core_basis - configuration.reserved_cores);
        configuration.policy = configuration.detected_physical_cores > 0
                               ? "auto_physical_cores_minus_two"
                               : "auto_available_processors_minus_two";
    }

    omp_set_dynamic(0);
    omp_set_num_threads(configuration.selected_threads);
#else
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    configuration.available_processors = static_cast<int>(
            hardware_threads == 0 ? 1 : hardware_threads);
    configuration.detected_physical_cores = 0;
#endif

    configuration.executable_sha256 = environmentValue(
            "SIMULATION_EXECUTABLE_SHA256", "unknown");
    setProcessEnvironment(
            "SIMULATION_OPENMP_ENABLED",
            configuration.openmp_enabled ? "true" : "false");
    setProcessEnvironment(
            "SIMULATION_OPENMP_AVAILABLE_PROCESSORS",
            std::to_string(configuration.available_processors));
    setProcessEnvironment(
            "SIMULATION_DETECTED_PHYSICAL_CORES",
            std::to_string(configuration.detected_physical_cores));
    setProcessEnvironment(
            "SIMULATION_RESERVED_CORES",
            std::to_string(configuration.reserved_cores));
    setProcessEnvironment(
            "SIMULATION_OPENMP_SELECTED_THREADS",
            std::to_string(configuration.selected_threads));
    setProcessEnvironment(
            "SIMULATION_OMP_EXPLICIT_OVERRIDE",
            configuration.explicit_override ? "true" : "false");
    setProcessEnvironment(
            "SIMULATION_THREAD_POLICY", configuration.policy);
    setProcessEnvironment("OMP_DYNAMIC", "FALSE");

    std::cout << "OpenMP runtime: available_processors="
              << configuration.available_processors
              << " detected_physical_cores="
              << configuration.detected_physical_cores
              << " reserved_cores=" << configuration.reserved_cores
              << " selected_threads=" << configuration.selected_threads
              << " policy=" << configuration.policy << std::endl;
    return configuration;
}

}

void runSimulation(const json &dashboard);

void initializeGP(const json &Dashboard);

json loadDashboard(const std::string &filename, bool createNew);

int main(int argc, char *argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: SimulationSV [Dashboard.json]" << std::endl;
        return 2;
    }
    const RuntimeThreadConfiguration runtime_threads =
            configureRuntimeThreads();
    const std::filesystem::path dashboard_path =
            argc == 2
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("configs/C10_mono_seed9210001.json");

    MyUtility::setOutputMode(MyUtility::OutputMode::Console);

    [[maybe_unused]] ProjectManager PM(false);
    json dashboard = loadDashboard(dashboard_path.generic_string(), true);
    const auto &total_configuration = dashboard.at("total").at(0);
    std::string run_description = total_configuration.value(
            "address", dashboard_path.stem().string());
    if (run_description.empty()) {
        run_description = dashboard_path.stem().string();
    }
    ProjectManager::initializeBuildDirectory(run_description);
    [[maybe_unused]] Saver S;
    ProjectManager::copyDashboardFiles(
            GP::source_folder_name, dashboard_path);
    std::cout << "Input Dashboard: "
              << dashboard_path.generic_string() << std::endl;

    initializeGP(dashboard);

    Converter::setDashboardValue(
            "runtime.openmp_enabled",
            runtime_threads.openmp_enabled);
    Converter::setDashboardValue(
            "runtime.openmp_available_processors",
            runtime_threads.available_processors);
    Converter::setDashboardValue(
            "runtime.detected_physical_cores",
            runtime_threads.detected_physical_cores);
    Converter::setDashboardValue(
            "runtime.reserved_cores",
            runtime_threads.reserved_cores);
    Converter::setDashboardValue(
            "runtime.openmp_selected_threads",
            runtime_threads.selected_threads);
    Converter::setDashboardValue(
            "runtime.explicit_thread_override",
            runtime_threads.explicit_override);
    Converter::setDashboardValue(
            "runtime.thread_policy",
            runtime_threads.policy);
    Converter::setDashboardValue(
            "runtime.executable_sha256",
            runtime_threads.executable_sha256);

    auto start_time = std::chrono::high_resolution_clock::now();

    runSimulation(dashboard);

    auto end_time = std::chrono::high_resolution_clock::now();

    const auto total_milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time
            ).count();

    const double total_seconds_exact =
            static_cast<double>(total_milliseconds) / 1000.0;

    GP::total_seconds = total_milliseconds / 1000;
    GP::hours = GP::total_seconds / 3600;
    GP::minutes = (GP::total_seconds % 3600) / 60;
    GP::seconds = GP::total_seconds % 60;

    std::cout << "Duration of run: "
              << total_milliseconds
              << " ms\n";

    std::cout << "Duration: "
              << GP::hours << " h : "
              << GP::minutes << " min : "
              << GP::seconds << " sec\n";

    Converter::setDashboardValue(
            "times.wall_runtime_ms",
            total_milliseconds
    );

    Converter::setDashboardValue(
            "times.wall_runtime_seconds",
            total_seconds_exact
    );

    std::cout << "Duration: " << GP::hours << " h : " << GP::minutes << " min : " << GP::seconds << " sec\n";

    Saver::saveAllDataToJson();

    return 0;
}

void initializeGP(const json &Dashboard) {

    if (Dashboard.is_null()) {

        std::cerr << "Error: Dashboard JSON is null!" << std::endl;
        return;
    }

    GP::centering = Dashboard["pre_test"][0]["centering"].get<int>();
    GP::f_off = Dashboard["pre_test"][0]["f_off"].get<bool>();
    GP::dataAnalysis = Dashboard["pre_test"][0]["dataAnalysis"].get<bool>();

    const auto &force_parameters = Dashboard["forces_parms"][0];
    GP::cut_off = force_parameters.value("connectivity_shell_nm", 2.0);
    GP::adjusted_max = force_parameters.value("adjusted_max", 1.0F);
    GP::dynamic_drift_limit_fraction_of_radius = force_parameters.value(
            "dynamic_drift_limit_fraction_of_radius",
            static_cast<double>(GP::adjusted_max) / 4.0);
    GP::enable_electrostatic = force_parameters.value(
            "enable_electrostatic", true);
    GP::enable_van_der_waals = force_parameters.value(
            "enable_van_der_waals", true);
    GP::enable_hydrophobic = force_parameters.value(
            "enable_hydrophobic", true);
    GP::continuous_contact_forces = force_parameters.value(
            "continuous_contact_forces", false);
    GP::core_allowed_overlap_nm = force_parameters.value(
            "core_allowed_overlap_nm", 0.5);
    GP::core_energy_kbt = force_parameters.value(
            "core_energy_kbt", 10.0);
    GP::enable_dynamic_steric_penalty = force_parameters.value(
            "enable_dynamic_steric_penalty", true);
    GP::minimum_surface_separation_nm = force_parameters.value(
            "minimum_surface_separation_nm", 0.3);
    GP::hydrophobic_decay_length_nm = force_parameters.value(
            "hydrophobic_decay_length_nm", 0.5);
    GP::hydrophobic_energy_syn_syn_kbt = force_parameters.value(
            "hydrophobic_energy_syn_syn_kbt", 0.5);
    GP::hydrophobic_energy_syn_ves_kbt = force_parameters.value(
            "hydrophobic_energy_syn_ves_kbt", 1.0);
    GP::hydrophobic_energy_ves_ves_kbt = force_parameters.value(
            "hydrophobic_energy_ves_ves_kbt", 0.0);
    GP::enable_overlap_projection = force_parameters.value(
            "enable_overlap_projection", true);
    GP::overlap_projection_max_iterations = force_parameters.value(
            "overlap_projection_max_iterations", 4);
    GP::overlap_projection_residual_tolerance_nm = force_parameters.value(
            "overlap_projection_residual_tolerance_nm", 1e-6);
    GP::enable_drift_component_diagnostics = force_parameters.value(
            "enable_drift_component_diagnostics", false);

    const auto finitePositive = [](double value) {
        return std::isfinite(value) && value > 0.0;
    };
    const auto finiteNonNegative = [](double value) {
        return std::isfinite(value) && value >= 0.0;
    };
    if (!finiteNonNegative(GP::cut_off)) {
        throw std::invalid_argument(
                "forces_parms.connectivity_shell_nm must be finite and non-negative");
    }
    if (!finitePositive(GP::adjusted_max)
        || !finitePositive(GP::dynamic_drift_limit_fraction_of_radius)
        || !finitePositive(GP::core_allowed_overlap_nm)
        || !finitePositive(GP::core_energy_kbt)
        || !finitePositive(GP::minimum_surface_separation_nm)
        || !finitePositive(GP::hydrophobic_decay_length_nm)) {
        throw std::invalid_argument(
                "Force length, core-energy and numerical-limit parameters must be finite and positive");
    }
    if (!finiteNonNegative(GP::hydrophobic_energy_syn_syn_kbt)
        || !finiteNonNegative(GP::hydrophobic_energy_syn_ves_kbt)
        || !finiteNonNegative(GP::hydrophobic_energy_ves_ves_kbt)) {
        throw std::invalid_argument(
                "Hydrophobic contact energies must be finite and non-negative");
    }
    if (GP::overlap_projection_max_iterations < 0) {
        throw std::invalid_argument(
                "forces_parms.overlap_projection_max_iterations must be non-negative");
    }
    if (!finitePositive(GP::overlap_projection_residual_tolerance_nm)) {
        throw std::invalid_argument(
                "forces_parms.overlap_projection_residual_tolerance_nm must be finite and positive");
    }

    const auto &time_parameters = Dashboard["times"][0];
    GP::Total_time = time_parameters["Ttime"].get<int>();
    GP::deltaT = time_parameters["time_step"].get<double>();
    GP::integrator = time_parameters.value(
            "integrator", std::string("euler_maruyama"));
    GP::sk_rock_stages = time_parameters.value("sk_rock_stages", 32);
    GP::sk_rock_damping = time_parameters.value("sk_rock_damping", 0.05);
    GP::srock2_degree = time_parameters.value("srock2_degree", 47);

    const int configured_minimizer_iterations =
            time_parameters["loop_minimiz"].get<int>();
    const bool has_adaptive_minimizer_settings =
            time_parameters.contains("minimizer_overlap_tolerance_nm")
            || time_parameters.contains("minimizer_consecutive_checks");

    GP::loop_minimiz = has_adaptive_minimizer_settings
                       ? configured_minimizer_iterations
                       : std::max(configured_minimizer_iterations, 100);
    GP::minimizer_overlap_tolerance_nm = time_parameters.value(
            "minimizer_overlap_tolerance_nm", 0.05);
    GP::minimizer_consecutive_checks = time_parameters.value(
            "minimizer_consecutive_checks", 3);

    if (GP::loop_minimiz < 0) {
        throw std::invalid_argument(
                "times.loop_minimiz must be non-negative");
    }
    if (!std::isfinite(GP::minimizer_overlap_tolerance_nm)
        || GP::minimizer_overlap_tolerance_nm < 0.0) {
        throw std::invalid_argument(
                "times.minimizer_overlap_tolerance_nm must be finite and non-negative");
    }
    if (GP::minimizer_consecutive_checks < 1) {
        throw std::invalid_argument(
                "times.minimizer_consecutive_checks must be at least 1");
    }

    GP::load = Dashboard["total"][0]["load"].get<int>();
    GP::seed = Dashboard["total"][0]["seed"].get<std::uint32_t>();
    GP::log = Dashboard["total"][0]["log"].get<bool>();
    GP::save_forces = Dashboard["total"][0]["save_forces"].get<bool>();
    GP::use_seed = Dashboard["total"][0]["use_seed"].get<bool>();
    GP::con = Dashboard["total"][0]["continue"].get<bool>();

    if (GP::Total_time < 1) {
        throw std::invalid_argument(
                "times.Ttime must be at least 1");
    }
    if (!std::isfinite(GP::deltaT) || GP::deltaT <= 0.0F) {
        throw std::invalid_argument(
                "times.time_step must be finite and positive");
    }
    if (GP::integrator != "euler_maruyama"
        && GP::integrator != "sk_rock"
        && GP::integrator != "s_rock2") {
        throw std::invalid_argument(
                "times.integrator must be euler_maruyama, sk_rock or s_rock2");
    }
    if (GP::sk_rock_stages < 2 || GP::sk_rock_stages > 128) {
        throw std::invalid_argument(
                "times.sk_rock_stages must be between 2 and 128");
    }
    if (!std::isfinite(GP::sk_rock_damping)
        || GP::sk_rock_damping < 0.0
        || GP::sk_rock_damping > 10.0) {
        throw std::invalid_argument(
                "times.sk_rock_damping must be finite and between 0 and 10");
    }
    if (GP::srock2_degree != 43
        && GP::srock2_degree != 47
        && GP::srock2_degree != 51
        && GP::srock2_degree != 72
        && GP::srock2_degree != 93) {
        throw std::invalid_argument(
                "times.srock2_degree must be 43, 47, 51, 72 or 93");
    }
    if (GP::load < 1) {
        throw std::invalid_argument(
                "total.load must be at least 1");
    }

    GP::kb = Dashboard["parameters"][0]["boltzmann"].get<float>();
    GP::T = Dashboard["parameters"][0]["temperature"].get<float>();
    GP::dim = Dashboard["parameters"][0]["dimension"].get<int>();
    GP::eta = Dashboard["parameters"][0]["eta"].get<float>();
    GP::eCharge = Dashboard["parameters"][0]["electronic"].get<double>();
    GP::e0 = Dashboard["parameters"][0]["permittivity"].get<double>();
    GP::A = Dashboard["parameters"][0]["hamaker"].get<double>();
    const auto &physical_parameters = Dashboard["parameters"][0];
    GP::eSolvent = physical_parameters.value(
            "solvent_relative_permittivity",
            physical_parameters.value("soulvent", 74.2));
    GP::ionic_strength_molar = physical_parameters.value(
            "ionic_strength_molar", 0.15);
    if (!finitePositive(GP::A)) {
        throw std::invalid_argument(
                "parameters.hamaker must be finite and positive");
    }
    GP::updateDerivedElectrostatics();

    GP::labels_out_bound = Dashboard["environment"][0]["labels"].get<std::vector<int>>();
    GP::length = Dashboard["environment"][0]["length"].get<float>();
    GP::width = Dashboard["environment"][0]["width"].get<float>();
    GP::height = Dashboard["environment"][0]["height"].get<float>();
    GP::scale = Dashboard["environment"][0]["scale"].get<float>();
    GP::slide = Dashboard["environment"][0]["slide"].get<float>();
    GP::box_fract = Dashboard["environment"][0]["box_fract"].get<float>();
    GP::steps = Dashboard["environment"][0]["steps"].get<std::vector<float>>();
    GP::out = Dashboard["environment"][0]["out"].get<int>();
    GP::volume = Dashboard["environment"][0]["volume"].get<float>();
    Environment::shape = static_cast<Environment::Shape>(Dashboard["environment"][0]["shape"].get<int>());
    GP::shape = Environment::shape;
    Environment::nei = static_cast<Environment::NeighborType>(Dashboard["environment"][0]["Neibor"].get<int>());
    GP::nei = Environment::nei;
    Environment::neiLev = static_cast<Environment::NeighborLevel>(Dashboard["environment"][0]["levelNei"].get<int>());
    GP::neiLev = Environment::neiLev;

    const auto &vesicle_parameters = Dashboard["vesicles"][0];
    GP::number_of_vesicles = vesicle_parameters["num"].get<int>();
    GP::vesicle_surface_potential_mV = vesicle_parameters.value(
            "surface_potential_mV", -20.0);
    GP::std_radius = vesicle_parameters.value(
            "std_core_radius_nm",
            vesicle_parameters.value("std_radius", 2.5F));
    GP::mean_radius = vesicle_parameters.value(
            "mean_core_radius_nm",
            vesicle_parameters.value("mean_radius", 20.45F));
    GP::vesicle_core_radius_min_nm = vesicle_parameters.value(
            "core_radius_min_nm", 20.0);
    GP::vesicle_core_radius_max_nm = vesicle_parameters.value(
            "core_radius_max_nm", 25.0);
    GP::vesicle_corona_thickness_nm = vesicle_parameters.value(
            "corona_thickness_nm", 0.0);
    GP::drag_ves = vesicle_parameters["drag_ves"].get<float>();
    GP::D_ves = vesicle_parameters["D_ves"].get<float>();

    if (!finiteNonNegative(GP::std_radius)
        || !finitePositive(GP::mean_radius)
        || !finiteNonNegative(GP::vesicle_core_radius_min_nm)
        || !finitePositive(GP::vesicle_core_radius_max_nm)
        || GP::vesicle_core_radius_min_nm
           >= GP::vesicle_core_radius_max_nm
        || !finiteNonNegative(GP::vesicle_corona_thickness_nm)) {
        throw std::invalid_argument(
                "Vesicle core-radius distribution and corona thickness are invalid");
    }
    if (GP::std_radius == 0.0F
        && (GP::mean_radius < GP::vesicle_core_radius_min_nm
            || GP::mean_radius > GP::vesicle_core_radius_max_nm)) {
        throw std::invalid_argument(
                "Monodisperse vesicle core radius must be within its configured bounds");
    }

    if (GP::vesicle_corona_thickness_nm > 0.0) {
        if (GP::nei != static_cast<int>(Environment::SUB)
            || GP::neiLev != static_cast<int>(Environment::SECONDARY)) {
            throw std::invalid_argument(
                    "Corona model requires the fixed two-cell (SUB/SECONDARY) neighborhood");
        }
        if (GP::steps.size() != 3) {
            throw std::invalid_argument(
                    "environment.steps must contain three cell dimensions");
        }
        const double outer_diameter_supremum_nm = 2.0 * (
                GP::vesicle_core_radius_max_nm
                + GP::vesicle_corona_thickness_nm);
        const double analysis_surface_shell_nm =
                Dashboard["environment"][0].value(
                        "analysis_surface_shell_nm", 6.0);
        if (!finiteNonNegative(analysis_surface_shell_nm)) {
            throw std::invalid_argument(
                    "environment.analysis_surface_shell_nm must be finite and non-negative");
        }
        for (const float cell_size: GP::steps) {
            if (!finitePositive(cell_size)
                || static_cast<double>(cell_size) + 1e-5
                   < outer_diameter_supremum_nm) {
                throw std::invalid_argument(
                        "Each linked cell must contain the supremum vesicle outer diameter");
            }
            if (2.0 * static_cast<double>(cell_size) + 1e-5
                < outer_diameter_supremum_nm
                  + analysis_surface_shell_nm) {
                throw std::invalid_argument(
                        "Two-cell neighborhood does not cover outer contact plus the analysis shell");
            }
        }
    }

    GP::number_of_SynapsinI = Dashboard["synapsinI"][0]["num"].get<int>();
    GP::synapsin_surface_potential_mV = Dashboard["synapsinI"][0].value(
            "surface_potential_mV", 16.0);
    GP::synapsinI_radius = Dashboard["synapsinI"][0]["radius"].get<float>();
    GP::drag_syn = Dashboard["synapsinI"][0]["drag_syn"].get<float>();
    GP::D_syn = Dashboard["synapsinI"][0]["D_syn"].get<float>();

    if (!std::isfinite(GP::vesicle_surface_potential_mV)
        || !std::isfinite(GP::synapsin_surface_potential_mV)) {
        throw std::invalid_argument(
                "Particle surface potentials must be finite");
    }

    MyUtility::print("total time: ", GP::Total_time);

    Interaction gf;
}

json loadDashboard(const std::string &filename, bool createNew) {

    const fs::path json_file_path = GP::source_folder_name.empty()
                                    ? fs::current_path() / fs::path(filename)
                                    : GP::source_folder_name / fs::path(filename);

    if (createNew && !fs::exists(json_file_path)) {
        Converter::saveConvertedUnits(filename);
    }

    std::ifstream inputFile(json_file_path);
    json dashboard;

    if (inputFile.is_open()) {

        inputFile >> dashboard;
        inputFile.close();
    } else {
        throw std::runtime_error(
                "Unable to open Dashboard at: "
                + json_file_path.generic_string());
    }

    const auto &total = dashboard.at("total").at(0);
    GP::con = total.value("continue", false);
    GP::new_directory = total.value(
            "resume_source", total.value("address", std::string{}));

    return dashboard;
}

void runSimulation(const json &dashboard) {

    const bool resumed_run = GP::con;
    std::string resume_mode = "none";
    int resume_mode_code = 0;
    bool resume_parameters_changed = false;
    Saver::ResumeCheckpoint resume_checkpoint;

    if (GP::con && (GP::integrator == "sk_rock"
                    || GP::integrator == "s_rock2")) {
        throw std::invalid_argument(
                "Resume is disabled for experimental stabilized-integrator runs until integrator metadata is part of checkpoint compatibility");
    }

    if (resumed_run) {
        const auto &total_configuration = dashboard.at("total").at(0);
        resume_mode = total_configuration.value(
                "resume_mode", std::string("recovery"));
        if (resume_mode == "same_parameters") resume_mode = "recovery";
        if (resume_mode == "extend") resume_mode = "extend_duration";

        if (resume_mode == "recovery") {
            resume_mode_code = 1;
        } else if (resume_mode == "extend_duration") {
            resume_mode_code = 2;
        } else if (resume_mode == "new_parameters") {
            resume_mode_code = 3;
        } else {
            throw std::invalid_argument(
                    "total.resume_mode must be recovery, extend_duration, "
                    "or new_parameters");
        }

        std::string configured_source = total_configuration.value(
                "resume_source", std::string{});
        if (configured_source.empty()) {
            configured_source = total_configuration.value(
                    "address", std::string{});
        }
        const fs::path resolved_source = Saver::resolveResumeSource(
                configured_source, GP::source_folder_name);
        resume_checkpoint = Saver::loadLatestResumeCheckpoint(
                resolved_source);
        resume_parameters_changed = Saver::validateResumeCompatibility(
                resume_checkpoint, resume_mode);

        if (GP::Total_time - 1 <= resume_checkpoint.simulation_step) {
            throw std::invalid_argument(
                    "times.Ttime is the target total trajectory length and "
                    "must end after the checkpoint step");
        }

        std::cout << "Resume checkpoint: "
                  << resume_checkpoint.checkpoint_file << '\n'
                  << "Resume mode: " << resume_mode << '\n'
                  << "Resume state step: "
                  << resume_checkpoint.simulation_step << std::endl;
    }

    GP::initializeGenerator();

    Converter::setDashboardValue("total.actual_seed", GP::seed);

    Converter::setDashboardValue(
            "times.save_interval_steps",
            GP::load
    );

    Converter::setDashboardValue(
            "times.frame_interval_ms",
            static_cast<double>(GP::load) * GP::deltaT
    );

    Converter::setDashboardValue(
            "times.intended_duration_ms",
            static_cast<double>(GP::Total_time) * GP::deltaT
    );

    Converter::setDashboardValue(
            "times.last_integrated_time_ms",
            static_cast<double>(GP::Total_time - 1) * GP::deltaT
    );
    Converter::setDashboardValue("times.integrator", GP::integrator);
    Converter::setDashboardValue(
            "times.sk_rock_stages", GP::sk_rock_stages);
    Converter::setDashboardValue(
            "times.sk_rock_damping", GP::sk_rock_damping);
    Converter::setDashboardValue(
            "times.srock2_degree", GP::srock2_degree);

    Converter::setDashboardValue(
            "times.loop_minimiz",
            GP::loop_minimiz
    );
    Converter::setDashboardValue(
            "times.minimizer_overlap_tolerance_nm",
            GP::minimizer_overlap_tolerance_nm
    );
    Converter::setDashboardValue(
            "times.minimizer_consecutive_checks",
            GP::minimizer_consecutive_checks
    );

    int loop_minimiz = GP::loop_minimiz;
    auto env = Environment();
    std::cout << "shape = " << env.shape << std::endl;
    GP::index_inside = env.index_inside;
    GP::initializeGrid(env.grid);
    GP::neighborhoods = env.neighborhoods;
    GP::new_coordinate = env.new_coordinate;
    GP::index_boundary = env.index_boundary;
    GP::index_outside = env.index_outside;
    auto load = GP::load;
    auto rawData = "simulation_Rawdata___" + ProjectManager::folder_name + ".h5";
    auto rawBinaryData =
            "simulation_Rawdata___" + ProjectManager::folder_name + ".bin";

    Capacity capacity_of_max;

    Saver::saveGridDebug();

    Vesicles vesicles(capacity_of_max);

    double mean_diff = MyUtility::meanDiffVesicles(Vesicles::diffusion_coefficient);
    Converter::setDashboardValue("vesicles.mean_diff_ves", mean_diff);

    Synapsin synapsinI(GP::synapsinI_radius);

    int segment_start_state_step = 0;
    if (resumed_run) {
        Saver::restoreResumeCheckpoint(resume_checkpoint);
        segment_start_state_step = resume_checkpoint.simulation_step;

        Converter::setDashboardValue("total.actual_seed", GP::seed);
        Converter::setDashboardValue("resume.enabled", true);
        Converter::setDashboardValue("resume.mode", resume_mode);
        Converter::setDashboardValue(
                "resume.mode_code", resume_mode_code);
        Converter::setDashboardValue(
                "resume.parameters_changed", resume_parameters_changed);
        Converter::setDashboardValue(
                "resume.parent_checkpoint_file",
                resume_checkpoint.checkpoint_file.filename().generic_string());
        Converter::setDashboardValue(
                "resume.parent_checkpoint_step",
                resume_checkpoint.simulation_step);
        Converter::setDashboardValue(
                "resume.parent_checkpoint_time_ms",
                resume_checkpoint.physical_time_ms);
        Converter::setDashboardValue(
                "resume.segment_start_state_step",
                segment_start_state_step);
    } else {
        Converter::setDashboardValue("resume.enabled", false);
        Converter::setDashboardValue(
                "resume.segment_start_state_step", 0);
    }

    mean_diff = MyUtility::meanDiffVesicles(
            Vesicles::diffusion_coefficient);
    Converter::setDashboardValue("vesicles.mean_diff_ves", mean_diff);
    Converter::setDashboardValue("synapsinI.mean_diff_syn", Synapsin::diffusion_coefficient);
    Saver::initializeRunMetadataToHDF5(
            rawData,
            segment_start_state_step,
            resume_mode_code,
            resume_parameters_changed);

    std::cout << "deltaT " << GP::deltaT << " " << std::endl;
    std::cout << "integrator " << GP::integrator << std::endl;
    if (GP::integrator == "sk_rock") {
        std::cout << "SK-ROCK stages " << GP::sk_rock_stages
                  << ", damping " << GP::sk_rock_damping << std::endl;
    } else if (GP::integrator == "s_rock2") {
        std::cout << "S-ROCK2 polynomial degree "
                  << GP::srock2_degree << std::endl;
    }
    std::cout << "minimizer maximum iterations "
              << GP::loop_minimiz << std::endl;
    std::cout << "minimizer overlap tolerance (nm) "
              << GP::minimizer_overlap_tolerance_nm << std::endl;
    std::cout << "minimizer consecutive checks "
              << GP::minimizer_consecutive_checks << std::endl;

    if (!resumed_run) {
    const auto synapsin_positions_before_minimizer = Synapsin::locations;
    const auto vesicle_positions_before_minimizer = Vesicles::locations;
    const auto overlap_before_minimizer =
            Interaction::measureOverlaps();

    std::vector<Interaction::OverlapDiagnostics> minimizer_history;
    minimizer_history.reserve(static_cast<std::size_t>(loop_minimiz) + 1);
    minimizer_history.push_back(overlap_before_minimizer);

    int minimizer_stop_reason_code = 0;
    std::string minimizer_stop_reason = "maximum_iterations";
    bool minimizer_converged = false;
    int minimizer_iterations_completed = 0;
    int consecutive_tolerance_checks = 0;

    if (overlap_before_minimizer.overlapping_pair_count == 0) {
        minimizer_converged = true;
        minimizer_stop_reason_code = 1;
        minimizer_stop_reason = "no_overlaps";
    } else if (overlap_before_minimizer.overlap_max_nm
               <= GP::minimizer_overlap_tolerance_nm) {
        consecutive_tolerance_checks = 1;
        if (consecutive_tolerance_checks
            >= GP::minimizer_consecutive_checks) {
            minimizer_converged = true;
            minimizer_stop_reason_code = 2;
            minimizer_stop_reason = "overlap_tolerance";
        }
    }

    for (int i = 0; i < loop_minimiz && !minimizer_converged; ++i) {
        Interaction::minimizerAgents();
        ++minimizer_iterations_completed;

        const auto current_overlap = Interaction::measureOverlaps();
        minimizer_history.push_back(current_overlap);

        std::cout << "Minimizer: " << minimizer_iterations_completed
                  << " pairs=" << current_overlap.overlapping_pair_count
                  << " max_overlap_nm=" << current_overlap.overlap_max_nm
                  << std::endl;

        if (current_overlap.overlapping_pair_count == 0) {
            minimizer_converged = true;
            minimizer_stop_reason_code = 1;
            minimizer_stop_reason = "no_overlaps";
            consecutive_tolerance_checks = 0;
        } else if (current_overlap.overlap_max_nm
                   <= GP::minimizer_overlap_tolerance_nm) {
            ++consecutive_tolerance_checks;
            if (consecutive_tolerance_checks
                >= GP::minimizer_consecutive_checks) {
                minimizer_converged = true;
                minimizer_stop_reason_code = 2;
                minimizer_stop_reason = "overlap_tolerance";
            }
        } else {
            consecutive_tolerance_checks = 0;
        }
    }

    const auto &overlap_after_minimizer = minimizer_history.back();

    Saver::saveMinimizerDiagnosticsToHDF5(
            rawData,
            synapsin_positions_before_minimizer,
            vesicle_positions_before_minimizer,
            Synapsin::locations,
            Vesicles::locations,
            overlap_before_minimizer,
            overlap_after_minimizer,
            minimizer_history,
            loop_minimiz,
            minimizer_iterations_completed,
            GP::minimizer_overlap_tolerance_nm,
            GP::minimizer_consecutive_checks,
            consecutive_tolerance_checks,
            minimizer_stop_reason_code,
            minimizer_converged);

    const auto overlapDiagnosticsToJson = [](
            const Interaction::OverlapDiagnostics &diagnostics) {
        return json{
                {"overlapping_pair_count",
                 diagnostics.overlapping_pair_count},
                {"overlapping_particle_count",
                 diagnostics.overlapping_particle_count},
                {"overlapping_synapsin_count",
                 diagnostics.overlapping_synapsin_count},
                {"overlapping_vesicle_count",
                 diagnostics.overlapping_vesicle_count},
                {"ss_pair_count",
                 diagnostics.synapsin_synapsin_pair_count},
                {"sv_pair_count",
                 diagnostics.synapsin_vesicle_pair_count},
                {"vv_pair_count",
                 diagnostics.vesicle_vesicle_pair_count},
                {"overlap_sum_nm",
                 diagnostics.overlap_sum_nm},
                {"overlap_mean_nm",
                 diagnostics.overlap_mean_nm},
                {"overlap_max_nm",
                 diagnostics.overlap_max_nm}
        };
    };

    Converter::setDashboardValue(
            "minimizer",
            json{
                    {"iterations_requested", loop_minimiz},
                    {"iterations_completed",
                     minimizer_iterations_completed},
                    {"history_points", minimizer_history.size()},
                    {"overlap_tolerance_nm",
                     GP::minimizer_overlap_tolerance_nm},
                    {"consecutive_checks_required",
                     GP::minimizer_consecutive_checks},
                    {"consecutive_checks_achieved",
                     consecutive_tolerance_checks},
                    {"converged", minimizer_converged},
                    {"stop_reason", minimizer_stop_reason},
                    {"stop_reason_code", minimizer_stop_reason_code},
                    {"before",
                     overlapDiagnosticsToJson(overlap_before_minimizer)},
                    {"after",
                     overlapDiagnosticsToJson(overlap_after_minimizer)}
            });

    std::cout << "Minimizer overlapping pairs: "
              << overlap_before_minimizer.overlapping_pair_count
              << " -> "
              << overlap_after_minimizer.overlapping_pair_count
              << std::endl;
    std::cout << "Minimizer max overlap (nm): "
              << overlap_before_minimizer.overlap_max_nm
              << " -> "
              << overlap_after_minimizer.overlap_max_nm
              << std::endl;
    std::cout << "Minimizer converged: "
              << std::boolalpha << minimizer_converged
              << " (" << minimizer_stop_reason << ")"
              << std::noboolalpha << std::endl;

    } else {

        const auto resume_overlap = Interaction::measureOverlaps();
        Converter::setDashboardValue(
                "minimizer",
                json{
                        {"skipped_on_resume", true},
                        {"iterations_requested", 0},
                        {"iterations_completed", 0},
                        {"overlapping_pair_count",
                         resume_overlap.overlapping_pair_count},
                        {"overlap_max_nm", resume_overlap.overlap_max_nm}
                });
        std::cout << "Minimizer skipped: restored checkpoint state is kept "
                     "unchanged" << std::endl;
    }

    int total_time = GP::Total_time;

    int saved_hdf5_frame_count = 0;

    Interaction::resetRuntimeDiagnostics();
    GF::resetBoundaryDiagnostics(true);

    if (!GP::f_off) {

        for (int state_step = segment_start_state_step;
             state_step < total_time - 1;
             ++state_step) {

            if (state_step > 0 && state_step % load == 0) {
                std::cout << "step : " << state_step << "-th" << std::endl;
            }

            const bool save_segment_start =
                    resumed_run
                    && state_step == segment_start_state_step;
            const bool save_regular_frame =
                    state_step > segment_start_state_step
                    && state_step % load == 0;
            const bool save_current_frame =
                    save_segment_start || save_regular_frame;

            if (GP::integrator == "sk_rock"
                || GP::integrator == "s_rock2") {
                if (save_current_frame) {
                    Interaction::agentInteraction(true);
                }
                Synapsin::drift_scale.assign(
                        Synapsin::locations.size(), 1.0);
                Vesicles::drift_scale.assign(
                        Vesicles::locations.size(), 1.0);
            } else {
                Interaction::agentInteraction(save_current_frame);
                Vesicles::prepareDriftScales();
                Synapsin::prepareDriftScales();
            }

            if (save_current_frame) {
                Saver::saveLocationsToFile(
                        "Agent_movement.xyz", state_step);
                Saver::saveRawDataToHDF5(rawData, state_step);
                ++saved_hdf5_frame_count;
            }

            if (GP::integrator == "sk_rock") {
                Interaction::advanceSkRockMacroStep();
            } else if (GP::integrator == "s_rock2") {
                Interaction::advanceSrock2MacroStep();
            } else {
                Vesicles::movementVesicle();
                synapsinI.movementSynapsin();
            }
            Interaction::resolveDynamicOverlaps();
            Interaction::recordRuntimeStep();

        }

        const int final_state_step = total_time - 1;
        if (final_state_step > segment_start_state_step
            && final_state_step % load == 0) {
            Interaction::agentInteraction(true);
            if (GP::integrator == "sk_rock"
                || GP::integrator == "s_rock2") {
                Synapsin::drift_scale.assign(
                        Synapsin::locations.size(), 1.0);
                Vesicles::drift_scale.assign(
                        Vesicles::locations.size(), 1.0);
            } else {
                Vesicles::prepareDriftScales();
                Synapsin::prepareDriftScales();
            }
            Saver::saveLocationsToFile(
                    "Agent_movement.xyz", final_state_step);
            Saver::saveRawDataToHDF5(rawData, final_state_step);
            ++saved_hdf5_frame_count;
        }

    } else {
        for (int i = segment_start_state_step + 1;
             i < total_time;
             ++i) {
            if (i % load == 0) {
                std::cout << "step : " << i << "-th" << std::endl;
            }
            if (GP::integrator == "sk_rock") {
                Interaction::advanceSkRockMacroStep();
            } else if (GP::integrator == "s_rock2") {
                Interaction::advanceSrock2MacroStep();
            } else {
                vesicles.movementVesicle();
                synapsinI.movementSynapsin();
            }
            Interaction::recordRuntimeStep();
            Saver::saveLocationsToFile("Agent_movement.xyz", i);
            if (GP::dataAnalysis) {
                Saver::saveRawDataToBinaryFile(rawBinaryData, i);
            }
            if (i % load == 0) {
                Saver::saveResumeCheckpoint(i, rawBinaryData);
            }

        }

    }

    Saver::finalizeRunMetadataToHDF5(
            rawData,
            total_time - 1,
            saved_hdf5_frame_count,
            segment_start_state_step,
            resumed_run);

}
