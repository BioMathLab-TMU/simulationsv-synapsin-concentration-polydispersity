#include "Converter.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <cmath>
#include <numbers>
#include <array>
#include <stdexcept>
#include "Environment.h"
#include "GP.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Parameters {
    std::string name;
    std::vector<double> numerator;
    std::vector<double> denominator;
};

const double Converter::Lu = 1e9;
const double Converter::tu = 1e3;
const double Converter::Mu = 1.380649e-11;
const double Converter::chu = 1.602e-19;

const std::unordered_map<std::string, Parameters> constants = {
        {"Boltzmann constant",         {"Boltzmann constant",         {Converter::Lu, Converter::Lu},                {Converter::tu, Converter::tu, Converter::Mu}}},
        {"Viscosity",                  {"Viscosity",                  {},                                            {Converter::Lu, Converter::tu, Converter::Mu}}},
        {"electronic charge",          {"electronic charge",          {},                                            {Converter::chu}}},
        {"hamaker constant",           {"hamaker constant",           {Converter::Lu, Converter::Lu},                {Converter::Mu, Converter::tu, Converter::tu}}},
        {"Permittivity of free space", {"Permittivity of free space", {Converter::Mu, Converter::tu, Converter::tu}, {Converter::Lu, Converter::Lu, Converter::Lu, Converter::chu, Converter::chu}}},
        {"EPS",                        {"EPS",                        {Converter::Mu, Converter::Lu, Converter::Lu}, {Converter::tu, Converter::tu}}}
};

namespace {

struct GeometryDimensions {
    double length_nm;
    double width_nm;
    double height_nm;
};

GeometryDimensions dimensionsForVolume(double volume_nm3, int shape) {
    const double pi = std::numbers::pi_v<double>;
    switch (shape) {
        case 0: {
            const double side = std::cbrt(volume_nm3);
            return {side, side, side};
        }
        case 2: {
            const double radius = std::cbrt(3.0 * volume_nm3 / (4.0 * pi));
            return {2.0 * radius, 2.0 * radius, 2.0 * radius};
        }
        case 3: {
            const double radius = std::cbrt(3.0 * volume_nm3 / (2.0 * pi));
            return {2.0 * radius, 2.0 * radius, radius};
        }
        case 4: {

            const double radius = std::cbrt(volume_nm3 / (2.0 * pi));
            return {2.0 * radius, 2.0 * radius, 2.0 * radius};
        }
        default:
            throw std::invalid_argument(
                    "Volume-derived dimensions require cube, sphere, hemisphere or cylinder");
    }
}

std::array<double, 3> cellStepsForOuterDiameter(
        const GeometryDimensions &dimensions,
        double outer_diameter_supremum_nm) {
    const std::array<double, 3> lengths = {
            dimensions.length_nm,
            dimensions.width_nm,
            dimensions.height_nm
    };
    std::array<double, 3> steps{};
    for (int axis = 0; axis < 3; ++axis) {
        const int cells = std::max(
                1,
                static_cast<int>(std::floor(
                        lengths[axis] / outer_diameter_supremum_nm)));
        steps[axis] = lengths[axis] / static_cast<double>(cells);
    }
    return steps;
}

double calculateDebyeLengthNm(
        double ionic_strength_molar,
        double temperature_kelvin,
        double relative_permittivity) {
    constexpr double elementary_charge_c = 1.602176634e-19;
    constexpr double vacuum_permittivity_f_per_m = 8.8541878128e-12;
    constexpr double boltzmann_j_per_k = 1.380649e-23;
    constexpr double avogadro_per_mol = 6.02214076e23;

    const double kappa_per_m = std::sqrt(
            2.0 * elementary_charge_c * elementary_charge_c
            * avogadro_per_mol * 1000.0 * ionic_strength_molar
            / (vacuum_permittivity_f_per_m * relative_permittivity
               * boltzmann_j_per_k * temperature_kelvin));
    return 1e9 / kappa_per_m;
}

}

double Converter::convertUnit(double value, const std::string &constant) {
    auto it = constants.find(constant);
    if (it != constants.end()) {
        Parameters param = it->second;
        double num_product = 1.0;
        for (const auto &num: param.numerator) {
            num_product *= num;
        }

        double denom_product = 1.0;
        for (const auto &denom: param.denominator) {
            denom_product *= denom;
        }

        double result = num_product / denom_product;
        return value * result;
    } else {
        std::cerr << "Error: Unknown constant \"" << constant << "\"" << std::endl;
        return 0.0;
    }
}

int Converter::calculate_slide(double area, int shape) {
    int slide;
    switch (shape) {

        case 0:
            slide = std::ceil(std::cbrt(area));
            break;
        case 1:
            slide = 0;
            break;

        case 2:
            slide = std::ceil(2.0 * std::cbrt(
                    (area * 3) / (4 * std::numbers::pi_v<double>)));
            break;
        case 3:
            slide = std::ceil(2.0 * std::cbrt(
                    (area * 3) / (2 * std::numbers::pi_v<double>)));
            break;
        case 4:
            slide = std::ceil(2.0 * std::cbrt(
                    area / (2.0 * std::numbers::pi_v<double>)));
            break;
        default:
            throw std::invalid_argument("Unknown geometry in calculate_slide");
    }
    return slide;
}

void Converter::saveConvertedUnits(const std::string &constant) {

    const fs::path configuration_root = GP::source_folder_name.empty()
                                        ? fs::current_path()
                                        : GP::source_folder_name;
    const fs::path registry_path = configuration_root
            / "model_parameters.json";
    std::ifstream registry_stream(registry_path);
    if (!registry_stream.is_open()) {
        throw std::runtime_error(
                "Canonical model-parameter registry was not found: "
                + registry_path.generic_string());
    }
    json model_registry;
    registry_stream >> model_registry;
    const auto &geometry_registry = model_registry.at("geometry");
    const auto &vesicle_registry = model_registry.at("vesicle");
    const auto &synapsin_registry = model_registry.at("synapsin");
    const auto &population_registry = model_registry.at("population");
    const auto &physical_registry = model_registry.at("physical_environment");
    const auto &force_registry = model_registry.at("force_model");
    const auto &contact_registry = model_registry.at("contact_and_overlap");
    const auto &numerics_registry = model_registry.at("numerics");
    const auto &analysis_registry = model_registry.at("analysis_observables");

    float box_fract = 1.0;
    float scale = 1.0;
    float control_num = 1.0;

    double volume = geometry_registry.at("canonical_volume_nm3").get<double>()
                    * box_fract * scale;
    const std::string canonical_shape =
            geometry_registry.at("canonical_shape").get<std::string>();
    int shape = geometry_registry.at("shape_codes")
            .at(canonical_shape).get<int>();
    const double core_radius_max_nm =
            vesicle_registry.at("core_radius_max_nm").get<double>();
    const double corona_thickness_nm =
            vesicle_registry.at("corona_thickness_nm").get<double>();
    const double outer_diameter_supremum_nm =
            2.0 * (core_radius_max_nm + corona_thickness_nm);
    const auto geometry = dimensionsForVolume(volume, shape);
    const auto cell_steps = cellStepsForOuterDiameter(
            geometry, outer_diameter_supremum_nm);
    int slide = static_cast<int>(std::ceil(std::max({
            geometry.length_nm, geometry.width_nm, geometry.height_nm})));
    auto kb = convertUnit(1.380649e-23, "Boltzmann constant");
    auto T = physical_registry.at("temperature_K").get<double>();
    auto eta = convertUnit(
            physical_registry.at("viscosity_Pa_s").get<double>(),
            "Viscosity");
    const double relative_permittivity =
            physical_registry.at("relative_permittivity").get<double>();
    const double ionic_strength_molar =
            physical_registry.at("ionic_strength_M").get<double>();
    const double debye_length_nm = calculateDebyeLengthNm(
            ionic_strength_molar, T, relative_permittivity);
    const auto filePath = configuration_root / fs::path(constant);
    float dt = numerics_registry.at("candidate_time_step_ms").get<float>();
    float totalTime = 7.0;
    int Ttime = std::ceil(totalTime / dt);
    float f_max = 1.0;

    json pre_test = {
            {"centering", 0},
            {"f_off",     false},
            {"dataAnalysis",     true},
    };

    json forces_parms = {

            {"connectivity_shell_nm",
             analysis_registry.at("connectivity_shell_nm_provisional")},
            {"adjusted_max",                       f_max},
            {"dynamic_drift_limit_fraction_of_radius",
             numerics_registry.at("drift_limit_fraction_of_outer_radius")},
            {"enable_overlap_projection",          true},
            {"overlap_projection_max_iterations",
             contact_registry.at("overlap_projection_max_iterations")},
            {"overlap_projection_residual_tolerance_nm",
             contact_registry.at("projection_residual_tolerance_nm")},
            {"enable_drift_component_diagnostics", false},
            {"core_allowed_overlap_nm",
             contact_registry.at("core_allowed_overlap_nm")},
            {"core_energy_kbt",                    10.0},
            {"enable_dynamic_steric_penalty",
             contact_registry.at("dynamic_steric_penalty_enabled")},
            {"minimum_surface_separation_nm",
             force_registry.at("van_der_waals").at(
                     "minimum_surface_separation_nm")},

            {"enable_electrostatic",
             force_registry.at("electrostatic").at("enabled")},
            {"enable_van_der_waals",
             force_registry.at("van_der_waals").at("enabled_in_article_configs")},
            {"enable_hydrophobic",
             force_registry.at("hydrophobic").at("enabled")},
            {"continuous_contact_forces",           false},
            {"hydrophobic_decay_length_nm",
             force_registry.at("hydrophobic").at("decay_length_nm")},

            {"hydrophobic_energy_syn_syn_kbt",
             force_registry.at("hydrophobic").at(
                     "contact_energy_syn_syn_kbt_prior")},
            {"hydrophobic_energy_syn_ves_kbt",
             force_registry.at("hydrophobic").at(
                     "contact_energy_syn_ves_kbt_prior")},
            {"hydrophobic_energy_ves_ves_kbt",
             force_registry.at("hydrophobic").at(
                     "contact_energy_ves_ves_kbt")},
    };

    json times = {
            {"Ttime",                           Ttime},
            {"time_step",                       dt},
            {"integrator",                      "euler_maruyama"},
            {"sk_rock_stages",                  32},
            {"sk_rock_damping",                 0.05},
            {"loop_minimiz",                    100 * (1 / f_max)},
            {"minimizer_overlap_tolerance_nm",  0.05},
            {"minimizer_consecutive_checks",    3},
    };

    json total = {
            {"load",        10},
            {"seed",        1234567890},
            {"actual_seed", nullptr},
            {"outputMode",  {"Console", "LogFile"}},
            {"log",         false},
            {"save_forces", true},
            {"use_seed",    false},
            {"continue",    false},

            {"resume_source", ""},

            {"resume_mode", "recovery"},

            {"address",     "simulation"}
    };

    json physical_const = {
            {"boltzmann",    kb},
            {"temperature",  T},
            {"dimension",    3},
            {"eta",          eta},
            {"electronic",   convertUnit(1.602e-19, "electronic charge")},
            {"permittivity", convertUnit(8.854e-12, "Permittivity of free space")},
            {"hamaker",      convertUnit(
                    force_registry.at("van_der_waals").at(
                            "hamaker_constant_J").get<double>(),
                    "hamaker constant")},
            {"solvent_relative_permittivity", relative_permittivity},
            {"ionic_strength_molar", ionic_strength_molar},

            {"debye_length_nm", debye_length_nm},
            {"kappa_nm_inverse", 1.0 / debye_length_nm}

    };

    json environment = {
            {"labels",         {-2,        -1}},
            {"length",         geometry.length_nm},
            {"width",          geometry.width_nm},
            {"height",         geometry.height_nm},
            {"slide",          slide},
            {"box_fract",      box_fract},
            {"scale",          scale},
            {"steps",          {cell_steps[0], cell_steps[1], cell_steps[2]}},
            {"out",            2},
            {"volume",         volume},
            {"analysis_surface_shell_nm",
             model_registry.at("linked_cells").at(
                     "analysis_surface_shell_nm")},
            {"cell_size_policy", "supremum_outer_vesicle_diameter"},
            {"boundary_condition", "closed_reflecting_no_flux"},
            {"Shapes",         {"CUBE", "DATA", "SPHERE", "HEMISPHERE", "CYLINDER"}},
            {"shape",          shape},
            {"NeighborTypes",  {"MAIN",    "SUB"}},
            {"Neibor",         1},
            {"NeighborLevels", {"PRIMARY", "SECONDARY"}},
            {"levelNei",       1}

    };

    json connectivity = {
            {"Synapsin–Synapsin", 10.0},
            {"Vesicle–Vesicle",   46.0},
            {"Synapsin–Vesicle",  28.0},
            {"Vesicle-Synapsin",  28.0}
    };

    json vesicles = {
            {"num", population_registry.at(
                    "canonical_vesicle_count").get<double>()
                    * scale * control_num},
            {"std_radius", vesicle_registry.at("std_core_radius_nm")},
            {"mean_radius", vesicle_registry.at("mean_core_radius_nm")},
            {"std_core_radius_nm", vesicle_registry.at("std_core_radius_nm")},
            {"mean_core_radius_nm", vesicle_registry.at("mean_core_radius_nm")},
            {"core_radius_min_nm", vesicle_registry.at("core_radius_min_nm")},
            {"core_radius_max_nm",   core_radius_max_nm},
            {"corona_thickness_nm",  corona_thickness_nm},
            {"surface_potential_mV", vesicle_registry.at("surface_potential_mV")},
            {"drag_ves",             1.0},
            {"D_ves",                1.0}
    };

    json SynapsinI = {
            {"num", population_registry.at(
                    "canonical_synapsin_count").get<double>()
                    * scale * control_num},
            {"surface_potential_mV", synapsin_registry.at("surface_potential_mV")},
            {"radius", synapsin_registry.at("effective_radius_nm")},
            {"drag_syn",         1.0},
            {"D_syn",            1.0}
    };

    json output = {
            {"total",        {total}},
            {"parameters",   {physical_const}},
            {"environment",  {environment}},
            {"vesicles",     {vesicles}},
            {"connectivity", {connectivity}},
            {"synapsinI",    {SynapsinI}},
            {"pre_test",     {pre_test}},
            {"forces_parms", {forces_parms}},
            {"times",        {times}}
    };

    std::ofstream file(filePath);
    if (file.is_open()) {
        file << output.dump(4);
        file.close();
        std::cout << "Data saved to " << constant << std::endl;
    } else {
        std::cerr << "Failed to open file: " << constant << std::endl;
    }
}

static fs::path resolve_dashboard_path() {
    fs::path p = fs::current_path() / "Dashboard.json";
    if (!fs::exists(p) && !GP::source_folder_name.empty()) {
        p = GP::source_folder_name / "Dashboard.json";
    }
    return p;
}

static json &ensure_section_array0(json &root, const std::string &section) {
    if (!root.contains(section) || !root[section].is_array() || root[section].empty()) {
        root[section] = json::array({json::object()});
    }
    return root[section][0];
}

static std::vector<std::string> split_path(const std::string &path) {
    std::vector<std::string> parts;
    std::stringstream ss(path);
    std::string item;
    while (std::getline(ss, item, '.')) {
        if (!item.empty()) parts.push_back(item);
    }
    return parts;
}

static bool is_array_section(const std::string& key) {
    static const std::vector<std::string> array_sections = {
            "total",
            "parameters",
            "environment",
            "vesicles",
            "synapsinI",
            "connectivity",
            "pre_test",
            "forces_parms",
            "times"
    };

    return std::find(
            array_sections.begin(),
            array_sections.end(),
            key
    ) != array_sections.end();
}

void Converter::setDashboardValue(const std::string &dotPath, const json &value) {
    fs::path filePath = resolve_dashboard_path();

    json j;
    if (fs::exists(filePath)) {
        std::ifstream in(filePath);
        if (in.is_open()) {
            try { in >> j; } catch (...) {}
        }
    }
    if (!j.is_object()) j = json::object();

    auto parts = split_path(dotPath);
    if (parts.empty()) return;

    json *cursor = &j;

    if (is_array_section(parts[0])) {
        json &sec0 = ensure_section_array0(j, parts[0]);
        cursor = &sec0;

        for (size_t i = 1; i + 1 < parts.size(); ++i) {
            const std::string &k = parts[i];
            if (!(*cursor)[k].is_object()) (*cursor)[k] = json::object();
            cursor = &(*cursor)[k];
        }
        (*cursor)[parts.back()] = value;
    } else {

        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            const std::string &k = parts[i];
            if (!(*cursor)[k].is_object()) (*cursor)[k] = json::object();
            cursor = &(*cursor)[k];
        }
        (*cursor)[parts.back()] = value;
    }

    std::ofstream out(filePath, std::ios::trunc);
    if (out.is_open()) out << j.dump(4);
}
