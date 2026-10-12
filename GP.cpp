#include "GP.h"
#include <cmath>
#include <stdexcept>

std::string GP::build_folder_name;
std::filesystem::path GP::source_folder_name;

std::string GP::new_directory;

int   GP::centering;
bool  GP::f_off;
bool GP::dataAnalysis;

float GP::cut_off;
float GP::adjusted_max;
double GP::dynamic_drift_limit_fraction_of_radius;
std::vector<std::array<double, 3>> GP::total_force;

bool GP::enable_electrostatic;
bool GP::enable_van_der_waals;
bool GP::enable_hydrophobic;
bool GP::continuous_contact_forces;
double GP::core_allowed_overlap_nm;
double GP::core_energy_kbt;
bool GP::enable_dynamic_steric_penalty;
double GP::minimum_surface_separation_nm;
double GP::hydrophobic_decay_length_nm;
double GP::hydrophobic_energy_syn_syn_kbt;
double GP::hydrophobic_energy_syn_ves_kbt;
double GP::hydrophobic_energy_ves_ves_kbt;
bool GP::enable_overlap_projection;
int GP::overlap_projection_max_iterations;
double GP::overlap_projection_residual_tolerance_nm;
bool GP::enable_drift_component_diagnostics;

int GP::Total_time;
float GP::deltaT;
std::string GP::integrator;
int GP::sk_rock_stages;
double GP::sk_rock_damping;
int GP::srock2_degree;
int GP::loop_minimiz;
double GP::minimizer_overlap_tolerance_nm;
int GP::minimizer_consecutive_checks;
long long GP::total_seconds;
long long GP::hours;
long long GP::minutes;
long long GP::seconds;

int GP::load;
std::uint32_t GP::seed;
int GP::log;
bool GP::save_forces;
bool GP::use_seed;
bool GP::con;
std::mt19937 GP::gen;

float GP::kb;
float GP::T;
int GP::dim;
float GP::eta;
float GP::eCharge;
float GP::e0;
double GP::A;
float GP::eSolvent;
double GP::ionic_strength_molar;
double GP::kapa;
double GP::debye_length_nm;

boost::multi_array<int, 3> GP::grid;
boost::multi_array<float, 3> GP::Dgrid;
std::vector<std::array<int, 3>> GP::new_coordinate;
std::vector<std::vector<std::vector<int>>>GP::neighborhoods;
std::vector<std::array<int, 3>> GP::index_inside;
std::vector<std::array<int, 3>> GP::index_outside;
std::vector<std::array<int, 3>> GP::index_boundary;
std::vector<int> GP::labels_out_bound;
float GP::length;
float GP::width;
float GP::height;
float GP::scale;
float GP::slide;

std::vector<float> GP::steps;
int GP::out;
float GP::box_fract;
float GP::volume;
int GP::shape;
int GP::nei;
int GP::neiLev;

int GP::x_Synapse;
int GP::y_Synapse;
int GP::z_Synapse;
int GP::x_size;
int GP::y_size;
int GP::z_size;

int GP::number_of_vesicles;
float GP::std_radius;
float GP::mean_radius;
double GP::vesicle_core_radius_min_nm;
double GP::vesicle_core_radius_max_nm;
double GP::vesicle_corona_thickness_nm;
double GP::vesicle_surface_potential_mV;
float GP::drag_ves;
float GP::D_ves;

int GP::number_of_SynapsinI;
float GP::synapsinI_radius;
double GP::synapsin_surface_potential_mV;
float GP::drag_syn;
float GP::D_syn;

GP::GP() {

}

void GP::initializeGrid(boost::multi_array<int, 3> net) {
    grid.resize(boost::extents[x_size][y_size][z_size]);
    grid = net;
};

void GP::updateDerivedElectrostatics() {

    constexpr double elementary_charge_c = 1.602176634e-19;
    constexpr double vacuum_permittivity_f_per_m = 8.8541878128e-12;
    constexpr double boltzmann_j_per_k = 1.380649e-23;
    constexpr double avogadro_per_mol = 6.02214076e23;

    if (!std::isfinite(ionic_strength_molar)
        || ionic_strength_molar <= 0.0) {
        throw std::invalid_argument(
                "parameters.ionic_strength_molar must be finite and positive");
    }
    if (!std::isfinite(static_cast<double>(T)) || T <= 0.0F) {
        throw std::invalid_argument(
                "parameters.temperature must be finite and positive");
    }
    if (!std::isfinite(static_cast<double>(eSolvent))
        || eSolvent <= 0.0F) {
        throw std::invalid_argument(
                "parameters.solvent_relative_permittivity must be finite and positive");
    }

    const double ionic_strength_mol_per_m3 =
            1000.0 * ionic_strength_molar;
    const double kappa_per_m = std::sqrt(
            2.0 * elementary_charge_c * elementary_charge_c
            * avogadro_per_mol * ionic_strength_mol_per_m3
            / (vacuum_permittivity_f_per_m
               * static_cast<double>(eSolvent)
               * boltzmann_j_per_k * static_cast<double>(T)));

    kapa = kappa_per_m * 1e-9;
    debye_length_nm = 1.0 / kapa;
}
