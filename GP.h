#ifndef CODE_MYDISSERATION_GP_H
#define CODE_MYDISSERATION_GP_H

#include <iostream>
#include <vector>
#include <boost/multi_array.hpp>
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "Environment.h"
#include <chrono>
#include <ctime>
#include "Converter.h"
#include <thread>
#include <string>
#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)

#include <sys/utsname.h>

#endif

class GP {
public:
    GP();

    static std::string build_folder_name;
    static  std::filesystem::path source_folder_name;

    static std::string new_directory;

    static int   centering;
    static bool  f_off;
    static bool dataAnalysis;

    static float cut_off;

    static float adjusted_max;

    static double dynamic_drift_limit_fraction_of_radius;
    static std::vector<std::array<double, 3>> total_force;

    static bool enable_electrostatic;
    static bool enable_van_der_waals;
    static bool enable_hydrophobic;

    static bool continuous_contact_forces;

    static double core_allowed_overlap_nm;
    static double core_energy_kbt;

    static bool enable_dynamic_steric_penalty;

    static double minimum_surface_separation_nm;
    static double hydrophobic_decay_length_nm;
    static double hydrophobic_energy_syn_syn_kbt;
    static double hydrophobic_energy_syn_ves_kbt;
    static double hydrophobic_energy_ves_ves_kbt;

    static bool enable_overlap_projection;
    static int overlap_projection_max_iterations;

    static double overlap_projection_residual_tolerance_nm;

    static bool enable_drift_component_diagnostics;

    static int Total_time;
    static float deltaT;

    static std::string integrator;
    static int sk_rock_stages;
    static double sk_rock_damping;

    static int srock2_degree;
    static int loop_minimiz;
    static double minimizer_overlap_tolerance_nm;
    static int minimizer_consecutive_checks;
    static long long total_seconds;
    static long long hours;
    static long long minutes;
    static long long seconds;

    static int load;
    static std::uint32_t seed;
    static int log;
    static bool save_forces;
    static bool use_seed;
    static bool con;
    static std::mt19937 gen;

    static float kb;
    static float T;
    static int dim;
    static float eta;
    static float eCharge;
    static float e0;
    static double A;
    static float eSolvent;
    static double ionic_strength_molar;
    static double kapa;
    static double debye_length_nm;

    static boost::multi_array<int, 3> grid;
    static boost::multi_array<float, 3> Dgrid;
    static std::vector<std::array<int, 3>> new_coordinate;
    static std::vector<std::vector<std::vector<int>>> neighborhoods;
    static std::vector<std::array<int, 3>> index_inside;
    static std::vector<std::array<int, 3>> index_outside;
    static std::vector<std::array<int, 3>> index_boundary;
    static std::vector<int> labels_out_bound;
    static float length;
    static float width;
    static float height;
    static float scale;
    static float slide;
    static std::vector<float> steps;
    static int out;
    static float box_fract;
    static float volume;
    static int shape;
    static int nei;
    static int neiLev;

    static int x_Synapse;
    static int y_Synapse;
    static int z_Synapse;
    static int x_size;
    static int y_size;
    static int z_size;

    static int number_of_vesicles;

    static float std_radius;
    static float mean_radius;
    static double vesicle_core_radius_min_nm;
    static double vesicle_core_radius_max_nm;
    static double vesicle_corona_thickness_nm;
    static double vesicle_surface_potential_mV;
    static float drag_ves;
    static float D_ves;

    static int number_of_SynapsinI;
    static float synapsinI_radius;
    static double synapsin_surface_potential_mV;
    static float drag_syn;
    static float D_syn;

    struct StochasticParameters {
        float sum;
        float avg;
        float std;
        float min;
        float max;
        StochasticParameters() : sum(0.0), avg(0.0), std(0.0), min(0.0), max(0.0) {}
        StochasticParameters(float t, float a, float s, float mi, float ma) : sum(t), avg(a), std(s), min(mi), max(ma) {}

    };

    static void initializeGrid(boost::multi_array<int, 3> net);

    static void updateDerivedElectrostatics();

    static void initializeGenerator() {
        if (use_seed) {
            gen.seed(seed);

        } else {
            std::random_device rd;
            seed = static_cast<std::uint32_t>(rd());
            gen.seed(seed);
        }
    }
};

#endif
