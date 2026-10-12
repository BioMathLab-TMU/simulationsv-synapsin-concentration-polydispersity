#ifndef GP_CPP_SAVER_H
#define GP_CPP_SAVER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <cstdint>

#include <boost/multi_array.hpp>
#include "nlohmann/json.hpp"

#include "GP.h"
#include "Interaction.h"

class Saver {
public:
    struct ResumeCheckpoint {
        int simulation_step = 0;
        double physical_time_ms = 0.0;
        std::uint32_t actual_seed = 0;
        std::filesystem::path checkpoint_file;
        nlohmann::json configuration;
        std::string rng_state;

        std::vector<std::array<float, 3>> synapsin_locations;
        std::vector<std::array<int, 3>> synapsin_positions;
        std::vector<std::vector<int>> which_synapsin_where;
        float synapsin_radius = 0.0F;
        float synapsin_diffusion_coefficient = 0.0F;
        float synapsin_drag = 0.0F;

        std::vector<std::array<float, 3>> vesicle_locations;
        std::vector<std::array<int, 3>> vesicle_positions;
        std::vector<std::vector<int>> which_vesicle_where;
        std::vector<float> vesicle_core_radii;
        std::vector<float> vesicle_radii;
        std::vector<float> vesicle_diffusion_coefficient;
        std::vector<float> vesicle_drag;
        std::vector<float> vesicle_volume;
        std::vector<bool> vesicle_active;
    };

    static void saveAllDataToJson();

    static void saveToJsonTotalData();

    static void saveGPDataToJson();

    static void saveInteractionDataToJson();

    static void saveVesicleDataToJson();

    static void saveSynapsinDataToJson();

    static void saveToJsonToEnvironment();

    static void saveToJsonToParameters();

    static void saveRawDataToHDF5(const std::string& filename, int t);

    static void initializeRunMetadataToHDF5(
            const std::string &filename,
            int segment_start_state_step,
            int resume_mode_code,
            bool resume_parameters_changed);

    static void finalizeRunMetadataToHDF5(
            const std::string &filename,
            int final_state_step,
            int saved_hdf5_frame_count,
            int segment_start_state_step,
            bool resumed_run);

    static std::filesystem::path resolveResumeSource(
            const std::string &configured_source,
            const std::filesystem::path &project_root);

    static ResumeCheckpoint loadLatestResumeCheckpoint(
            const std::filesystem::path &source);

    static bool validateResumeCompatibility(
            const ResumeCheckpoint &checkpoint,
            const std::string &resume_mode);

    static void restoreResumeCheckpoint(
            const ResumeCheckpoint &checkpoint);

    static void saveResumeCheckpoint(
            int simulation_step,
            const std::string &output_filename);

    static void saveMinimizerDiagnosticsToHDF5(
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
            bool converged);

    static void saveRawDataToBinaryFile(const std::string &filename, int t);

    static void saveGridDebug();

    static void saveLocationsToFile(const std::string &filename, int t);

    static void save_multi_array(const boost::multi_array<int, 3> &arr, const std::string &filename);

    static void saveThreeDimensionalVector(const std::vector<std::vector<std::vector<int>>> &myVector, const std::string &filename);

    static void saveArrayToFile(const std::array<double, 3> &data, const std::string &filename);

    template<class T>
    static void saveVectorToFile(const std::vector<std::array<T, 3>> &vectors, const std::string &filename) {
        std::ofstream outFile(filename);
        if (outFile.is_open()) {
            for (const auto &vec: vectors) {
                outFile << vec[0] << " " << vec[1] << " " << vec[2] << "\n";
            }
            outFile.close();
            std::cout << "Vector saved to " << filename << "\n";
        } else {
            std::cerr << "⚠ Error: Unable to open file " << filename << " for writing.\n";
        }
    }

    static void saveVectorOfArraysToJson(const std::vector<std::array<int, 3>> &data, const std::string &filename);

    static void saveVectorToFile(const std::vector<std::vector<int>> &data, const std::string &filename);

    static void saveVectorToJson(const std::vector<std::vector<int>> &data, const std::string &filename);

    static void logOverlapEvent(int timeStep, int agent_i, int agent_j, int type_i, int type_j, double distance, double interactionRadius);

    static std::string getCurrentTime();

    static nlohmann::json getSystemInfo();
};

#endif
