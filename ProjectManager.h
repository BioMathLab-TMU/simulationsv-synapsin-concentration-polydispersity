#ifndef PROJECT_MANAGER_H
#define PROJECT_MANAGER_H

#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

class ProjectManager {
public:

    ProjectManager(bool simple = true);

    ProjectManager(const std::string &scenarioName, const std::filesystem::path &sourcePath);

    static void initializeBuildDirectory(const std::string &scenarioDescription);

    static void copyAllFiles(const std::filesystem::path &source_directory);

    static void copyDashboardFiles(
            const fs::path &source_directory,
            const fs::path &selected_dashboard = fs::path("Dashboard.json"));

    static void captureRunProvenance();

    static const nlohmann::json &getRunProvenance();

    static void runNewBuild();

    static void initializeBuildDirectory();

    static std::string folder_name;

    static void clearAllData();

private:

    static std::string generateTimestamp();

    static nlohmann::json run_provenance;
};

#endif
