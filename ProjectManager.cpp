#include "ProjectManager.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <array>
#include <cctype>
#include <utility>
#include "GP.h"
#include "Vesicles.h"
#include "Synapsin.h"
#include "Interaction.h"
#include "Environment.h"
#include "Converter.h"

namespace fs = std::filesystem;
std::string ProjectManager::folder_name;
nlohmann::json ProjectManager::run_provenance;

namespace {

bool looksLikeProjectSource(const fs::path &directory) {
    std::error_code ec;
    return fs::is_directory(directory, ec)
           && fs::exists(directory / "CMakeLists.txt", ec)
           && fs::exists(directory / "ProjectManager.cpp", ec)
           && fs::exists(directory / "ProjectManager.h", ec);
}

fs::path searchUpwardForProjectSource(fs::path directory) {
    std::error_code ec;
    directory = fs::absolute(directory, ec);
    if (ec) return {};

    while (!directory.empty()) {
        if (looksLikeProjectSource(directory)) {
            return fs::weakly_canonical(directory, ec);
        }
        const fs::path parent = directory.parent_path();
        if (parent.empty() || parent == directory) break;
        directory = parent;
    }
    return {};
}

fs::path findProjectSourceDirectory(const fs::path &launch_directory) {

    const fs::path compiled_source_file = fs::path(__FILE__);
    if (compiled_source_file.is_absolute()
        && looksLikeProjectSource(compiled_source_file.parent_path())) {
        std::error_code ec;
        return fs::weakly_canonical(
                compiled_source_file.parent_path(), ec);
    }

    const fs::path discovered = searchUpwardForProjectSource(
            launch_directory);
    if (!discovered.empty()) return discovered;

    throw std::runtime_error(
            "Could not locate the project source directory. Expected an "
            "ancestor containing CMakeLists.txt, ProjectManager.cpp, and "
            "ProjectManager.h.");
}

bool isSourceSnapshotFile(const fs::path &path) {
    const std::string extension = path.extension().string();
    return extension == ".cpp" || extension == ".h"
           || extension == ".hpp" || extension == ".cc"
           || extension == ".cxx"
           || path.filename() == "CMakeLists.txt";
}

constexpr std::uint64_t FNV1A_OFFSET_BASIS = 14695981039346656037ULL;
constexpr std::uint64_t FNV1A_PRIME = 1099511628211ULL;

void updateFnv1a(
        std::uint64_t &hash,
        const char *data,
        std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<unsigned char>(data[i]);
        hash *= FNV1A_PRIME;
    }
}

std::uint64_t fnv1aFile(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(
                "Could not hash provenance file: "
                + path.generic_string());
    }

    std::uint64_t hash = FNV1A_OFFSET_BASIS;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            updateFnv1a(
                    hash, buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
                "Failed while hashing provenance file: "
                + path.generic_string());
    }
    return hash;
}

std::string uint64Hex(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

std::string trim(std::string value) {
    const auto first = std::find_if_not(
            value.begin(), value.end(), [](unsigned char c) {
                return std::isspace(c) != 0;
            });
    const auto last = std::find_if_not(
            value.rbegin(), value.rend(), [](unsigned char c) {
                return std::isspace(c) != 0;
            }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string sanitizeRunDescription(std::string value) {
    value = trim(std::move(value));
    for (char &character: value) {
        const unsigned char byte = static_cast<unsigned char>(character);
        if (!(std::isalnum(byte) != 0
              || character == '-'
              || character == '_'
              || character == '.')) {
            character = '_';
        }
    }
    if (value.empty() || value == "." || value == "..") {
        return "run";
    }
    return value;
}

std::string readFirstLine(const fs::path &path) {
    std::ifstream input(path);
    std::string line;
    if (input.is_open()) std::getline(input, line);
    return trim(line);
}

fs::path resolveGitDirectory(const fs::path &source_directory) {
    const fs::path marker = source_directory / ".git";
    if (fs::is_directory(marker)) return marker;
    if (!fs::is_regular_file(marker)) return {};

    const std::string marker_text = readFirstLine(marker);
    const std::string prefix = "gitdir:";
    if (marker_text.compare(0, prefix.size(), prefix) != 0) return {};
    fs::path git_directory = fs::path(trim(marker_text.substr(prefix.size())));
    if (git_directory.is_relative()) {
        git_directory = source_directory / git_directory;
    }
    std::error_code ec;
    return fs::weakly_canonical(git_directory, ec);
}

std::string shellQuote(const fs::path &path) {
    const std::string text = path.string();
#if defined(_WIN32)
    std::string quoted = "\"";
    for (char c: text) {
        if (c == '\"') quoted += "\"\"";
        else quoted += c;
    }
    quoted += "\"";
    return quoted;
#else
    std::string quoted = "'";
    for (char c: text) {
        if (c == '\'') quoted += "'\\''";
        else quoted += c;
    }
    quoted += "'";
    return quoted;
#endif
}

nlohmann::json collectGitMetadata(const fs::path &source_directory) {
    nlohmann::json git = {
            {"repository_detected", false},
            {"commit", "unknown"},
            {"branch", "unknown"},
            {"dirty_known", false},
            {"dirty", false},
            {"status_entry_count", 0}
    };

    const fs::path git_directory = resolveGitDirectory(source_directory);
    if (git_directory.empty() || !fs::is_directory(git_directory)) {
        return git;
    }
    git["repository_detected"] = true;

    const std::string head = readFirstLine(git_directory / "HEAD");
    const std::string reference_prefix = "ref:";
    if (head.compare(0, reference_prefix.size(), reference_prefix) == 0) {
        const std::string reference = trim(
                head.substr(reference_prefix.size()));
        const std::string branch_prefix = "refs/heads/";
        git["branch"] = reference.compare(
                0, branch_prefix.size(), branch_prefix) == 0
                        ? reference.substr(branch_prefix.size())
                        : reference;

        std::string commit = readFirstLine(git_directory / reference);
        if (commit.empty()) {
            std::ifstream packed_refs(git_directory / "packed-refs");
            std::string line;
            while (std::getline(packed_refs, line)) {
                if (line.empty() || line[0] == '#' || line[0] == '^') {
                    continue;
                }
                std::istringstream row(line);
                std::string packed_commit;
                std::string packed_reference;
                row >> packed_commit >> packed_reference;
                if (packed_reference == reference) {
                    commit = packed_commit;
                    break;
                }
            }
        }
        if (!commit.empty()) git["commit"] = commit;
    } else if (!head.empty()) {
        git["commit"] = head;
        git["branch"] = "DETACHED";
    }

    const fs::path status_file =
            fs::current_path() / ".git_status_porcelain.tmp";
    const std::string command =
            "git -C " + shellQuote(source_directory)
            + " status --porcelain=v1 --untracked-files=all > "
            + shellQuote(status_file) + " 2>&1";
    const int status_code = std::system(command.c_str());
    if (status_code == 0) {
        std::ifstream status_input(status_file);
        std::string line;
        int entry_count = 0;
        while (std::getline(status_input, line)) {
            if (!trim(line).empty()) ++entry_count;
        }
        git["dirty_known"] = true;
        git["dirty"] = entry_count > 0;
        git["status_entry_count"] = entry_count;
    }
    std::error_code remove_error;
    fs::remove(status_file, remove_error);
    return git;
}

std::string utcTimestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

nlohmann::json compilerMetadata() {
    std::string compiler_id = "unknown";
    std::string compiler_version = "unknown";
#if defined(__clang__)
    compiler_id = "Clang";
    compiler_version = __clang_version__;
#elif defined(__GNUC__)
    compiler_id = "GNU";
    compiler_version = __VERSION__;
#elif defined(_MSC_VER)
    compiler_id = "MSVC";
    compiler_version = std::to_string(_MSC_FULL_VER);
#endif

#if defined(NDEBUG)
    const bool release_build = true;
#else
    const bool release_build = false;
#endif

    return nlohmann::json{
            {"compiler_id", compiler_id},
            {"compiler_version", compiler_version},
            {"cplusplus", static_cast<std::int64_t>(__cplusplus)},
            {"release_build", release_build}
    };
}

std::string environmentMetadataValue(
        const char *name,
        const std::string &fallback = "unknown") {
    const char *raw_value = std::getenv(name);
    return raw_value == nullptr || raw_value[0] == '\0'
           ? fallback : std::string(raw_value);
}

std::int64_t environmentMetadataInteger(
        const char *name,
        std::int64_t fallback) {
    const std::string value = environmentMetadataValue(name, "");
    if (value.empty()) return fallback;
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed, 10);
        return consumed == value.size() ? parsed : fallback;
    } catch (...) {
        return fallback;
    }
}

bool environmentMetadataBoolean(const char *name, bool fallback) {
    const std::string value = environmentMetadataValue(name, "");
    if (value == "1" || value == "true" || value == "TRUE") return true;
    if (value == "0" || value == "false" || value == "FALSE") return false;
    return fallback;
}

nlohmann::json runtimeMetadata() {
    const std::string executable_sha256 = environmentMetadataValue(
            "SIMULATION_EXECUTABLE_SHA256");
    const bool executable_sha256_known =
            executable_sha256.size() == 64
            && std::all_of(
                    executable_sha256.begin(), executable_sha256.end(),
                    [](unsigned char character) {
                        return std::isxdigit(character) != 0;
                    });

    const std::int64_t available_processors = environmentMetadataInteger(
            "SIMULATION_OPENMP_AVAILABLE_PROCESSORS", 1);
    return nlohmann::json{
            {"openmp_enabled", environmentMetadataBoolean(
                    "SIMULATION_OPENMP_ENABLED", false)},
            {"logical_cores", environmentMetadataInteger(
                    "SIMULATION_DETECTED_LOGICAL_CORES",
                    available_processors)},
            {"openmp_available_processors", available_processors},
            {"detected_physical_cores", environmentMetadataInteger(
                    "SIMULATION_DETECTED_PHYSICAL_CORES", 0)},
            {"reserved_cores", environmentMetadataInteger(
                    "SIMULATION_RESERVED_CORES", 0)},
            {"openmp_selected_threads", environmentMetadataInteger(
                    "SIMULATION_OPENMP_SELECTED_THREADS", 1)},
            {"explicit_thread_override", environmentMetadataBoolean(
                    "SIMULATION_OMP_EXPLICIT_OVERRIDE", false)},
            {"thread_policy", environmentMetadataValue(
                    "SIMULATION_THREAD_POLICY", "unknown")},
            {"executable_sha256_known", executable_sha256_known},
            {"executable_sha256", executable_sha256}
    };
}

}

ProjectManager::ProjectManager(bool simple) {
    if(simple){
        initializeBuildDirectory();
    }else{
        GP::source_folder_name = findProjectSourceDirectory(
                fs::current_path());
    }

}

void ProjectManager::initializeBuildDirectory() {

    const fs::path launch_directory = fs::current_path();
    const fs::path source_directory = findProjectSourceDirectory(
            launch_directory);

    std::string timestamp = generateTimestamp();

    std::string description;
    std::cout << "Provide build information: " << std::endl;
    std::getline(std::cin, description);

    folder_name = description + "----" + timestamp;
    const fs::path relative_build_directory =
            fs::path("builds") / folder_name;
    GP::build_folder_name = relative_build_directory.generic_string();

    const fs::path build_dir = launch_directory / relative_build_directory;
    fs::create_directories(build_dir);
    fs::current_path(build_dir);

    copyAllFiles(source_directory);

    std::cout << "Project source directory: "
              << source_directory << std::endl;
    std::cout << "Run output directory: "
              << build_dir << std::endl;
}

void ProjectManager::initializeBuildDirectory(const std::string &scenarioDescription) {
    std::string timestamp = generateTimestamp();
    folder_name = sanitizeRunDescription(scenarioDescription) + "----" + timestamp;
    const fs::path relative_build_directory =
            fs::path("builds") / folder_name;
    GP::build_folder_name = relative_build_directory.generic_string();
    const std::filesystem::path source_path = GP::source_folder_name;
    const std::filesystem::path build_dir =
            std::filesystem::current_path() / relative_build_directory;
    std::filesystem::create_directories(build_dir);

    std::filesystem::current_path(build_dir);
    GP::source_folder_name = source_path;
    copyAllFiles(source_path);
    std::cout << "Project source directory: "
              << source_path << std::endl;
    std::cout << "Run output directory: "
              << build_dir << std::endl;
}

std::string ProjectManager::generateTimestamp() {

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&now_c);

    std::stringstream timestamp_ss;
    timestamp_ss << std::put_time(&local_tm, "%Y-%m-%d_%H-%M-%S");
    return timestamp_ss.str();
}

void ProjectManager::runNewBuild() {

    std::cout << "Saved in " << GP::build_folder_name << std::endl;
}

void ProjectManager::captureRunProvenance() {
    std::vector<fs::path> snapshot_files;
    for (const auto &entry: fs::directory_iterator(fs::current_path())) {
        if (entry.is_regular_file() && isSourceSnapshotFile(entry.path())) {
            snapshot_files.push_back(entry.path());
        }
    }
    std::sort(
            snapshot_files.begin(), snapshot_files.end(),
            [](const fs::path &left, const fs::path &right) {
                return left.filename().generic_string()
                       < right.filename().generic_string();
            });
    if (snapshot_files.empty()) {
        throw std::runtime_error(
                "Cannot capture provenance because the run source snapshot "
                "is empty");
    }

    nlohmann::json source_files = nlohmann::json::array();
    std::uint64_t aggregate_hash = FNV1A_OFFSET_BASIS;
    for (const fs::path &path: snapshot_files) {
        const std::string filename = path.filename().generic_string();
        const std::uint64_t file_hash = fnv1aFile(path);
        const std::string file_hash_hex = uint64Hex(file_hash);
        std::error_code size_error;
        const std::uintmax_t file_size = fs::file_size(path, size_error);
        if (size_error) {
            throw std::runtime_error(
                    "Could not read source snapshot file size: "
                    + path.generic_string());
        }

        updateFnv1a(aggregate_hash, filename.data(), filename.size());
        const char separator = '\0';
        updateFnv1a(aggregate_hash, &separator, 1);
        updateFnv1a(
                aggregate_hash,
                file_hash_hex.data(),
                file_hash_hex.size());

        source_files.push_back({
                {"file", filename},
                {"size_bytes", file_size},
                {"fnv1a64", file_hash},
                {"fnv1a64_hex", file_hash_hex}
        });
    }

    const fs::path dashboard_path =
            fs::current_path() / "Dashboard.input.json";
    if (!fs::is_regular_file(dashboard_path)) {
        throw std::runtime_error(
                "Dashboard.input.json is missing before provenance capture");
    }
    std::error_code dashboard_size_error;
    const std::uintmax_t dashboard_size =
            fs::file_size(dashboard_path, dashboard_size_error);
    if (dashboard_size_error) {
        throw std::runtime_error(
                "Could not read Dashboard.input.json size");
    }
    const std::uint64_t dashboard_hash = fnv1aFile(dashboard_path);

    run_provenance = {
            {"provenance_schema_version", 2},
            {"captured_utc", utcTimestamp()},
            {"hash_algorithm", "fnv1a64"},
            {"source_directory",
             GP::source_folder_name.generic_string()},
            {"run_directory",
             fs::current_path().generic_string()},
            {"source_snapshot", {
                    {"file_count", snapshot_files.size()},
                    {"aggregate_fnv1a64", aggregate_hash},
                    {"aggregate_fnv1a64_hex", uint64Hex(aggregate_hash)},
                    {"files", source_files}
            }},
            {"input_dashboard", {
                    {"file", "Dashboard.input.json"},
                    {"size_bytes", dashboard_size},
                    {"fnv1a64", dashboard_hash},
                    {"fnv1a64_hex", uint64Hex(dashboard_hash)}
            }},
            {"git", collectGitMetadata(GP::source_folder_name)},
            {"build", compilerMetadata()},
            {"runtime", runtimeMetadata()}
    };

    const fs::path final_path =
            fs::current_path() / "run_provenance.json";
    fs::path temporary_path = final_path;
    temporary_path += ".tmp";
    {
        std::ofstream output(
                temporary_path, std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            throw std::runtime_error(
                    "Could not create run_provenance.json temporary file");
        }
        output << run_provenance.dump(4);
        output.flush();
        if (!output.good()) {
            throw std::runtime_error(
                    "Could not finish run_provenance.json");
        }
    }

    std::error_code rename_error;
    if (fs::exists(final_path)) {
        fs::remove(final_path, rename_error);
        if (rename_error) {
            throw std::runtime_error(
                    "Could not replace existing run_provenance.json: "
                    + rename_error.message());
        }
    }
    fs::rename(temporary_path, final_path, rename_error);
    if (rename_error) {
        throw std::runtime_error(
                "Could not publish run_provenance.json: "
                + rename_error.message());
    }

    const auto &git = run_provenance.at("git");
    std::cout << "Run provenance captured: git="
              << git.value("commit", std::string("unknown"))
              << " dirty=";
    if (git.value("dirty_known", false)) {
        std::cout << std::boolalpha
                  << git.value("dirty", false)
                  << std::noboolalpha;
    } else {
        std::cout << "unknown";
    }
    std::cout << " source_hash="
              << run_provenance["source_snapshot"]
                         ["aggregate_fnv1a64_hex"].get<std::string>()
              << std::endl;
}

const nlohmann::json &ProjectManager::getRunProvenance() {
    return run_provenance;
}

void ProjectManager::clearAllData() {

    Vesicles::labels.clear();
    Vesicles::diffusion_coefficient.clear();
    Vesicles::active.clear();
    Vesicles::drag.clear();
    Vesicles::radii.clear();
    Vesicles::volume.clear();
    Vesicles::state.clear();
    Vesicles::positions.clear();
    Vesicles::locations.clear();
    Vesicles::vesicles_forces.clear();
    Vesicles::drift_scale.clear();
    Vesicles::total_forces.clear();
    Vesicles::which_vesicle_where.clear();
    Vesicles::capacity_of_vesicle.clear();
    Vesicles::inside.clear();
    Vesicles::net.resize(boost::extents[0][0][0]);
    Vesicles::net_vesicles.resize(boost::extents[0][0][0]);

    Synapsin::labels.clear();
    Synapsin::synapsin_forces.clear();
    Synapsin::positions.clear();
    Synapsin::color.clear();
    Synapsin::locations.clear();
    Synapsin::synapsin_forces.clear();
    Synapsin::drift_scale.clear();
    Synapsin::total_forces.clear();
    Synapsin::net.resize(boost::extents[0][0][0]);
    Synapsin::capacity_of_connection_to_vesicles.clear();
    Synapsin::labels.clear();
    Synapsin::inside.clear();
    Synapsin::which_synapsin_where.clear();

    GP::grid.resize(boost::extents[0][0][0]);
    GP::Dgrid.resize(boost::extents[0][0][0]);
    GP::neighborhoods.clear();
    GP::index_inside.clear();
    GP::index_outside.clear();
    GP::index_boundary.clear();
    GP::new_coordinate.clear();
    GP::total_force.clear();
    GP::labels_out_bound.clear();

    Interaction::synapsin_force_components.clear();
    Interaction::vesicle_force_components.clear();
    Interaction::last_overlap_projection_diagnostics = {};

    std::ofstream outputFile;
    if (outputFile.is_open()) {
        outputFile.close();
    }

}

void ProjectManager::copyAllFiles(const fs::path &source_directory) {
    GP::source_folder_name = source_directory;
    std::size_t copied_file_count = 0;

    for (const auto &entry: fs::directory_iterator(source_directory)) {

        if (entry.is_regular_file()) {
            if (isSourceSnapshotFile(entry.path())) {
                fs::copy_file(entry.path(),
                              fs::current_path() / entry.path().filename(),
                              fs::copy_options::overwrite_existing);
                ++copied_file_count;
            }
        }
    }

    if (copied_file_count == 0) {
        throw std::runtime_error(
                "No C++ source files were copied from the detected project "
                "source directory: " + source_directory.generic_string());
    }

    std::cout << "Source snapshot copied: "
              << copied_file_count << " source/build files" << std::endl;
}

void ProjectManager::copyDashboardFiles(
        const fs::path &source_directory,
        const fs::path &selected_dashboard) {

    for (const auto &entry: fs::directory_iterator(source_directory)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension();
            if (ext == ".json") {
                fs::copy(entry.path(),
                         fs::current_path() / entry.path().filename(),
                         fs::copy_options::overwrite_existing);
            }
        }
    }

    const fs::path selected_source = selected_dashboard.is_absolute()
                                     ? selected_dashboard
                                     : source_directory / selected_dashboard;
    if (!fs::is_regular_file(selected_source)) {
        throw std::runtime_error(
                "Selected Dashboard does not exist: "
                + selected_source.generic_string());
    }
    fs::copy_file(
            selected_source,
            fs::current_path() / "Dashboard.json",
            fs::copy_options::overwrite_existing);

    const fs::path dashboard = fs::current_path() / "Dashboard.json";
    if (!fs::is_regular_file(dashboard)) {
        throw std::runtime_error(
                "Dashboard.json was not copied into the run directory");
    }
    fs::copy_file(
            dashboard,
            fs::current_path() / "Dashboard.input.json",
            fs::copy_options::overwrite_existing);

    captureRunProvenance();
}
