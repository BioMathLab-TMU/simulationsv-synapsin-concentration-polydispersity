#include "MyUtility.h"
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

std::ofstream MyUtility::logFile;
MyUtility::OutputMode MyUtility::outputMode = MyUtility::OutputMode::Console;

void MyUtility::setOutputMode(OutputMode mode) {
    outputMode = mode;
}

void MyUtility::openLogFile(const std::string& filename) {
    try {
        fs::path logPath(filename);
        if (logPath.has_parent_path() && !fs::exists(logPath.parent_path())) {
            fs::create_directories(logPath.parent_path());
        }
        logFile.open(filename, std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Error: Unable to open log file: " << filename << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error while opening log file: " << e.what() << std::endl;
    }
}

void MyUtility::closeLogFile() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

std::ostream& MyUtility::selectOutputStream() {
    if (outputMode == OutputMode::LogFile && logFile.is_open()) {
        return logFile;
    }
    return std::cout;
}

void MyUtility::logCheckpoint(const std::string& message, const char* file, const char* function, int line) {
    std::ostream& out = selectOutputStream();
    out << "[CHECKPOINT] " << message
        << " (File: " << fs::path(file).filename().string()
        << ", Func: " << function
        << ", Line: " << line << ")" << std::endl;
}

std::chrono::high_resolution_clock::time_point MyUtility::tic() {
    return std::chrono::high_resolution_clock::now();
}

void MyUtility::toc(const std::chrono::high_resolution_clock::time_point& start_time, const std::string& functionName) {
    auto endTime = std::chrono::high_resolution_clock::now();
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - start_time).count();

    long long minutes = durationMs / 60000;
    long long seconds = (durationMs % 60000) / 1000;
    long long milliseconds = durationMs % 1000;

    std::ostream& out = selectOutputStream();
    out << "[TIMER] " << functionName << " took: "
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << seconds << "."
        << std::setw(3) << std::setfill('0') << milliseconds
        << " (MM:SS.mmm)" << std::endl;
}

void MyUtility::printValue(const std::string& value) {
    selectOutputStream() << '"' << value << '"';
}

std::vector<std::array<int, 3>> MyUtility::loadJsonToArray(const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open file for reading: " << filename << std::endl;
        return {};
    }

    json jsonData;
    try {
        inputFile >> jsonData;
    } catch (json::parse_error& e) {
        std::cerr << "JSON parse error in " << filename << ": " << e.what() << std::endl;
        return {};
    }

    std::vector<std::array<int, 3>> result;
    for (const auto& item : jsonData) {
        if (item.is_array() && item.size() == 3) {
            try {
                result.push_back({item.at(0).get<int>(), item.at(1).get<int>(), item.at(2).get<int>()});
            } catch (json::type_error& e) {
                std::cerr << "JSON type error while reading array: " << e.what() << std::endl;

            }
        }
    }
    return result;
}

double MyUtility::euclideanSize(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

double MyUtility::euclideanSize(const std::array<double, 3>& A) {
    return std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
}

void MyUtility::printVectorArray(const std::vector<std::array<int, 3>>& vecArr) {
    for (const auto& arr : vecArr) {

        MyUtility::print(arr[0], arr[1], arr[2]);
    }
}

void MyUtility::printThreeDimensionalVectorRow(const std::vector<std::vector<int>>& row) {
    for (const auto& subVector : row) {
        for (const auto& element : subVector) {
            std::cout << element << " ";
        }
        std::cout << "  ";
    }
    std::cout << std::endl;
}

void MyUtility::printMultiArray(const boost::multi_array<int, 3>& arr) {
    for (size_t i = 0; i < arr.shape()[0]; ++i) {
        for (size_t j = 0; j < arr.shape()[1]; ++j) {
            for (size_t k = 0; k < arr.shape()[2]; ++k) {
                std::cout << "(" << arr[i][j][k] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void MyUtility::displayMultiArray(const boost::multi_array<double, 3>& array) {
    for (size_t i = 0; i < array.shape()[0]; ++i) {
        for (size_t j = 0; j < array.shape()[1]; ++j) {
            for (size_t k = 0; k < array.shape()[2]; ++k) {
                std::cout << "array[" << i << "][" << j << "][" << k << "] = " << array[i][j][k] << std::endl;
            }
        }
    }
}

double MyUtility::meanDiffVesicles(const std::vector<float>& data) {
    return MyUtility::calculateMean(data);
}

void MyUtility::logCheckpoint(const std::string &message, const char *file, const char *function, int line,
                              const std::string &logFile) {
    std::string logMessage = message + " | File: " + std::string(file) +
                             " | Function: " + std::string(function) + " | Line: " + std::to_string(line);
    std::cout << logMessage << std::endl;

    if (!logFile.empty()) {
        std::ofstream outFile(logFile, std::ios::app);
        if (outFile.is_open()) {
            outFile << logMessage << std::endl;
            outFile.close();
        } else {
            std::cerr << "⚠ Error: Unable to open log file!" << std::endl;
        }
    }
}
