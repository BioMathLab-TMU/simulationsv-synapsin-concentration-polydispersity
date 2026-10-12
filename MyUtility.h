#ifndef MY_UTILITY_H
#define MY_UTILITY_H

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <numeric>
#include <limits>
#include <type_traits>
#include <boost/multi_array.hpp>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class MyUtility {
public:

    MyUtility() = delete;

    enum class OutputMode {
        Console,
        LogFile
    };

    static void setOutputMode(OutputMode mode);

    static void openLogFile(const std::string &filename);

    static void closeLogFile();

    static void logCheckpoint(const std::string &message, const char *file, const char *function, int line);

    static std::chrono::high_resolution_clock::time_point tic();

    static void toc(const std::chrono::high_resolution_clock::time_point &start_time, const std::string &functionName);

    template<typename... Args>
    static void print(const Args &... args);

    static std::vector<std::array<int, 3>> loadJsonToArray(const std::string &filename);

    static double euclideanSize(double x, double y, double z);

    static double euclideanSize(const std::array<double, 3> &A);

    template<typename T>
    static double calculateMean(const std::vector<T> &data);

    static void printVectorArray(const std::vector<std::array<int, 3>> &vecArr);

    static void printThreeDimensionalVectorRow(const std::vector<std::vector<int>> &row);

    static void printMultiArray(const boost::multi_array<int, 3> &arr);

    static void displayMultiArray(const boost::multi_array<double, 3> &array);

    static double meanDiffVesicles(const std::vector<float> &data);

    static void logCheckpoint(const std::string &message, const char *file, const char *function, int line,
                              const std::string &logFile);

private:

    static std::ofstream logFile;
    static OutputMode outputMode;

    static std::ostream &selectOutputStream();

    template<typename T>
    static std::enable_if_t<std::is_scalar_v<T> || std::is_same_v<T, const char *>> printValue(const T &value);

    static void printValue(const std::string &value);

    template<typename T, size_t N>
    static void printValue(const std::array<T, N> &arr);

    template<typename Container>
    static auto printValue(const Container &c) -> decltype(c.begin(), c.end(), void());
};

template<typename T>
std::enable_if_t<std::is_scalar_v<T> || std::is_same_v<T, const char *>> MyUtility::printValue(const T &value) {
    selectOutputStream() << value;
}

template<typename T, size_t N>
void MyUtility::printValue(const std::array<T, N> &arr) {
    selectOutputStream() << "{";
    for (size_t i = 0; i < arr.size(); ++i) {
        printValue(arr[i]);
        if (i < arr.size() - 1) {
            selectOutputStream() << ", ";
        }
    }
    selectOutputStream() << "}";
}

template<typename Container>
auto MyUtility::printValue(const Container &c) -> decltype(c.begin(), c.end(), void()) {
    selectOutputStream() << "[";
    for (auto it = c.begin(); it != c.end(); ++it) {
        printValue(*it);
        if (std::next(it) != c.end()) {
            selectOutputStream() << ", ";
        }
    }
    selectOutputStream() << "]";
}

template<typename... Args>
void MyUtility::print(const Args &... args) {
    ((printValue(args), selectOutputStream() << " "), ... );
    selectOutputStream() << std::endl;
}

template<typename T>
double MyUtility::calculateMean(const std::vector<T> &data) {
    if (data.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

#define LOG_CHECKPOINT(message) MyUtility::logCheckpoint(message, __FILE__, __FUNCTION__, __LINE__)

#endif
