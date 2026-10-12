#ifndef CODE_MYDISSERATION_CONVERTER_H
#define CODE_MYDISSERATION_CONVERTER_H

#include <iostream>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace fs = std::filesystem;

class Converter {
public:

    static double convertUnit(double value, const std::string &constant);

    static void saveConvertedUnits(const std::string &constant);
    static void setDashboardValue(const std::string& dotPath, const json& value);
    static int calculate_slide(double area, int shape);

    static const double Lu;
    static const double tu;
    static const double Mu;
    static const double chu;
};

#endif
