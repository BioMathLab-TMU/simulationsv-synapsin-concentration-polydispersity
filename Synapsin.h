#ifndef CODE_MYDISSERATION_SYNAPSIN_H
#define CODE_MYDISSERATION_SYNAPSIN_H

#include "DiscreteAgent.h"
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <boost/multi_array.hpp>
#include <nlohmann/json.hpp>
#include "GP.h"

class Synapsin : public DiscreteAgent {
private:

    static void loadFromJsonFile(const std::string &filename);

    double D_t;

    static int _size_area;

public:

    Synapsin(float rad);
    Synapsin();

    void movementSynapsin();

    static void prepareDriftScales();

    static void whichSynapsinWhere();

    static float diffusion_coefficient;

    static float drag;

    static float radius;

    static int number;

    static std::vector<float> color;

    static std::vector<int> labels;

    static std::vector<int> capacity_of_connection_to_vesicles;

    static std::vector<std::array<int, 3>> positions;

    static std::vector<std::array<float, 3>> locations;

    static std::vector<std::array<double, 3>> synapsin_forces;
    static std::vector<double> drift_scale;

    static boost::multi_array<int, 3> net;

    static std::vector<std::array<int, 3>> inside;

    static std::vector<std::vector<int>> which_synapsin_where;

    static std::vector<double> total_forces;
    static std::vector<int> neighborhood_self;
    static std::vector<int> neighborhood_cross;
    static std::vector<GP::StochasticParameters> self_Ndistance;
    static std::vector<GP::StochasticParameters> cross_Ndistance;
    static std::vector<GP::StochasticParameters> self_Noverlaps;
    static std::vector<GP::StochasticParameters> cross_Noverlaps;
    static std::vector<GP::StochasticParameters> self_Nforces;
    static std::vector<GP::StochasticParameters> cross_Nforces;

};

#endif
