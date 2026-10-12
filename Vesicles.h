#ifndef VESICLES_H
#define VESICLES_H

#include "DiscreteAgent.h"
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <boost/multi_array.hpp>
#include <nlohmann/json.hpp>
#include "GP.h"

struct Capacity {
    int SynapsinI;
    int Actin;
    int Clathrin;
};

class Vesicles : public DiscreteAgent {
private:

    enum Types {
        Endosome, Fill, Empty, OK
    };

    static Capacity _capacity_value;

    static int _size_area;

    static void loadFromJsonFile(const std::string &filename);

public:

    static std::vector<int> labels;

    static std::vector<float> diffusion_coefficient;
    static double D_ves_C , drag_ves_C;

    static std::vector<bool> active;

    static std::vector<std::array<int, 3>> positions;
    static std::vector<std::array<float, 3>> locations;

    static std::vector<float> core_radii;
    static std::vector<float> radii;
    static std::vector<float> volume;

    static std::vector<std::array<double, 3>> vesicles_forces;
    static std::vector<double> drift_scale;

    static std::vector<Types> state;

    static boost::multi_array<int, 3> net;
    static boost::multi_array<int, 3> net_vesicles;

    static std::vector<std::array<int, 3>> inside;

    static int number;

    static std::vector<Capacity> capacity_of_vesicle;

    static std::vector<float> drag;

    static std::vector<std::vector<int>> which_vesicle_where;

    static std::vector<double> total_forces;
    static std::vector<int> neighborhood_self;
    static std::vector<int> neighborhood_cross;
    static std::vector<GP::StochasticParameters> self_Ndistance;
    static std::vector<GP::StochasticParameters> cross_Ndistance;
    static std::vector<GP::StochasticParameters> self_Noverlaps;
    static std::vector<GP::StochasticParameters> cross_Noverlaps;
    static std::vector<GP::StochasticParameters> self_Nforces;
    static std::vector<GP::StochasticParameters> cross_Nforces;

    Vesicles();

    Vesicles(Capacity cap);

    static void movementVesicle();

    static void prepareDriftScales();

    static void whichVesiclesWhere();
};

#endif
