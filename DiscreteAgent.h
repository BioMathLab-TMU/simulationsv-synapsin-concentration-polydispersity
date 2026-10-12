#ifndef DISCRETE_AGENT_H
#define DISCRETE_AGENT_H

#include <vector>
#include <array>
#include <boost/multi_array.hpp>
#include "GP.h"

class DiscreteAgent {
protected:
    int _size_area;
    int number;
    boost::multi_array<int, 3> net;

    float kb;
    float T;
    int dim;
    float eta;

    int x_size;
    int y_size;
    int z_size;

    int min_val;
    int max_val;

public:
    DiscreteAgent();

    virtual ~DiscreteAgent() = default;
};

#endif
