#include "DiscreteAgent.h"
#include "GP.h"
#include <cmath>

#include "DiscreteAgent.h"

DiscreteAgent::DiscreteAgent()
        : kb(GP::kb), T(GP::T), dim(GP::dim), eta(GP::eta),
          x_size(GP::x_size), y_size(GP::y_size), z_size(GP::z_size),
          min_val((GP::out + 1) * GP::steps[0]),
          max_val((GP::out + GP::x_Synapse) * GP::steps[0]) {

}
