#include "GF.h"
#include "Synapsin.h"
#include <fstream>
#include "GP.h"
#include "Vesicles.h"
#include "Interaction.h"
#include "MyUtility.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

GF::BoundaryDiagnostics GF::boundary_diagnostics{};
bool GF::boundary_diagnostics_enabled = false;

void GF::resetBoundaryDiagnostics(bool enable) {
    boundary_diagnostics = {};
    boundary_diagnostics_enabled = enable;
}

std::vector<double> &GF::getTforce(Agent agent) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::total_forces;

        case Agent::Synapsin_agent:
            return Synapsin::total_forces;
        default:
            throw std::invalid_argument("Invalid Agent in initializeObject");
    }
}

std::vector<std::array<double, 3>> &GF::getForces(Agent agent) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::vesicles_forces;

        case Agent::Synapsin_agent:
            return Synapsin::synapsin_forces;
        default:
            throw std::invalid_argument("Invalid Agent in initializeObject");
    }
}

int GF::getNumberAgent(Agent agent) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::number;

        case Agent::Synapsin_agent:
            return Synapsin::number;
        default:
            throw std::invalid_argument("Invalid Agent in initializeObject");
    }
}

const std::vector<std::vector<int>>& GF::getWhichWhere(Agent agent) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::which_vesicle_where;

        case Agent::Synapsin_agent:
            return Synapsin::which_synapsin_where;

        default:
            throw std::invalid_argument("Invalid Agent in getWhichWhere");
    }
}

double GF::getDrag(Agent agent, int i) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::drag[i];

        case Agent::Synapsin_agent:
            return Synapsin::drag;
        default:
            throw std::invalid_argument("Invalid Agent in initializeObject");
    }
}

float GF::getRadius(Agent agent, int i) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::radii[i];

        case Agent::Synapsin_agent:
            return Synapsin::radius;
        default:
            throw std::invalid_argument("Invalid Agent in initializeObject");
    }
}

std::array<float, 3> GF::getLocation(Agent agent, int i) {
    switch (agent) {
        case Agent::Vesicle_agent:
            return Vesicles::locations[i];

        case Agent::Synapsin_agent:
            return Synapsin::locations[i];

        default:
            throw std::invalid_argument("Invalid Agent in initializeObject");
    }
}

void GF::updateParticlePosition(int i, const std::array<double, 3> &F, Agent objectType) {

    std::array<float, 3> loc;
    std::array<int, 3> pos;

    if (boundary_diagnostics_enabled) {
        ++boundary_diagnostics.position_update_calls;
    }
    if (!std::all_of(
            F.begin(), F.end(),
            [](double value) { return std::isfinite(value); })) {
        ++boundary_diagnostics.nonfinite_position_update_count;
        throw std::runtime_error("Non-finite particle displacement requested");
    }
    boundary_diagnostics.maximum_requested_displacement_nm = std::max(
            boundary_diagnostics.maximum_requested_displacement_nm,
            MyUtility::euclideanSize(F));

    if (objectType == Agent::Vesicle_agent) {
        loc[0] = F[0] + Vesicles::locations[i][0];
        loc[1] = F[1] + Vesicles::locations[i][1];
        loc[2] = F[2] + Vesicles::locations[i][2];
    } else if (objectType == Agent::Synapsin_agent) {
        loc[0] = F[0] + Synapsin::locations[i][0];
        loc[1] = F[1] + Synapsin::locations[i][1];
        loc[2] = F[2] + Synapsin::locations[i][2];
    }
    if (!std::all_of(
            loc.begin(), loc.end(),
            [](float value) { return std::isfinite(value); })) {
        ++boundary_diagnostics.nonfinite_position_update_count;
        throw std::runtime_error("Non-finite particle position proposed");
    }

    pos[0] = static_cast<int>(loc[0] / GP::steps[0]);
    pos[1] = static_cast<int>(loc[1] / GP::steps[1]);
    pos[2] = static_cast<int>(loc[2] / GP::steps[2]);

    auto checked_position = checkBoundary(i, pos, loc, objectType);

    if (objectType == Agent::Vesicle_agent) {
        double x = checked_position.second[0] - Vesicles::locations[i][0];
        double y = checked_position.second[1] - Vesicles::locations[i][1];
        double z = checked_position.second[2] - Vesicles::locations[i][2];

        Vesicles::locations[i] = checked_position.second;
        Vesicles::positions[i] = checked_position.first;
    } else if (objectType == Agent::Synapsin_agent) {
        double x = checked_position.second[0] - Synapsin::locations[i][0];
        double y = checked_position.second[1] - Synapsin::locations[i][1];
        double z = checked_position.second[2] - Synapsin::locations[i][2];

        Synapsin::locations[i] = checked_position.second;
        Synapsin::positions[i] = checked_position.first;
    }
}

std::pair<std::array<int, 3>, std::array<float, 3>>
GF::checkBoundary(int i, std::array<int, 3> pos, std::array<float, 3> loc, Agent objectType) {
    const int min_pos = GP::out + 1;
    const std::array<int, 3> max_pos = {
            GP::out + GP::x_Synapse,
            GP::out + GP::y_Synapse,
            GP::out + GP::z_Synapse
    };
    const std::array<double, 3> lower = {
            static_cast<double>(min_pos) * GP::steps[0],
            static_cast<double>(min_pos) * GP::steps[1],
            static_cast<double>(min_pos) * GP::steps[2]
    };
    const std::array<double, 3> upper = {
            static_cast<double>(max_pos[0] + 1) * GP::steps[0],
            static_cast<double>(max_pos[1] + 1) * GP::steps[1],
            static_cast<double>(max_pos[2] + 1) * GP::steps[2]
    };
    const double radius = static_cast<double>(getRadius(objectType, i));
    const auto proposed_loc = loc;
    bool reflected_update = false;

    const auto foldInterval = [&](double value, double lo, double hi) {
        if (!(hi > lo)) {
            throw std::runtime_error(
                    "Particle radius leaves no accessible boundary interval");
        }
        const double span = hi - lo;
        const double period = 2.0 * span;
        double folded = std::fmod(value - lo, period);
        if (folded < 0.0) folded += period;
        return folded <= span ? lo + folded : hi - (folded - span);
    };
    const auto reflectPlanarAxis = [&](int d, double lo, double hi) {
        const double value = static_cast<double>(loc[d]);
        if (value < lo || value > hi) {
            reflected_update = true;
            if (boundary_diagnostics_enabled) {
                ++boundary_diagnostics.reflected_axis_count;
            }
        }
        loc[d] = static_cast<float>(foldInterval(value, lo, hi));
    };
    const auto reflectRadially = [&](bool three_dimensional) {
        const double cx = 0.5 * (lower[0] + upper[0]);
        const double cy = 0.5 * (lower[1] + upper[1]);
        const double cz = GP::shape == 3
                          ? upper[2]
                          : 0.5 * (lower[2] + upper[2]);
        const double domain_radius = 0.5 * std::min(
                upper[0] - lower[0], upper[1] - lower[1]);
        const double accessible_radius = domain_radius - radius;
        if (!(accessible_radius > 0.0)) {
            throw std::runtime_error(
                    "Particle radius is not smaller than the curved domain");
        }
        double dx = static_cast<double>(loc[0]) - cx;
        double dy = static_cast<double>(loc[1]) - cy;
        double dz = three_dimensional
                    ? static_cast<double>(loc[2]) - cz
                    : 0.0;
        const double rho = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (rho <= accessible_radius) return;
        reflected_update = true;
        if (boundary_diagnostics_enabled) {
            ++boundary_diagnostics.reflected_axis_count;
        }
        const double reflected_rho = foldInterval(
                rho, 0.0, accessible_radius);
        const double scale = rho > 0.0 ? reflected_rho / rho : 0.0;
        loc[0] = static_cast<float>(cx + dx * scale);
        loc[1] = static_cast<float>(cy + dy * scale);
        if (three_dimensional) {
            loc[2] = static_cast<float>(cz + dz * scale);
        }
    };

    switch (GP::shape) {
        case 0:
            for (int d = 0; d < 3; ++d) {
                reflectPlanarAxis(d, lower[d] + radius, upper[d] - radius);
            }
            break;
        case 2:
            reflectRadially(true);
            break;
        case 3:
            reflectPlanarAxis(2, lower[2] + radius, upper[2] - radius);
            reflectRadially(true);

            reflectPlanarAxis(2, lower[2] + radius, upper[2] - radius);
            break;
        case 4:
            reflectPlanarAxis(2, lower[2] + radius, upper[2] - radius);
            reflectRadially(false);
            break;
        default:
            for (int d = 0; d < 3; ++d) {
                reflectPlanarAxis(d, lower[d] + radius, upper[d] - radius);
            }
            break;
    }

    if (!isInsideAccessibleDomain(loc, radius)) {
        throw std::runtime_error(
                "Analytic reflecting boundary failed to return an admissible point");
    }
    for (int d = 0; d < 3; ++d) {
        pos[d] = static_cast<int>(loc[d] / GP::steps[d]);
        pos[d] = std::clamp(pos[d], min_pos, max_pos[d]);
    }
    if (boundary_diagnostics_enabled && reflected_update) {
        ++boundary_diagnostics.reflected_update_count;
        const double correction = MyUtility::euclideanSize(
                static_cast<double>(loc[0] - proposed_loc[0]),
                static_cast<double>(loc[1] - proposed_loc[1]),
                static_cast<double>(loc[2] - proposed_loc[2]));
        boundary_diagnostics.maximum_proposal_correction_nm = std::max(
                boundary_diagnostics.maximum_proposal_correction_nm,
                correction);
    }
    return {pos, loc};
}

bool GF::isInsideAccessibleDomain(
        const std::array<float, 3> &loc,
        double particle_outer_radius_nm) {
    const int min_pos = GP::out + 1;
    const std::array<int, 3> max_pos = {
            GP::out + GP::x_Synapse,
            GP::out + GP::y_Synapse,
            GP::out + GP::z_Synapse
    };
    const std::array<double, 3> lower = {
            static_cast<double>(min_pos) * GP::steps[0],
            static_cast<double>(min_pos) * GP::steps[1],
            static_cast<double>(min_pos) * GP::steps[2]
    };
    const std::array<double, 3> upper = {
            static_cast<double>(max_pos[0] + 1) * GP::steps[0],
            static_cast<double>(max_pos[1] + 1) * GP::steps[1],
            static_cast<double>(max_pos[2] + 1) * GP::steps[2]
    };
    const double r = particle_outer_radius_nm;
    constexpr double tolerance_nm = 1e-4;
    if (!std::isfinite(r) || r < 0.0) return false;

    if (GP::shape == 0 || GP::shape == 1) {
        for (int d = 0; d < 3; ++d) {
            if (static_cast<double>(loc[d]) < lower[d] + r - tolerance_nm
                || static_cast<double>(loc[d])
                   > upper[d] - r + tolerance_nm) {
                return false;
            }
        }
        return true;
    }

    const double cx = 0.5 * (lower[0] + upper[0]);
    const double cy = 0.5 * (lower[1] + upper[1]);
    const double domain_radius = 0.5 * std::min(
            upper[0] - lower[0], upper[1] - lower[1]);
    const double accessible_radius = domain_radius - r;
    if (!(accessible_radius > 0.0)) return false;
    const double dx = static_cast<double>(loc[0]) - cx;
    const double dy = static_cast<double>(loc[1]) - cy;

    if (GP::shape == 4) {
        const double rho2 = dx * dx + dy * dy;

        const double tolerated_radius = accessible_radius + tolerance_nm;
        return rho2 <= tolerated_radius * tolerated_radius
               && static_cast<double>(loc[2])
                  >= lower[2] + r - tolerance_nm
               && static_cast<double>(loc[2])
                  <= upper[2] - r + tolerance_nm;
    }

    const double cz = GP::shape == 3
                      ? upper[2]
                      : 0.5 * (lower[2] + upper[2]);
    const double dz = static_cast<double>(loc[2]) - cz;
    const double tolerated_radius = accessible_radius + tolerance_nm;
    const bool within_sphere = dx * dx + dy * dy + dz * dz
                               <= tolerated_radius * tolerated_radius;
    if (GP::shape == 3) {
        return within_sphere
               && static_cast<double>(loc[2])
                  <= upper[2] - r + tolerance_nm;
    }
    return within_sphere;
}

void
GF::adjustBoundary(std::array<int, 3> &pos, std::array<float, 3> &loc, const std::array<int, 3> &defaultPos,
                   const std::array<float, 3> &defaultLoc, int minPos, int maxPos) {

    for (int d = 0; d < 3; ++d) {
        if (pos[d] < minPos || pos[d] > maxPos) {
            pos[d] = defaultPos[d];
            loc[d] = defaultLoc[d];
        }
    }
}

void GF::zeroFill(std::vector<std::array<double, 3>> &vec) {
    for (auto &arr: vec) {
        std::fill(arr.begin(), arr.end(), 0.0);
    }

}

const std::vector<std::vector<int>>&
GF::neiborhoodList(int i, Agent objectType_i)
{
    const auto& positions =
            (objectType_i == Agent::Vesicle_agent)
            ? Vesicles::positions
            : Synapsin::positions;

    int xi = positions[i][0];
    int yi = positions[i][1];
    int zi = positions[i][2];

    int home = GP::grid[xi][yi][zi];

    return GP::neighborhoods[home];
}

std::string GF::getPrefix(Agent type_i, Agent type_j) {
    if (type_i == type_j) {
        if (type_i == Agent::Vesicle_agent) {
            return "vesicle";
        } else if (type_i == Agent::Synapsin_agent) {
            return "synapsin";
        }
    }
    return "Double";
}

int GF::getAgentTypeAsInt(Agent agentType) {
    if (agentType == Agent::Synapsin_agent) {
        return 1;
    }
    if (agentType == Agent::Vesicle_agent) {
        return 2;
    }
    return 0;
}
