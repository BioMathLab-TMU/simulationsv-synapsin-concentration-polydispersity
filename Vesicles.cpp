#include "Vesicles.h"
#include <fstream>
#include <random>
#include <boost/multi_array.hpp>
#include <cstdlib>
#include <ctime>
#include "GP.h"
#include "GF.h"
#include "Interaction.h"
#include "MyUtility.h"
#include "Saver.h"
#include <numbers>
#include <cmath>
#include <stdexcept>

using json = nlohmann::json;

std::vector<int> Vesicles::labels;
std::vector<float> Vesicles::diffusion_coefficient;
double Vesicles::D_ves_C = 0, Vesicles::drag_ves_C = 0;
std::vector<bool> Vesicles::active;
std::vector<std::array<int, 3>> Vesicles::positions;
std::vector<float> Vesicles::drag;
std::vector<std::array<float, 3>> Vesicles::locations;
std::vector<float> Vesicles::core_radii;
std::vector<float> Vesicles::radii;
std::vector<Vesicles::Types> Vesicles::state;
boost::multi_array<int, 3> Vesicles::net;
boost::multi_array<int, 3> Vesicles::net_vesicles;
std::vector<std::array<int, 3>> Vesicles::inside;
int Vesicles::number;
Capacity Vesicles::_capacity_value;
std::vector<Capacity> Vesicles::capacity_of_vesicle;
std::vector<std::vector<int>> Vesicles::which_vesicle_where;
std::vector<float> Vesicles::volume;

int Vesicles::_size_area;

std::vector<int> Vesicles::neighborhood_self;
std::vector<int> Vesicles::neighborhood_cross;
std::vector<GP::StochasticParameters> Vesicles::self_Ndistance;
std::vector<GP::StochasticParameters> Vesicles::cross_Ndistance;
std::vector<GP::StochasticParameters> Vesicles::self_Noverlaps;
std::vector<GP::StochasticParameters> Vesicles::cross_Noverlaps;
std::vector<GP::StochasticParameters> Vesicles::self_Nforces;
std::vector<GP::StochasticParameters> Vesicles::cross_Nforces;

std::vector<double> Vesicles::total_forces;

std::vector<std::array<double, 3>> Vesicles::vesicles_forces;
std::vector<double> Vesicles::drift_scale;

Vesicles::Vesicles() : DiscreteAgent() {
    loadFromJsonFile("VesicleData.json");
    number = labels.size();
    inside = GP::index_inside;
    _size_area = inside.size();
    net.resize(boost::extents[x_size][y_size][z_size]);
    net = GP::grid;
    GF::zeroFill(vesicles_forces);
    whichVesiclesWhere();
    std::cout << number << " vesicles created!" << std::endl;
}

Vesicles::Vesicles(Capacity cap) : DiscreteAgent() {
    net_vesicles.resize(boost::extents[x_size][y_size][z_size]);
    net_vesicles = GP::grid;
    number = GP::number_of_vesicles;
    inside = GP::index_inside;

    _size_area = inside.size();
    float drag_control = GP::drag_ves;
    float diffusion_control = GP::D_ves;
    net.resize(boost::extents[x_size][y_size][z_size]);
    net = GP::grid;

    std::normal_distribution<float> radius_dist(
            GP::mean_radius, GP::std_radius > 0.0F ? GP::std_radius : 1.0F);

    std::uniform_int_distribution<int> cell_dist(0, inside.size() - 1);

    std::uniform_real_distribution<float> offset_x(0, GP::steps[0]);
    std::uniform_real_distribution<float> offset_y(0, GP::steps[1]);
    std::uniform_real_distribution<float> offset_z(0, GP::steps[2]);

    self_Nforces.resize(number);
    self_Ndistance.resize(number);
    self_Noverlaps.resize(number);
    cross_Nforces.resize(number);
    cross_Ndistance.resize(number);
    cross_Noverlaps.resize(number);

    for (int i = 0; i < number; ++i) {

        float core_radius = GP::mean_radius;
        if (GP::std_radius > 0.0F) {
            do {
                core_radius = radius_dist(GP::gen);
            } while (core_radius < GP::vesicle_core_radius_min_nm
                     || core_radius > GP::vesicle_core_radius_max_nm);
        }
        const float outer_radius = static_cast<float>(
                static_cast<double>(core_radius)
                + GP::vesicle_corona_thickness_nm);
        core_radii.push_back(core_radius);
        radii.push_back(outer_radius);

        std::array<int, 3> cell_pos{};
        std::array<float, 3> location{};
        bool accepted = false;
        constexpr int maximum_initialization_attempts = 100000;
        for (int attempt = 0;
             attempt < maximum_initialization_attempts && !accepted;
             ++attempt) {
            const int random_cell_idx = cell_dist(GP::gen);
            cell_pos = inside[random_cell_idx];
            location = {
                    cell_pos[0] * GP::steps[0] + offset_x(GP::gen),
                    cell_pos[1] * GP::steps[1] + offset_y(GP::gen),
                    cell_pos[2] * GP::steps[2] + offset_z(GP::gen)
            };
            accepted = GF::isInsideAccessibleDomain(
                    location, static_cast<double>(outer_radius));
        }
        if (!accepted) {
            throw std::runtime_error(
                    "Could not initialize a vesicle inside the accessible domain");
        }
        locations.push_back(location);
        positions.push_back(cell_pos);

        labels.push_back(i);
        capacity_of_vesicle.push_back({0, 0, 0});
        active.push_back(true);

        total_forces.push_back(0.0);
        neighborhood_self.push_back(0);
        neighborhood_cross.push_back(0);

        volume.push_back(
                4 * std::numbers::pi_v<double>
                * core_radii.back() * core_radii.back()
                * core_radii.back() / 3);
        drag.push_back(
                6 * std::numbers::pi_v<double>
                * eta * radii.back() * drag_control);
        diffusion_coefficient.push_back(
                kb * T * diffusion_control
                / (2 * dim * std::numbers::pi_v<double>
                   * radii.back() * eta));
        D_ves_C += diffusion_coefficient[i];
        drag_ves_C += drag[i];
        state.push_back(Vesicles::OK);
        vesicles_forces.push_back({0, 0, 0});
    }

    D_ves_C /= number;
    drag_ves_C /= number;

    whichVesiclesWhere();
    std::cout << number << " vesicles created!" << std::endl;
}

void Vesicles::prepareDriftScales() {
    drift_scale.assign(static_cast<std::size_t>(number), 1.0);
    total_forces.resize(static_cast<std::size_t>(number), 0.0);

    const double time_step = static_cast<double>(GP::deltaT);
    for (int i = 0; i < number; ++i) {
        const double force_norm =
                MyUtility::euclideanSize(vesicles_forces[i]);
        if (!std::isfinite(force_norm)) {
            throw std::runtime_error(
                    "Non-finite vesicle force before drift limiting");
        }
        total_forces[i] = force_norm;
        const double mobility = 1.0 / std::max(
                static_cast<double>(drag[i]), 1e-12);
        const double raw_drift_norm = force_norm * time_step * mobility;
        const double max_drift_nm =
                GP::dynamic_drift_limit_fraction_of_radius
                * static_cast<double>(radii[i]);
        if (raw_drift_norm > max_drift_nm && raw_drift_norm > 0.0) {
            drift_scale[i] = max_drift_nm / raw_drift_norm;
        }
    }
}

void Vesicles::movementVesicle() {
    std::normal_distribution<float> norm_dist(0, 1.0);
    auto t = GP::deltaT;

    if (drift_scale.size() != static_cast<std::size_t>(number)) {
        prepareDriftScales();
    }

    for (int i = 0; i < number; ++i) {

        double D = std::sqrt(2 * diffusion_coefficient[i] * t);
        double d = t / drag[i];

        const double deterministic_scale = d * drift_scale[i];
        const std::array<double, 3> raw_drift = {
                vesicles_forces[i][0] * d,
                vesicles_forces[i][1] * d,
                vesicles_forces[i][2] * d
        };
        const std::array<double, 3> used_drift = {
                vesicles_forces[i][0] * deterministic_scale,
                vesicles_forces[i][1] * deterministic_scale,
                vesicles_forces[i][2] * deterministic_scale
        };
        const std::array<double, 3> brownian = {
                D * norm_dist(GP::gen),
                D * norm_dist(GP::gen),
                D * norm_dist(GP::gen)
        };
        const std::array<double, 3> proposal = {
                used_drift[0] + brownian[0],
                used_drift[1] + brownian[1],
                used_drift[2] + brownian[2]
        };
        Interaction::recordEulerParticleProposal(
                Agent::Vesicle_agent,
                raw_drift,
                used_drift,
                brownian,
                proposal,
                static_cast<double>(radii[i]),
                D);

        total_forces[i] = MyUtility::euclideanSize(vesicles_forces[i]);

        GF::updateParticlePosition(i, proposal, Agent::Vesicle_agent);
    }

    whichVesiclesWhere();

}

void Vesicles::loadFromJsonFile(const std::string &filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        json data;
        file >> data;
        file.close();

        labels = data["Vesicles_labels"].get<std::vector<int>>();
        diffusion_coefficient = data["vesicles_diffusion_coefficient"].get<std::vector<float>>();
        drag = data["vesicles_drag"].get<std::vector<float>>();
        active = data["vesicles_active"].get<std::vector<bool>>();
        positions = data["vesicles_positions"].get<std::vector<std::array<int, 3>>>();
        locations = data["vesicles_locations"].get<std::vector<std::array<float, 3>>>();
        radii = data["vesicles_radii"].get<std::vector<float>>();
        if (data.contains("vesicles_core_radii")) {
            core_radii = data["vesicles_core_radii"].get<std::vector<float>>();
        } else if (data.contains("core_radii")) {
            core_radii = data["core_radii"].get<std::vector<float>>();
        } else {
            core_radii = radii;
        }
        which_vesicle_where = data["vesicles_which_vesicle_where"].get<std::vector<std::vector<int>>>();
        std::vector<int> stateInt = data["vesicles_state"].get<std::vector<int>>();
        for (int stateValue: stateInt) {
            state.push_back(static_cast<Types>(stateValue));
        }

        std::cout << "Data loaded from " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }
}

void Vesicles::whichVesiclesWhere() {
    if (which_vesicle_where.size() != static_cast<std::size_t>(_size_area)) {
        which_vesicle_where.resize(_size_area);
    }
    for (auto &bucket: which_vesicle_where) {
        bucket.clear();
    }

    for (int i = 0; i < number; ++i) {
        int x = positions[i][0];
        int y = positions[i][1];
        int z = positions[i][2];
        int j = net[x][y][z];
        which_vesicle_where[j].push_back(i);
    }
}
