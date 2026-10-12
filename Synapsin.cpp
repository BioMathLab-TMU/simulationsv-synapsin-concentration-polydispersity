#include "Synapsin.h"
#include <fstream>
#include <random>
#include <boost/multi_array.hpp>
#include "GP.h"
#include "GF.h"
#include "Interaction.h"
#include "MyUtility.h"
#include <algorithm>
#include <set>
#include <numbers>
#include <cmath>
#include <stdexcept>

std::vector<std::array<int, 3>> Synapsin::positions;
std::vector<std::array<float, 3>> Synapsin::locations;
std::vector<std::array<double, 3>> Synapsin::synapsin_forces;
std::vector<double> Synapsin::drift_scale;

boost::multi_array<int, 3> Synapsin::net;
std::vector<int> Synapsin::capacity_of_connection_to_vesicles;
std::vector<int> Synapsin::labels;
std::vector<std::array<int, 3>> Synapsin::inside;
std::vector<std::vector<int>> Synapsin::which_synapsin_where;
float Synapsin::diffusion_coefficient;
int Synapsin::number;
float Synapsin::radius;

std::vector<double>  Synapsin::total_forces;
std::vector<int> Synapsin::neighborhood_self;
std::vector<int> Synapsin::neighborhood_cross;
std::vector<GP::StochasticParameters> Synapsin::self_Ndistance;
std::vector<GP::StochasticParameters> Synapsin::cross_Ndistance;
std::vector<GP::StochasticParameters> Synapsin::self_Noverlaps;
std::vector<GP::StochasticParameters> Synapsin::cross_Noverlaps;
std::vector<GP::StochasticParameters> Synapsin::self_Nforces;
std::vector<GP::StochasticParameters> Synapsin::cross_Nforces;

std::vector<float> Synapsin::color;
float Synapsin::drag;
int Synapsin::_size_area;

Synapsin::Synapsin() : DiscreteAgent() {
    Synapsin::diffusion_coefficient =
            kb * T
            / (2 * dim * std::numbers::pi_v<double>
               * GP::synapsinI_radius * eta);
    loadFromJsonFile("SynapsinData.json");

    number = labels.size();
    inside = GP::index_inside;
    _size_area = inside.size();
    net.resize(boost::extents[x_size][y_size][z_size]);
    net = GP::grid;
    GF::zeroFill(synapsin_forces);
    whichSynapsinWhere();
    std::cout << Synapsin::number << " Synapsin created!" << std::endl;
}

Synapsin::Synapsin(float rad) : DiscreteAgent() {
    Synapsin::radius = GP::synapsinI_radius;
    Synapsin::number = GP::number_of_SynapsinI;
    float diffusion_control = GP::D_syn;
    float drag_control = GP::drag_syn;

    float h = 1.0 / number;
    Synapsin::diffusion_coefficient =
            kb * T * diffusion_control
            / (2 * dim * std::numbers::pi_v<double> * radius * eta);

    drag = 6 * std::numbers::pi_v<double> * eta * radius * drag_control;

    Synapsin::inside = GP::index_inside;
    _size_area = inside.size();
    net.resize(boost::extents[x_size][y_size][z_size]);
    Synapsin::net = GP::grid;

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
                    location, static_cast<double>(radius));
        }
        if (!accepted) {
            throw std::runtime_error(
                    "Could not initialize a synapsin inside the accessible domain");
        }

        color.push_back(i * h);
        locations.push_back(location);
        synapsin_forces.push_back({0.0, 0.0, 0.0});

        total_forces.push_back(0.0);
        neighborhood_cross.push_back(0);
        neighborhood_self.push_back(0);

        positions.push_back(cell_pos);

        labels.push_back(i);
        capacity_of_connection_to_vesicles.push_back(3);
    }

    whichSynapsinWhere();
    std::cout << Synapsin::number << " Synapsin created!" << std::endl;
}

void Synapsin::prepareDriftScales() {
    drift_scale.assign(static_cast<std::size_t>(number), 1.0);
    total_forces.resize(static_cast<std::size_t>(number), 0.0);

    const double time_step = static_cast<double>(GP::deltaT);
    const double mobility = 1.0 / std::max(
            static_cast<double>(drag), 1e-12);
    const double max_drift_nm =
            GP::dynamic_drift_limit_fraction_of_radius
            * static_cast<double>(radius);

    for (int i = 0; i < number; ++i) {
        const double force_norm =
                MyUtility::euclideanSize(synapsin_forces[i]);
        if (!std::isfinite(force_norm)) {
            throw std::runtime_error(
                    "Non-finite synapsin force before drift limiting");
        }
        total_forces[i] = force_norm;
        const double raw_drift_norm = force_norm * time_step * mobility;
        if (raw_drift_norm > max_drift_nm && raw_drift_norm > 0.0) {
            drift_scale[i] = max_drift_nm / raw_drift_norm;
        }
    }
}

void Synapsin::movementSynapsin() {
    auto t = GP::deltaT;
    double d = t / drag;
    double D = std::sqrt(2 * diffusion_coefficient * t);

    if (drift_scale.size() != static_cast<std::size_t>(number)) {
        prepareDriftScales();
    }

    std::normal_distribution<float> norm_dist(0, 1);

    for (int i = 0; i < number; ++i) {

        const double deterministic_scale = d * drift_scale[i];
        const std::array<double, 3> raw_drift = {
                synapsin_forces[i][0] * d,
                synapsin_forces[i][1] * d,
                synapsin_forces[i][2] * d
        };
        const std::array<double, 3> used_drift = {
                synapsin_forces[i][0] * deterministic_scale,
                synapsin_forces[i][1] * deterministic_scale,
                synapsin_forces[i][2] * deterministic_scale
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
                Agent::Synapsin_agent,
                raw_drift,
                used_drift,
                brownian,
                proposal,
                static_cast<double>(radius),
                D);

        total_forces[i] = MyUtility::euclideanSize(synapsin_forces[i]);

        GF::updateParticlePosition(i, proposal, Agent::Synapsin_agent);
    }

    whichSynapsinWhere();

}

void Synapsin::whichSynapsinWhere() {
    if (which_synapsin_where.size() != static_cast<std::size_t>(_size_area)) {
        which_synapsin_where.resize(_size_area);
    }
    for (auto &bucket: which_synapsin_where) {
        bucket.clear();
    }

    for (int i = 0; i < number; ++i) {
        int x = positions[i][0];
        int y = positions[i][1];
        int z = positions[i][2];
        int j = net[x][y][z];
        which_synapsin_where[j].push_back(i);
    }
}

void Synapsin::loadFromJsonFile(const std::string &filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        json data;
        file >> data;
        file.close();

        labels = data["Synapsin_labels"].get<std::vector<int>>();
        if (labels.size() <= 0) {
            return;
        } else {
            drag = data["Synapsin_drag"].get<float>();

            positions = data["Synapsin_positions"].get<std::vector<std::array<int, 3>>>();
            locations = data["Synapsin_locations"].get<std::vector<std::array<float, 3>>>();
            which_synapsin_where = data["Synapsin_which_vesicle_where"].get<std::vector<std::vector<int>>>();
            std::vector<int> stateInt = data["vesicles_state"].get<std::vector<int>>();
        }

        std::cout << "Data loaded from " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }
}
