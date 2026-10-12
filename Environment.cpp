#include <iostream>
#include <vector>
#include <boost/multi_array.hpp>
#include <array>
#include <string>
#include <cmath>
#include "nlohmann/json.hpp"
#include "Environment.h"
#include "MyUtility.h"
#include "GP.h"
#include <set>
#include <stdexcept>

Environment::NeighborType Environment::nei;
Environment::Shape Environment::shape;
Environment::NeighborLevel Environment::neiLev;
int Environment::num_inside_cells = 0;

struct MyHash {
    std::size_t operator()(const std::array<int, 3> &arr) const {
        return std::hash<int>()(arr[0]) ^ std::hash<int>()(arr[1]) ^ std::hash<int>()(arr[2]);
    }
};

Environment::Environment() {

    length_ = GP::length;
    width_ = GP::width;
    height_ = GP::height;
    steps_ = GP::steps;
    labels_ = GP::labels_out_bound;
    out = GP::out;
    shape = Environment::convert_shape(GP::shape);
    nei = Environment::convert_neibortype(GP::nei);
    neiLev = Environment::convert_neiborlevel(GP::neiLev);

    x_Synapse = static_cast<int>(std::ceil(length_ / steps_[0] - 1e-9));
    y_Synapse = static_cast<int>(std::ceil(width_ / steps_[1] - 1e-9));
    z_Synapse = static_cast<int>(std::ceil(height_ / steps_[2] - 1e-9));

    GP::x_Synapse = x_Synapse;
    GP::y_Synapse = y_Synapse;
    GP::z_Synapse = z_Synapse;

    x_size = x_Synapse + 2 * (out + 1);
    y_size = y_Synapse + 2 * (out + 1);
    z_size = z_Synapse + 2 * (out + 1);

    GP::x_size = x_size;
    GP::y_size = y_size;
    GP::z_size = z_size;

    grid.resize(boost::extents[x_size][y_size][z_size]);
    std::fill(grid.data(), grid.data() + grid.num_elements(), labels_[0]);

    num_inside_cells = createShape(shape);

    indexInside();

    indexOutside();

    indexBoundary();

    findNeighbors();

    std::cout << "Environment created!" << std::endl;
}

boost::multi_array<float, 3> Environment::copyGrid() const {
    boost::multi_array<float, 3> gridCopy(boost::extents[x_size][y_size][z_size]);

    for (int i = 0; i < x_size; ++i) {
        for (int j = 0; j < y_size; ++j) {
            for (int k = 0; k < z_size; ++k) {
                gridCopy[i][j][k] = static_cast<float>(grid[i][j][k]);
            }
        }
    }

    int _size_area = index_inside.size();
    for (int i = 0; i < _size_area; ++i) {
        int x = index_inside[i][0];
        int y = index_inside[i][1];
        int z = index_inside[i][2];
        gridCopy[x][y][z] = 0.0;
    }

    return gridCopy;
}

void Environment::findIndexes(int label, std::vector<std::array<int, 3>> &indexes) {
    for (int k = 0; k < z_size; ++k) {
        for (int j = 0; j < y_size; ++j) {
            for (int i = 0; i < x_size; ++i) {
                if (grid[i][j][k] == label) {
                    indexes.push_back({i, j, k});
                }
            }
        }
    }
}

int Environment::createShape(Shape s) {
    if (s == DATA) {
        throw std::invalid_argument(
                "DATA geometry requires an explicit imported domain");
    }
    if (s < CUBE || s > CYLINDER) {
        throw std::invalid_argument("Unknown environment shape");
    }

    int index = 0;
    const int begin = out + 1;
    const int end_x = out + x_Synapse;
    const int end_y = out + y_Synapse;
    const int end_z = out + z_Synapse;
    for (int i = begin; i <= end_x; ++i) {
        for (int j = begin; j <= end_y; ++j) {
            for (int k = begin; k <= end_z; ++k) {
                grid[i][j][k] = index++;
            }
        }
    }
    std::cout << "Created analytic shape " << static_cast<int>(s)
              << " on a rectangular linked-cell grid" << std::endl;
    return index;
}

void Environment::findNeighbors() {
    int numRo = index_inside.size();
    neighborhoods.resize(numRo);

#pragma omp parallel for

    for (int rowNum = 0; rowNum < numRo; ++rowNum) {
        int i = index_inside[rowNum][0];
        int j = index_inside[rowNum][1];
        int k = index_inside[rowNum][2];

        std::unordered_set<std::array<int, 3>, MyHash> uniqueNeighbors;

        uniqueNeighbors.insert({i, j, k});

        std::vector<std::vector<int>> neighborOffsets = getNeighborOffsets(nei == SUB);

        for (const auto &xyz: neighborOffsets) {
            int x = i + xyz[0];
            int y = j + xyz[1];
            int z = k + xyz[2];

            if (grid[x][y][z] >= 0) {
                uniqueNeighbors.insert({x, y, z});

                if (neiLev == SECONDARY) {
                    for (const auto &sub_xyz: neighborOffsets) {
                        int sub_x = x + sub_xyz[0];
                        int sub_y = y + sub_xyz[1];
                        int sub_z = z + sub_xyz[2];

                        if (grid[sub_x][sub_y][sub_z] >= 0) {
                            uniqueNeighbors.insert({sub_x, sub_y, sub_z});
                        }
                    }
                }
            }
        }

        neighborhoods[rowNum].clear();
        for (const auto &neighbor: uniqueNeighbors) {
            neighborhoods[rowNum].push_back({neighbor[0], neighbor[1], neighbor[2]});
        }
    }
}

void Environment::indexOutside() {
    findIndexes(labels_[0], index_outside);
}

void Environment::indexBoundary() {
    findIndexes(labels_[1], index_boundary);
}

void Environment::indexInside() {
    index_inside.resize(num_inside_cells);
    for (int k = 0; k < z_size; ++k) {
        for (int j = 0; j < y_size; ++j) {
            for (int i = 0; i < x_size; ++i) {
                if (Environment::grid[i][j][k] >= 0) {
                    std::array<int, 3> temp_array = {i, j, k};
                    index_inside[Environment::grid[i][j][k]] = temp_array;
                }
            }
        }
    }
}

std::vector<std::vector<int>> Environment::getNeighborOffsets(bool nei) {
    std::vector<std::vector<int>> neighborOffsets;

    if (nei) {
        neighborOffsets = {
                {-1, -1, -1},
                {-1, -1, 0},
                {-1, -1, 1},
                {-1, 0,  -1},
                {-1, 0,  0},
                {-1, 0,  1},
                {-1, 1,  -1},
                {-1, 1,  0},
                {-1, 1,  1},
                {0,  -1, -1},
                {0,  -1, 0},
                {0,  -1, 1},
                {0,  0,  -1},
                {0,  0,  1},
                {0,  1,  -1},
                {0,  1,  0},
                {0,  1,  1},
                {1,  -1, -1},
                {1,  -1, 0},
                {1,  -1, 1},
                {1,  0,  -1},
                {1,  0,  0},
                {1,  0,  1},
                {1,  1,  -1},
                {1,  1,  0},
                {1,  1,  1}
        };
    } else {
        neighborOffsets = {{-1, 0,  0},
                           {0,  -1, 0},
                           {0,  0,  -1},
                           {0,  0,  1},
                           {0,  1,  0},
                           {1,  0,  0}};
    }

    return neighborOffsets;
}

Environment::Shape Environment::convert_shape(int s) {
    switch (s) {
        case 0:
            return Shape::CUBE;
        case 1:
            return Shape::DATA;
        case 2:
            return Shape::SPHERE;
        case 3:
            return Shape::HEMISPHERE;
        case 4:
            return Shape::CYLINDER;
        default:
            std::cerr << "Unknown shape: " << s << std::endl;
            return Shape::CUBE;
    }
}

Environment::NeighborType Environment::convert_neibortype(int s) {
    switch (s) {
        case 0:
            return NeighborType::MAIN;
        case 1:
            return NeighborType::SUB;
        default:
            std::cerr << "Unknown neighbor type: " << s << std::endl;
            return NeighborType::MAIN;
    }
}

Environment::NeighborLevel Environment::convert_neiborlevel(int s) {
    switch (s) {
        case 0:
            return NeighborLevel::PRIMARY;
        case 1:
            return NeighborLevel::SECONDARY;
        default:
            std::cerr << "Unknown neighbor level: " << s << std::endl;
            return NeighborLevel::PRIMARY;
    }
}
