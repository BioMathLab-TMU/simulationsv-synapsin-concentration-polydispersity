#ifndef SIMULATIONSV_PROJECTION_NEIGHBORHOOD_H
#define SIMULATIONSV_PROJECTION_NEIGHBORHOOD_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ProjectionNeighborhood {

inline int requiredCellReach(
        double center_cutoff_nm,
        const std::vector<float> &cell_steps_nm) {
    if (!std::isfinite(center_cutoff_nm) || center_cutoff_nm < 0.0) {
        throw std::invalid_argument(
                "Projection center cutoff must be finite and non-negative");
    }
    if (cell_steps_nm.size() != 3) {
        throw std::invalid_argument(
                "Projection neighbor search requires three cell widths");
    }

    double minimum_cell_step_nm = std::numeric_limits<double>::infinity();
    for (const float step_nm: cell_steps_nm) {
        if (!std::isfinite(step_nm) || step_nm <= 0.0f) {
            throw std::invalid_argument(
                    "Projection cell widths must be finite and positive");
        }
        minimum_cell_step_nm = std::min(
                minimum_cell_step_nm,
                static_cast<double>(step_nm));
    }

    return static_cast<int>(std::ceil(
            center_cutoff_nm / minimum_cell_step_nm));
}

template<typename Visitor>
void visitCellCube(
        const std::array<int, 3> &home,
        int reach,
        const std::array<int, 3> &grid_size,
        const Visitor &visitor) {
    if (reach < 0) {
        throw std::invalid_argument(
                "Projection neighbor cell reach must be non-negative");
    }
    for (int dx = -reach; dx <= reach; ++dx) {
        const int x = home[0] + dx;
        if (x < 0 || x >= grid_size[0]) continue;
        for (int dy = -reach; dy <= reach; ++dy) {
            const int y = home[1] + dy;
            if (y < 0 || y >= grid_size[1]) continue;
            for (int dz = -reach; dz <= reach; ++dz) {
                const int z = home[2] + dz;
                if (z < 0 || z >= grid_size[2]) continue;
                visitor(std::array<int, 3>{x, y, z});
            }
        }
    }
}

}

#endif
