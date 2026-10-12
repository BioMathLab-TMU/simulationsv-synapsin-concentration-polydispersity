#ifndef GP_CPP_GF_H
#define GP_CPP_GF_H

#include <cmath>
#include <fstream>
#include "Synapsin.h"
#include "GP.h"
#include "Vesicles.h"
#include "Interaction.h"
#include "MyUtility.h"
#include "Saver.h"
#include <cstdint>

class GF {

public:
    struct BoundaryDiagnostics {
        std::int64_t position_update_calls = 0;
        std::int64_t reflected_update_count = 0;
        std::int64_t reflected_axis_count = 0;
        std::int64_t rejected_update_count = 0;
        std::int64_t nonfinite_position_update_count = 0;
        double maximum_requested_displacement_nm = 0.0;
        double maximum_proposal_correction_nm = 0.0;
    };

    static BoundaryDiagnostics boundary_diagnostics;
    static bool boundary_diagnostics_enabled;
    static void resetBoundaryDiagnostics(bool enable = true);

    static std::vector<std::array<double, 3>> &getForces(Agent agent);
    static std::vector<double> &getTforce(Agent agent);

    static int getNumberAgent(Agent agent);

    static const std::vector<std::vector<int>>& getWhichWhere(Agent agent);

    static double getDrag(Agent agent, int i);

    static float getRadius(Agent agent, int i);

    static std::array<float, 3> getLocation(Agent agent, int i);

    static void updateParticlePosition(int i, const std::array<double, 3> &F, Agent objectType);

    static std::pair<std::array<int, 3>, std::array<float, 3>>
    checkBoundary(int i, std::array<int, 3> pos, std::array<float, 3> loc, Agent objectType);

    static bool isInsideAccessibleDomain(
            const std::array<float, 3> &loc,
            double particle_outer_radius_nm);

    static void
    adjustBoundary(std::array<int, 3> &pos, std::array<float, 3> &loc, const std::array<int, 3> &defaultPos,
                   const std::array<float, 3> &defaultLoc, int minPos, int maxPos);

    static void zeroFill(std::vector<std::array<double, 3>> &vec);

    static const std::vector<std::vector<int>>& neiborhoodList(int i, Agent objectType_i);

    static std::string getPrefix(Agent type_i, Agent type_j);

    static int getAgentTypeAsInt(Agent agentType);

};

#endif
