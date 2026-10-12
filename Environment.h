#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

#include <iostream>
#include <vector>
#include <boost/multi_array.hpp>
#include <array>
#include <fstream>
#include "nlohmann/json.hpp"
#include "GP.h"
#include <unordered_set>

class Environment {
public:

    typedef enum {
        CUBE,
        DATA,
        SPHERE,
        HEMISPHERE,
        CYLINDER
    } Shape;

    typedef enum {
        PRIMARY,
        SECONDARY
    } NeighborLevel;

    typedef enum {
        MAIN,
        SUB
    } NeighborType;

    Environment();

    void
    findIndexes(int label, std::vector<std::array<int, 3>> &indexes);
    boost::multi_array<float, 3> copyGrid() const;

    boost::multi_array<int, 3> grid;
    std::vector<std::array<int, 3>> new_coordinate;
    std::vector<std::vector<std::vector<int>>> neighborhoods;
    std::vector<std::array<int, 3>> index_inside;
    std::vector<std::array<int, 3>> index_outside;
    std::vector<std::array<int, 3>> index_boundary;
    static NeighborType nei;
    static Shape shape;
    static NeighborLevel neiLev;

    static int num_inside_cells;

private:

    int createShape(Shape s);
    void findNeighbors();
    void indexInside();
    void indexOutside();
    void indexBoundary();

    int x_Synapse;
    int x_size;
    int y_Synapse;
    int y_size;
    int z_Synapse;
    int z_size;

    int out;

    std::vector<int> labels_;

    float length_;
    float width_;
    float height_;

    std::vector<float> steps_;

    static std::vector<std::vector<int>> getNeighborOffsets(bool nei);
    static Environment::Shape convert_shape(int s);

    static Environment::NeighborLevel convert_neiborlevel(int s);

    static Environment::NeighborType convert_neibortype(int s);
};

#endif
