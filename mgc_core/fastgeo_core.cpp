// Copyright 2024 grab-cut contributors
// SPDX-License-Identifier: MIT
//
/// @file fastgeo_core.cpp
/// @brief Fast multi-source geodesic distance on 2D lattice images.
///
/// Implements Dijkstra's algorithm with a priority queue to compute geodesic
/// distances from multiple seed pixels over a cost surface. Exposed to Python
/// via pybind11.
///
/// Build:
///   python setup.py build_ext --inplace

// C++ standard library
#include <cmath>
#include <limits>
#include <queue>

// Third-party: pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Diagonal step weight for 8-connectivity (sqrt(2)).
static const double kDiagonalWeight = std::sqrt(2.0);

/// Y-offsets for 4-connectivity (down, up, right, left).
static const int kDeltaY4[4] = {1, -1, 0, 0};

/// X-offsets for 4-connectivity.
static const int kDeltaX4[4] = {0, 0, 1, -1};

/// Y-offsets for 8-connectivity (4-connected + diagonals).
static const int kDeltaY8[8] = {1, -1, 0, 0, 1, 1, -1, -1};

/// X-offsets for 8-connectivity.
static const int kDeltaX8[8] = {0, 0, 1, -1, 1, -1, 1, -1};

/// Edge weights for 8-connectivity (1.0 for cardinal, sqrt(2) for diagonal).
static const double kWeight8[8] = {
    1.0, 1.0, 1.0, 1.0,
    kDiagonalWeight, kDiagonalWeight, kDiagonalWeight, kDiagonalWeight};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// @brief Priority queue node for Dijkstra's algorithm.
///
/// Stores pixel coordinates (y, x) and tentative distance. Ordered by distance
/// ascending (min-heap via reversed comparison operator).
struct Node
{
    int y;       ///< Row coordinate.
    int x;       ///< Column coordinate.
    double dist; ///< Tentative geodesic distance.

    /// @brief Comparison operator for min-heap ordering.
    /// @param other Node to compare against.
    /// @return True if this node has greater distance (yields min-heap).
    bool operator<(const Node &other) const { return dist > other.dist; }
};

// ---------------------------------------------------------------------------
// Core Algorithm
// ---------------------------------------------------------------------------

/// @brief Compute geodesic distance from seed pixels over a cost surface.
///
/// Uses multi-source Dijkstra's algorithm on a 2D lattice. Edge weights are
/// computed as the average cost of adjacent pixels, scaled by step length
/// (1.0 for cardinal directions, sqrt(2) for diagonals).
///
/// @param cost     Cost map, shape (H, W), dtype float64. Non-negative values.
/// @param seeds    Seed mask, shape (H, W), dtype uint8. Non-zero = seed pixel.
/// @param eight_connected If true, use 8-connectivity; otherwise 4-connectivity.
/// @return Distance map, shape (H, W), dtype float64. Infinity for unreachable.
///
/// @throws std::runtime_error If cost and seeds shapes do not match or are not 2D.
py::array_t<double> Geodesic(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> seeds,
    bool eight_connected)
{
    // Request buffer info for input arrays.
    auto cost_buf = cost.request();
    auto seeds_buf = seeds.request();

    // Validate input dimensions.
    if (cost_buf.ndim != 2 || seeds_buf.ndim != 2)
    {
        throw std::runtime_error("cost and seeds must be 2D arrays");
    }
    if (cost_buf.shape[0] != seeds_buf.shape[0] ||
        cost_buf.shape[1] != seeds_buf.shape[1])
    {
        throw std::runtime_error("cost and seeds must have matching shapes");
    }

    const int height = static_cast<int>(cost_buf.shape[0]);
    const int width = static_cast<int>(cost_buf.shape[1]);
    const double *cost_data = static_cast<double *>(cost_buf.ptr);
    const uint8_t *seeds_data = static_cast<uint8_t *>(seeds_buf.ptr);

    // Allocate output distance array.
    py::array_t<double> output({height, width});
    auto out_buf = output.request();
    double *dist_data = static_cast<double *>(out_buf.ptr);

    // Initialize all distances to infinity.
    const double kInfinity = std::numeric_limits<double>::infinity();
    const int total_pixels = height * width;
    for (int i = 0; i < total_pixels; ++i)
    {
        dist_data[i] = kInfinity;
    }

    // Priority queue for Dijkstra traversal.
    std::priority_queue<Node> pq;

    // Lambda to enqueue a seed pixel with distance 0.
    auto enqueue_seed = [&](int y, int x)
    {
        const int idx = y * width + x;
        dist_data[idx] = 0.0;
        pq.push(Node{y, x, 0.0});
    };

    // Initialize queue with all seed pixels.
    for (int y = 0; y < height; ++y)
    {
        const int row_offset = y * width;
        for (int x = 0; x < width; ++x)
        {
            if (seeds_data[row_offset + x] != 0)
            {
                enqueue_seed(y, x);
            }
        }
    }

    // Dijkstra main loop.
    while (!pq.empty())
    {
        Node current = pq.top();
        pq.pop();

        const int current_idx = current.y * width + current.x;

        // Skip if we've already found a shorter path.
        if (current.dist != dist_data[current_idx])
        {
            continue;
        }

        // Select neighbour offsets based on connectivity.
        const int num_neighbours = eight_connected ? 8 : 4;
        const int *delta_y = eight_connected ? kDeltaY8 : kDeltaY4;
        const int *delta_x = eight_connected ? kDeltaX8 : kDeltaX4;

        // Explore neighbours.
        for (int k = 0; k < num_neighbours; ++k)
        {
            const int ny = current.y + delta_y[k];
            const int nx = current.x + delta_x[k];

            // Bounds check using unsigned comparison trick.
            if (static_cast<unsigned>(ny) >= static_cast<unsigned>(height) ||
                static_cast<unsigned>(nx) >= static_cast<unsigned>(width))
            {
                continue;
            }

            const int neighbour_idx = ny * width + nx;

            // Edge weight: average cost * step length.
            const double step_weight = eight_connected ? kWeight8[k] : 1.0;
            const double edge_cost =
                0.5 * (cost_data[current_idx] + cost_data[neighbour_idx]) * step_weight;
            const double new_dist = current.dist + edge_cost;

            // Relaxation step.
            if (new_dist < dist_data[neighbour_idx])
            {
                dist_data[neighbour_idx] = new_dist;
                pq.push(Node{ny, nx, new_dist});
            }
        }
    }

    return output;
}

// ---------------------------------------------------------------------------
// Python Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(fastgeo, m)
{
    m.doc() = "Fast geodesic distance (multi-source Dijkstra) for lattice images.";

    m.def("geodesic", &Geodesic,
          py::arg("cost"),
          py::arg("seeds"),
          py::arg("eight_connected") = true,
          R"doc(
Compute geodesic distance from seed pixels over a cost surface.

Args:
    cost: Cost map, shape (H, W), dtype float64. Must be non-negative.
    seeds: Seed mask, shape (H, W), dtype uint8. Non-zero values mark seeds.
    eight_connected: If True, use 8-connectivity; otherwise 4-connectivity.

Returns:
    Distance map, shape (H, W), dtype float64. Unreachable pixels have inf.

Raises:
    RuntimeError: If inputs are not 2D or have mismatched shapes.
)doc");
}
