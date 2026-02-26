/***
Fast geodesic distance via multi-source Dijkstra on 2D lattices.
Exposed to Python using pybind11. Build with: python setup.py build_ext --inplace

Source: M. Campen, M. Heistermann, and L. Kobbelt, “Practical Anisotropic Geodesy,” Computer Graphics Forum, vol. 32, no. 5, pp. 63–71, Aug. 2013, doi: 10.1111/cgf.12173.

Implied by the source paper above as they mentioned that the standard Dijkstra can be useful for geodesic computation but suffers from jagged edges. This, in turn, is already accounted for by the blurring of edge respones. Additionally, the paper focuses on anisotropic geodesy, where the use case here is isotropic as defined by the cost map. This also ensures that the geodesic computation is as efficient and lightweight as possible due to the nature of being used in an interactive setting.
***/

#include <cmath>
#include <limits>
#include <queue>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Derived for diagnoal steps in 8-connected case sqrt(2) based on Pythagorean theorem for right triangles with equal legs of length 1.
static const double kDiagonalWeight = std::sqrt(2.0);

// Serve as Y and X offsets for 4-connected and 8-connected neighbor traversal.
static const int kDeltaY4[4] = {1, -1, 0, 0};
static const int kDeltaX4[4] = {0, 0, 1, -1};
static const int kDeltaY8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
static const int kDeltaX8[8] = {0, 0, 1, -1, 1, -1, 1, -1};

// Represents the weights of moving associated with each neighbor in 8-connected case, where diagonal moves are weighted by sqrt(2) and orthogonal moves are weighted by 1.0.
static const double kWeight8[8] = {1.0, 1.0, 1.0, 1.0, kDiagonalWeight, kDiagonalWeight, kDiagonalWeight, kDiagonalWeight};

/***
Dijkstra priority queue node: (y, x) coords and distance.
Min-heap ordered by dist (reversed comparison).
***/
struct Node
{
    int y, x;
    double dist;
    bool operator<(const Node &other) const { return dist > other.dist; }
};

/***
 * Multi-source Dijkstra to compute geodesic distances from seeds over a cost surface.
 * Edge weights are average costs between neighbors scaled by step length.
 ***/

py::array_t<double> Geodesic(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> seeds,
    bool eight_connected)
{
    auto cost_buf = cost.request(), seeds_buf = seeds.request();
    if (cost_buf.ndim != 2 || seeds_buf.ndim != 2)
        throw std::runtime_error("Inputs must be 2D");
    if (cost_buf.shape[0] != seeds_buf.shape[0] || cost_buf.shape[1] != seeds_buf.shape[1])
        throw std::runtime_error("Shapes mismatch");

    const int height = (int)cost_buf.shape[0], width = (int)cost_buf.shape[1];
    const double *cost_data = (double *)cost_buf.ptr;
    const uint8_t *seeds_data = (uint8_t *)seeds_buf.ptr;

    // Allocate distance map and initialize with infinity.
    py::array_t<double> output({height, width});
    double *dist_data = (double *)output.request().ptr;
    std::fill_n(dist_data, height * width, std::numeric_limits<double>::infinity());

    std::priority_queue<Node> pq;

    // Initialize priority queue with seed pixels (distance 0).
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (seeds_data[y * width + x])
            {
                dist_data[y * width + x] = 0.0;
                pq.push(Node{y, x, 0.0});
            }
        }
    }

    // Dijkstra main loop populating shortest paths.
    while (!pq.empty())
    {
        Node cur = pq.top();
        pq.pop();

        // When triggered, a shorter path to cur has already been found, so skip processing.
        if (cur.dist != dist_data[cur.y * width + cur.x])
            continue;

        const int num_nb = eight_connected ? 8 : 4;
        const int *dy = eight_connected ? kDeltaY8 : kDeltaY4;
        const int *dx = eight_connected ? kDeltaX8 : kDeltaX4;

        // Neighbor relaxation process.
        for (int k = 0; k < num_nb; ++k)
        {
            int ny = cur.y + dy[k], nx = cur.x + dx[k];
            if ((unsigned)ny < (unsigned)height && (unsigned)nx < (unsigned)width)
            {
                int n_idx = ny * width + nx, c_idx = cur.y * width + cur.x;

                double edge_w = (eight_connected ? kWeight8[k] : 1.0) * 0.5 * (cost_data[c_idx] + cost_data[n_idx]);
                double new_dist = cur.dist + edge_w;

                if (new_dist < dist_data[n_idx])
                {
                    dist_data[n_idx] = new_dist;
                    pq.push(Node{ny, nx, new_dist});
                }
            }
        }
    }
    return output;
}

// Export Geodesic distance function to 'fastgeo' Python module.
PYBIND11_MODULE(fastgeo, m)
{
    m.doc() = "Fast geodesic distance for lattice images.";
    m.def("geodesic", &Geodesic, py::arg("cost"), py::arg("seeds"), py::arg("eight_connected") = true, "Compute geodesic distance from seeds over cost surface.");
}
