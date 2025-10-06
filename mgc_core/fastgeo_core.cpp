// Filename: fastgeo_core.cpp
// Build: python setup.py build_ext --inplace
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <cmath>
#include <limits>

namespace py = pybind11;

struct Node {
    int y, x;
    double d;
    bool operator<(const Node& o) const { return d > o.d; } // min-heap
};

py::array_t<double> geodesic(py::array_t<double, py::array::c_style | py::array::forcecast> cost,
                             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> seeds,
                             bool eight_connected) {
    auto cbuf = cost.request();
    auto sbuf = seeds.request();
    if (cbuf.ndim != 2 || sbuf.ndim != 2 || cbuf.shape[0] != sbuf.shape[0] || cbuf.shape[1] != sbuf.shape[1]) {
        throw std::runtime_error("cost and seeds must be HxW");
    }
    const int H = static_cast<int>(cbuf.shape[0]);
    const int W = static_cast<int>(cbuf.shape[1]);
    const double* C = static_cast<double*>(cbuf.ptr);
    const uint8_t* S = static_cast<uint8_t*>(sbuf.ptr);

    py::array_t<double> out({H, W});
    auto obuf = out.request();
    double* D = static_cast<double*>(obuf.ptr);

    const double INF = std::numeric_limits<double>::infinity();
    for (int i = 0; i < H * W; ++i) D[i] = INF;

    std::priority_queue<Node> pq;

    auto push_seed = [&](int y, int x) {
        const int idx = y * W + x;
        D[idx] = 0.0;
        pq.push(Node{y, x, 0.0});
    };

    for (int y = 0; y < H; ++y) {
        const int row = y * W;
        for (int x = 0; x < W; ++x) {
            if (S[row + x] != 0) push_seed(y, x);
        }
    }

    const int dy4[4] = {1, -1, 0, 0};
    const int dx4[4] = {0, 0, 1, -1};
    const int dy8[8] = {1,-1,0,0, 1, 1,-1,-1};
    const int dx8[8] = {0,0,1,-1, 1,-1, 1,-1};
    const double w8[8] = {1,1,1,1, std::sqrt(2.0), std::sqrt(2.0), std::sqrt(2.0), std::sqrt(2.0)};

    while (!pq.empty()) {
        Node cur = pq.top(); pq.pop();
        const int idx = cur.y * W + cur.x;
        if (cur.d != D[idx]) continue;

        if (eight_connected) {
            for (int k = 0; k < 8; ++k) {
                int ny = cur.y + dy8[k], nx = cur.x + dx8[k];
                if ((unsigned)ny >= (unsigned)H || (unsigned)nx >= (unsigned)W) continue;
                const int nidx = ny * W + nx;
                const double step = 0.5 * (C[idx] + C[nidx]) * w8[k];
                const double nd = cur.d + step;
                if (nd < D[nidx]) { D[nidx] = nd; pq.push(Node{ny, nx, nd}); }
            }
        } else {
            for (int k = 0; k < 4; ++k) {
                int ny = cur.y + dy4[k], nx = cur.x + dx4[k];
                if ((unsigned)ny >= (unsigned)H || (unsigned)nx >= (unsigned)W) continue;
                const int nidx = ny * W + nx;
                const double step = 0.5 * (C[idx] + C[nidx]);
                const double nd = cur.d + step;
                if (nd < D[nidx]) { D[nidx] = nd; pq.push(Node{ny, nx, nd}); }
            }
        }
    }

    return out;
}

PYBIND11_MODULE(fastgeo, m) {
    m.doc() = "Fast geodesic distance (multi-source Dijkstra) for lattice images";
    m.def("geodesic", &geodesic, py::arg("cost"), py::arg("seeds"), py::arg("eight_connected")=true);
}
