/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

using namespace gridtools;

// lap-begin
auto laplacian = [](auto &lap, auto &in, int boundary_size) {
    auto lengths = in.lengths();
    int Ni = lengths[0];
    int Nj = lengths[1];
    int Nk = lengths[2];
    for (int i = boundary_size; i < Ni - boundary_size; ++i) {
        for (int j = boundary_size; j < Nj - boundary_size; ++j) {
            for (int k = boundary_size; k < Nk - boundary_size; ++k) {
                lap(i, j, k) = -4.0 * in(i, j, k)                  //
                               + in(i + 1, j, k) + in(i - 1, j, k) //
                               + in(i, j + 1, k) + in(i, j - 1, k);
            }
        }
    }
};
// lap-end

const auto storage_builder = storage::builder<storage::cpu_ifirst>.type<double>();

// smoothing-begin
auto naive_smoothing = [](auto &out, auto &in, double alpha, int kmax) {
    int lap_boundary = 1;
    int full_boundary = 2;

    int Ni = in.lengths()[0];
    int Nj = in.lengths()[1];
    int Nk = in.lengths()[2];

    // Instantiate temporary fields
    auto make_storage = storage_builder.dimensions(Ni, Nj, Nk);
    auto lap_storage = make_storage();
    auto lap = lap_storage->target_view();
    auto laplap_storage = make_storage();
    auto laplap = laplap_storage->target_view();

    // laplacian of phi
    laplacian(lap, in, lap_boundary);
    // laplacian of lap
    laplacian(laplap, lap, full_boundary);

    for (int i = full_boundary; i < Ni - full_boundary; ++i) {
        for (int j = full_boundary; j < Nj - full_boundary; ++j) {
            for (int k = full_boundary; k < Nk - full_boundary; ++k) {
                if (k < kmax)
                    out(i, j, k) = in(i, j, k) - alpha * laplap(i, j, k);
                else
                    out(i, j, k) = in(i, j, k);
            }
        }
    }
};
// smoothing-end

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;
    uint_t kmax = 12;

    auto make_storage = storage_builder.dimensions(Ni, Nj, Nk);

    auto phi = make_storage();
    auto phi_new = make_storage();

    auto phi_view = phi->target_view();
    auto phi_new_view = phi_new->target_view();

    laplacian(phi_new_view, phi_view, 1);
    naive_smoothing(phi_new_view, phi_view, 0.5, kmax);
}
