/*
 * GT4Py - GridTools4Py - GridTools for Python
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * This file is part the GT4Py project and the GridTools framework.
 * GT4Py is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or any later
 * version. See the LICENSE.txt file at the top-level directory of this
 * distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gridtools/regression/horizontal_diffusion_repository.hpp"
#include "gridtools/regression/vertical_advection_repository.hpp"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = ::pybind11;
namespace gt = ::gridtools;

std::array<std::array<gt::uint_t, 2>, 3> make_extent(
    std::array<gt::uint_t, 2> ei, std::array<gt::uint_t, 2> ej, std::array<gt::uint_t, 2> ek) {
    return {ei, ej, ek};
}

template <typename DType, typename Function>
py::array_t<DType> apply_function(gt::uint_t d1,
    gt::uint_t d2,
    gt::uint_t d3,
    Function func,
    std::array<gt::uint_t, 3> origin,
    std::array<std::array<gt::uint_t, 2>, 3> extent = make_extent({0, 0}, {0, 0}, {0, 0})) {
    gt::uint_t d1_loc = d1 + extent[0][0] + extent[0][1];
    gt::uint_t d2_loc = d2 + extent[1][0] + extent[1][1];
    gt::uint_t d3_loc = d3 + extent[2][0] + extent[2][1];

    auto result = py::array_t<DType>({d1_loc, d2_loc, d3_loc});
    size_t i_stride = result.strides()[0]/8;
    size_t j_stride = result.strides()[1]/8;
    size_t k_stride = result.strides()[2]/8;
    DType* ptr = static_cast<DType*>(result.request().ptr);
    
    for (gt::uint_t i = 0; i < d1_loc; i++) {
        for (gt::uint_t j = 0; j < d2_loc; j++) {
            for (gt::uint_t k = 0; k < d3_loc; k++) {
                auto value = func(i + origin[0], j + origin[1], k + origin[2]);
                ptr[i * i_stride + j * j_stride + k * k_stride] = value;
            }
        }
    }
    return result;
}

namespace tridiagonal_solver {

    template <typename DType>
    py::dict get(gt::uint_t d1, gt::uint_t d2, gt::uint_t d3) {
        std::array<gt::uint_t, 3> zero_origin{0, 0, 0};

        py::dict fields;
        fields["inf"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t k) { return -1.; }, zero_origin);
        fields["diag"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t k) { return 3.; }, zero_origin);
        fields["sup"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t k) { return 1.; }, zero_origin);
        fields["rhs"] = apply_function<DType>(d1,
            d2,
            d3,
            [=](gt::int_t, gt::int_t, gt::int_t k) { return k == 0 ? 4. : k == (gt::int_t)d3 - 1 ? 2. : 3.; },
            zero_origin);
        fields["out"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t) { return 0; }, zero_origin);
        fields["out_reference"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t) { return 1; }, zero_origin);
        return fields;
    }

} // namespace tridiagonal_solver

namespace vertical_advection_dycore {
    template <typename DType>
    py::dict get(gt::uint_t d1, gt::uint_t d2, gt::uint_t d3) {

        auto max_extent = make_extent({0, 1}, {0, 0}, {0, 0});

        auto extent_wcon = make_extent({0, 1}, {0, 0}, {0, 0});
        auto extent_u_stage = make_extent({0, 0}, {0, 0}, {0, 0});

        std::array<gt::uint_t, 3> origin_utens_stage{0, 0, 0};
        std::array<gt::uint_t, 3> origin_u_stage{0, 0, 0};
        std::array<gt::uint_t, 3> origin_wcon{0, 0, 0};
        std::array<gt::uint_t, 3> origin_u_pos{0, 0, 0};
        std::array<gt::uint_t, 3> origin_utens{0, 0, 0};

        auto d1_all = d1 + max_extent[0][0] + max_extent[0][1];
        auto d2_all = d2 + max_extent[1][0] + max_extent[1][1];
        auto d3_all = d3 + max_extent[2][0] + max_extent[2][1];

        gt::vertical_advection_repository repo{d1_all, d2_all, d3_all};

        py::dict fields;
        // fields
        fields["utens_stage"] = apply_function<DType>(d1, d2, d3, repo.utens_stage_in, origin_utens_stage);
        fields["utens_stage_reference"] = apply_function<DType>(d1, d2, d3, repo.utens_stage_out, origin_utens_stage);
        fields["u_stage"] = apply_function<DType>(d1, d2, d3, repo.u_stage, origin_u_stage, extent_u_stage);
        fields["wcon"] = apply_function<DType>(d1, d2, d3, repo.wcon, origin_wcon, extent_wcon);
        fields["u_pos"] = apply_function<DType>(d1, d2, d3, repo.u_pos, origin_u_pos);
        fields["utens"] = apply_function<DType>(d1, d2, d3, repo.utens, origin_utens);

        // scalar param
        fields["dtr_stage"] = (DType)repo.dtr_stage;
        return fields;
    }
} // namespace vertical_advection_dycore

namespace vertical_advection_dycore_with_scalar_storage {
    template <typename DType>
    py::dict get(gt::uint_t d1, gt::uint_t d2, gt::uint_t d3) {

        auto max_extent = make_extent({0, 1}, {0, 0}, {0, 0});

        auto extent_wcon = make_extent({0, 1}, {0, 0}, {0, 0});
        auto extent_u_stage = make_extent({0, 0}, {0, 0}, {0, 0});

        std::array<gt::uint_t, 3> origin_utens_stage{0, 0, 0};
        std::array<gt::uint_t, 3> origin_u_stage{0, 0, 0};
        std::array<gt::uint_t, 3> origin_wcon{0, 0, 0};
        std::array<gt::uint_t, 3> origin_u_pos{0, 0, 0};
        std::array<gt::uint_t, 3> origin_utens{0, 0, 0};
        std::array<gt::uint_t, 3> origin_dtr_stage{0, 0, 0};

        auto d1_all = d1 + max_extent[0][0] + max_extent[0][1];
        auto d2_all = d2 + max_extent[1][0] + max_extent[1][1];
        auto d3_all = d3 + max_extent[2][0] + max_extent[2][1];

        gt::vertical_advection_repository repo{d1_all, d2_all, d3_all};

        py::dict fields;
        // fields
        fields["utens_stage"] = apply_function<DType>(d1, d2, d3, repo.utens_stage_in, origin_utens_stage);
        fields["utens_stage_reference"] = apply_function<DType>(d1, d2, d3, repo.utens_stage_out, origin_utens_stage);
        fields["u_stage"] = apply_function<DType>(d1, d2, d3, repo.u_stage, origin_u_stage, extent_u_stage);
        fields["wcon"] = apply_function<DType>(d1, d2, d3, repo.wcon, origin_wcon, extent_wcon);
        fields["u_pos"] = apply_function<DType>(d1, d2, d3, repo.u_pos, origin_u_pos);
        fields["utens"] = apply_function<DType>(d1, d2, d3, repo.utens, origin_utens);

        // scalar param
        fields["dtr_stage"] = apply_function<DType>(
            1, 1, 1, [=](gt::int_t, gt::int_t, gt::int_t) { return repo.dtr_stage; }, origin_dtr_stage);

        return fields;
    }
} // namespace vertical_advection_dycore_with_scalar_storage

namespace horizontal_diffusion {

    template <typename DType>
    py::dict get(gt::uint_t d1, gt::uint_t d2, gt::uint_t d3) {

        auto max_extent = make_extent({2, 2}, {2, 2}, {0, 0});

        auto extent_in_field = make_extent({2, 2}, {2, 2}, {0, 0});

        std::array<gt::uint_t, 3> origin_in_field{0, 0, 0};
        std::array<gt::uint_t, 3> origin_coeff{2, 2, 0};
        std::array<gt::uint_t, 3> origin_out_field{2, 2, 0};

        auto d1_all = d1 + max_extent[0][0] + max_extent[0][1];
        auto d2_all = d2 + max_extent[1][0] + max_extent[1][1];
        auto d3_all = d3 + max_extent[2][0] + max_extent[2][1];

        gt::horizontal_diffusion_repository repo{d1_all, d2_all, d3_all};

        py::dict fields;
        fields["in_field"] = apply_function<DType>(d1, d2, d3, repo.in, origin_in_field, extent_in_field);
        fields["coeff"] = apply_function<DType>(d1, d2, d3, repo.coeff, origin_coeff);
        fields["out_field"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t) { return 0.0; }, origin_out_field);
        fields["out_field_reference"] = apply_function<DType>(d1, d2, d3, repo.out, origin_out_field);
        return fields;
    }

} // namespace horizontal_diffusion

namespace large_k_interval {

    template <typename DType>
    py::dict get(gt::uint_t d1, gt::uint_t d2, gt::uint_t d3) {
        std::array<gt::uint_t, 3> zero_origin{0, 0, 0};

        py::dict fields;
        fields["in_field"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t) { return 1.; }, zero_origin);
        fields["out_field"] =
            apply_function<DType>(d1, d2, d3, [](gt::int_t, gt::int_t, gt::int_t) { return 0; }, zero_origin);
        fields["out_field_reference"] =
            apply_function<DType>(d1, d2, d3, [d3](gt::int_t, gt::int_t, gt::int_t k) { return k >= 6 && k < d3-10 ? 2 : 1; }, zero_origin);
        return fields;
    }

} // namespace tridiagonal_solver

PYBIND11_MODULE(reference_cpp_regression, m) {
    m.def("tridiagonal_solver", &tridiagonal_solver::get<double>);
    m.def("vertical_advection_dycore", &vertical_advection_dycore::get<double>);
    m.def("vertical_advection_dycore_with_scalar_storage", &vertical_advection_dycore_with_scalar_storage::get<double>);
    m.def("horizontal_diffusion", &horizontal_diffusion::get<double>);
    m.def("large_k_interval", &large_k_interval::get<double>);
}
