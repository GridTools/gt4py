/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// Keep first to test for missing includes
#include <gridtools/storage/adapter/python_sid_adapter.hpp>

#include <cassert>
#include <cstdlib>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gridtools/common/for_each.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#include <gridtools/stencil/naive.hpp>

namespace py = pybind11;

using namespace gridtools;
using namespace stencil;
using namespace cartesian;

struct copy_functor {
    using in = in_accessor<0>;
    using out = inout_accessor<1>;
    using param_list = make_param_list<in, out>;
    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

// Here is a generic implementation of the copy algorithm.
// The input and output fields are passed as SIDs
template <class From, class To>
void copy(From &&from, To &&to) {
    static_assert(is_sid<From>());
    static_assert(is_sid<To>());
    auto &&size = sid::get_upper_bounds(to);
    run_single_stage(copy_functor(),
        naive(),
        make_grid(at_key<dim::i>(size), at_key<dim::j>(size), at_key<dim::k>(size)),
        std::forward<From>(from),
        std::forward<To>(to));
}

template <class T>
void check_hymap(T const &actual, std::vector<size_t> const &expected) {
    assert(tuple_util::size<T>() == expected.size());
    using key_t = meta::make_indices<tuple_util::size<T>>;
    for_each<get_keys<T>>([&](auto key) {
        using key_t = decltype(key);
        assert(at_key<key_t>(actual) == expected[key_t::value]);
    });
}

template <class T>
void check_cuda_sid(T &&testee,
    size_t expected_ptr,
    std::vector<size_t> const &expected_strides,
    std::vector<size_t> const &expected_dims) {
    static_assert(is_sid<T>());
    using lower_bounds_t = sid::lower_bounds_type<T>;
    using upper_bounds_t = sid::upper_bounds_type<T>;
    using strides_t = sid::strides_type<T>;
    static_assert(tuple_util::size<lower_bounds_t>() == tuple_util::size<strides_t>());
    static_assert(tuple_util::size<upper_bounds_t>() == tuple_util::size<strides_t>());

    assert(reinterpret_cast<size_t>(sid::get_origin(testee)()) == expected_ptr);
    check_hymap(sid::get_strides(testee), expected_strides);
    check_hymap(sid::get_upper_bounds(testee), expected_dims);
    check_hymap(sid::get_lower_bounds(testee), std::vector<size_t>(tuple_util::size<lower_bounds_t>()));
}

// The module exports several instantiations of the generic `copy` to python.
// The differences between exported functions are in the way how parameters model the SID concept.
// Note that the generic algorithm stays the same.
PYBIND11_MODULE(py_implementation, m) {
    m.def(
        "copy_from_3D",
        [](py::buffer from, py::buffer to) { copy(as_sid<double const, 3>(from), as_sid<double, 3>(to)); },
        "Copy from one 3D buffer of doubles to another.");
    m.def(
        "copy_from_3D_with_unit_stride",
        [](py::buffer from, py::buffer to) {
            copy(as_sid<double const, 3, void, 2>(from), as_sid<double, 3, void, 2>(to));
        },
        "Copy from one 3D buffer of doubles to another, requires `from.strides[2] == to.strides[2] == "
        "sizeof(double)`.");
    m.def(
        "copy_from_1D",
        [](py::buffer from, py::buffer to) { copy(as_sid<double const, 1>(from), as_sid<double, 3>(to)); },
        "Copy from the 1D double buffer to a 3D one.");
    m.def(
        "copy_from_scalar",
        [](double from, py::buffer to) { copy(global_parameter(from), as_sid<double, 3>(to)); },
        "Copy from the scalar to a 3D buffer of doubles.");
    m.def(
        "check_cuda_sid",
        [](py::object testeee, size_t ptr, std::vector<size_t> const &strides, std::vector<size_t> const &dims) {
            check_cuda_sid(as_cuda_sid<double const, 3>(testeee), ptr, strides, dims);
        },
        "Check CUDA Sid.");
}
