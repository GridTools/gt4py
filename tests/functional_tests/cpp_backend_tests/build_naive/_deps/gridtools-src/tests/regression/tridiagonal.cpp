/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

/*
  @file This file shows an implementation of the Thomas algorithm, done using stencil operations.

  Important convention: the linear system as usual is represented with 4 vectors: the main diagonal
  (diag), the upper and lower first diagonals (sup and inf respectively), and the right hand side
  (rhs). Note that the dimensions and the memory layout are, for an NxN system
  rank(diag)=N       [xxxxxxxxxxxxxxxxxxxxxxxx]
  rank(inf)=N-1      [0xxxxxxxxxxxxxxxxxxxxxxx]
  rank(sup)=N-1      [xxxxxxxxxxxxxxxxxxxxxxx0]
  rank(rhs)=N        [xxxxxxxxxxxxxxxxxxxxxxxx]
  where x denotes any number and 0 denotes the padding, a dummy value which is not used in
  the algorithm. This choice coresponds to having the same vector index for each row of the matrix.
 */

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;
    using namespace expressions;

    // This is the definition of the special regions in the "vertical" direction
    using axis_t = axis<1>;
    using full_t = axis_t::full_interval;

    struct forward_thomas {
        using inf = in_accessor<0>;                               // a
        using diag = in_accessor<1>;                              // b
        using sup = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>; // c
        using rhs = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>; // d
        using param_list = make_param_list<inf, diag, sup, rhs>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, 0>) {
            eval(sup{}) = eval(sup{} / (diag{} - sup{0, 0, -1} * inf{}));
            eval(rhs{}) = eval((rhs{} - inf{} * rhs{0, 0, -1}) / (diag{} - sup{0, 0, -1} * inf{}));
        }

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
            eval(sup{}) = eval(sup{}) / eval(diag{});
            eval(rhs{}) = eval(rhs{}) / eval(diag{});
        }
    };

    struct backward_thomas {
        using out = inout_accessor<0, extent<0, 0, 0, 0, 0, 1>>;
        using sup = in_accessor<1>; // c
        using rhs = in_accessor<2>; // d
        using param_list = make_param_list<out, sup, rhs>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
            eval(out{}) = eval(rhs{} - sup{} * out{0, 0, 1});
        }

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
            eval(out{}) = eval(rhs{});
        }
    };

    GT_ENVIRONMENT_TEST_SUITE(tridiagonal,
        vertical_test_environment<>,
        stencil_backend_t,
        (double, inlined_params<12, 33, 6>),
        (double, inlined_params<23, 11, 6>));
    GT_ENVIRONMENT_TYPED_TEST(tridiagonal, test) {
        auto out = TypeParam::make_storage();
        run(
            [](auto inf, auto diag, auto sup, auto rhs, auto out) {
                return multi_pass(execute_forward().stage(forward_thomas(), inf, diag, sup, rhs),
                    execute_backward().stage(backward_thomas(), out, sup, rhs));
            },
            stencil_backend_t(),
            TypeParam::make_grid(),
            TypeParam::make_storage(-1),
            TypeParam::make_storage(3),
            TypeParam::make_storage(1),
            TypeParam::make_storage([](int, int, int k) { return k == 0   ? 4
                                                                 : k == 5 ? 2
                                                                          : 3; }),
            out);
        TypeParam::verify(1, out);
    }
} // namespace
