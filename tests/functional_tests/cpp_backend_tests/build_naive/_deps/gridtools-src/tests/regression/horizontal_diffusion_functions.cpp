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

#include "horizontal_diffusion_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct lap_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 1, -1, 1>>;

        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(out()) = 4. * eval(in()) - (eval(in(-1, 0)) + eval(in(0, -1)) + eval(in(0, 1)) + eval(in(1, 0)));
        }
    };

    template <class Eval, class Acc>
    using float_type = std::decay_t<decltype(std::declval<Eval>()(Acc()))>;

    enum class variation { monolithic, call, call_offsets, procedures, procedures_offsets };

    template <variation, class Acc>
    struct lap;

    template <typename Acc>
    struct lap<variation::monolithic, Acc> {
        template <typename Evaluation>
        GT_FUNCTION static auto do00(Evaluation &eval) {
            return float_type<Evaluation, Acc>(4) * eval(Acc()) -
                   (eval(Acc(-1, 0)) + eval(Acc(0, -1)) + eval(Acc(0, 1)) + eval(Acc(1, 0)));
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do10(Evaluation &eval) {
            return float_type<Evaluation, Acc>(4) * eval(Acc(1, 0)) -
                   (eval(Acc(0, 0)) + eval(Acc(1, -1)) + eval(Acc(1, 1)) + eval(Acc(2, 0)));
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do01(Evaluation &eval) {
            return float_type<Evaluation, Acc>(4) * eval(Acc(0, 1)) -
                   (eval(Acc(-1, 1)) + eval(Acc(0, 0)) + eval(Acc(0, 2)) + eval(Acc(1, 1)));
        }
    };

    template <typename Acc>
    struct lap<variation::call, Acc> {
        template <typename Evaluation>
        GT_FUNCTION static auto do00(Evaluation &eval) {
            return call<lap_function>::at<0, 0, 0>::with(eval, Acc());
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do10(Evaluation &eval) {
            return call<lap_function>::at<1, 0, 0>::with(eval, Acc());
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do01(Evaluation &eval) {
            return call<lap_function>::at<0, 1, 0>::with(eval, Acc());
        }
    };

    template <typename Acc>
    struct lap<variation::call_offsets, Acc> {
        template <typename Evaluation>
        GT_FUNCTION static auto do00(Evaluation &eval) {
            return call<lap_function>::with(eval, Acc(0, 0));
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do10(Evaluation &eval) {
            return call<lap_function>::with(eval, Acc(1, 0));
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do01(Evaluation &eval) {
            return call<lap_function>::with(eval, Acc(0, 1));
        }
    };

    template <typename Acc>
    struct lap<variation::procedures, Acc> {
        template <typename Evaluation>
        GT_FUNCTION static auto do00(Evaluation &eval) {
            float_type<Evaluation, Acc> res;
            call_proc<lap_function>::at<0, 0, 0>::with(eval, res, Acc());
            return res;
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do10(Evaluation &eval) {
            float_type<Evaluation, Acc> res;
            call_proc<lap_function>::at<1, 0, 0>::with(eval, res, Acc());
            return res;
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do01(Evaluation &eval) {
            float_type<Evaluation, Acc> res;
            call_proc<lap_function>::at<0, 1, 0>::with(eval, res, Acc());
            return res;
        }
    };

    template <typename Acc>
    struct lap<variation::procedures_offsets, Acc> {
        template <typename Evaluation>
        GT_FUNCTION static auto do00(Evaluation &eval) {
            float_type<Evaluation, Acc> res;
            call_proc<lap_function>::with(eval, res, Acc(0, 0));
            return res;
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do10(Evaluation &eval) {
            float_type<Evaluation, Acc> res;
            call_proc<lap_function>::with(eval, res, Acc(1, 0));
            return res;
        }
        template <typename Evaluation>
        GT_FUNCTION static auto do01(Evaluation &eval) {
            float_type<Evaluation, Acc> res;
            call_proc<lap_function>::with(eval, res, Acc(0, 1));
            return res;
        }
    };

    template <variation Variation>
    struct flx_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 2, -1, 1>>;

        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(out()) = lap<Variation, in>::do10(eval) - lap<Variation, in>::do00(eval);
            eval(out()) = eval(out()) * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0.0 : eval(out());
        }
    };

    template <variation Variation>
    struct fly_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1, extent<-1, 1, -1, 2>>;

        using param_list = make_param_list<out, in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(out()) = lap<Variation, in>::do01(eval) - lap<Variation, in>::do00(eval);
            eval(out()) = eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0 ? 0.0 : eval(out());
        }
    };

    struct out_function {
        using out = inout_accessor<0>;
        using in = in_accessor<1>;
        using flx = in_accessor<2, extent<-1, 0, 0, 0>>;
        using fly = in_accessor<3, extent<0, 0, -1, 0>>;
        using coeff = in_accessor<4>;

        using param_list = make_param_list<out, in, flx, fly, coeff>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation eval) {
            eval(out()) =
                eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0)) + eval(fly()) - eval(fly(0, -1)));
        }
    };

    template <class Env, variation Variation>
    void do_test() {
        auto out = Env::make_storage();
        horizontal_diffusion_repository repo(Env::d(0), Env::d(1), Env::d(2));
        run(
            [](auto coeff, auto in, auto out) {
                GT_DECLARE_TMP(typename Env::float_t, flx, fly);
                return execute_parallel()
                    .ij_cached(flx, fly)
                    .stage(flx_function<Variation>(), flx, in)
                    .stage(fly_function<Variation>(), fly, in)
                    .stage(out_function(), out, in, flx, fly, coeff);
            },
            stencil_backend_t(),
            Env::make_grid(),
            Env::make_storage(repo.coeff),
            Env::make_storage(repo.in),
            out);
        Env::verify(repo.out, out);
    }

#define TEST_VARIATION(v)                                                                            \
    GT_REGRESSION_TEST(horizontal_diffusion_functions_##v, test_environment<2>, stencil_backend_t) { \
        do_test<TypeParam, variation::v>();                                                          \
    }                                                                                                \
    static_assert(1)

    TEST_VARIATION(monolithic);
    TEST_VARIATION(call);
    TEST_VARIATION(call_offsets);
    TEST_VARIATION(procedures);
    TEST_VARIATION(procedures_offsets);
} // namespace
