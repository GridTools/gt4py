/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/fn/cartesian.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

#include "../horizontal_diffusion_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace cartesian;
    using namespace literals;

    struct laplacian {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in) {
                constexpr auto i = cartesian::dim::i();
                constexpr auto j = cartesian::dim::j();
                return 4 * deref(in) - (deref(shift(in, i, 1)) + deref(shift(in, i, -1)) + deref(shift(in, j, 1)) +
                                           deref(shift(in, j, -1)));
            };
        }
    };

    template <class D>
    struct flux {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in, auto const &lap) {
                auto tmp = deref(shift(lap, D(), 1)) - deref(lap);
                return tmp * (deref(shift(in, D(), 1)) - deref(in)) > 0 ? 0 : tmp;
            };
        }
    };

    struct hdiff {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in, auto const &coeff, auto const &flx, auto const &fly) {
                constexpr auto i = cartesian::dim::i();
                constexpr auto j = cartesian::dim::j();
                return deref(in) -
                       deref(coeff) * (deref(flx) - deref(shift(flx, i, -1)) + deref(fly) - deref(shift(fly, j, -1)));
            };
        }
    };

    struct hdiff_fused {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in, auto const &coeff) {
                constexpr auto i = cartesian::dim::i();
                constexpr auto j = cartesian::dim::j();

                auto lap = 4 * deref(in) - (deref(shift(in, i, 1)) + deref(shift(in, i, -1)) + deref(shift(in, j, 1)) +
                                               deref(shift(in, j, -1)));
                auto lap_ip1 =
                    4 * deref(shift(in, i, 1)) -
                    (deref(shift(in, i, 2)) + deref(in) + deref(shift(in, i, 1, j, 1)) + deref(shift(in, i, 1, j, -1)));
                auto lap_im1 =
                    4 * deref(shift(in, i, -1)) - (deref(in) + deref(shift(in, i, -2)) + deref(shift(in, i, -1, j, 1)) +
                                                      deref(shift(in, i, -1, j, -1)));
                auto lap_jp1 =
                    4 * deref(shift(in, j, 1)) -
                    (deref(shift(in, i, 1, j, 1)) + deref(shift(in, i, -1, j, 1)) + deref(shift(in, j, 2)) + deref(in));
                auto lap_jm1 =
                    4 * deref(shift(in, j, -1)) - (deref(shift(in, i, 1, j, -1)) + deref(shift(in, i, -1, j, -1)) +
                                                      deref(in) + deref(shift(in, j, -2)));

                auto tmp0 = lap_ip1 - lap;
                auto flx = tmp0 * (deref(shift(in, i, 1)) - deref(in)) > 0 ? 0 : tmp0;
                auto tmp1 = lap - lap_im1;
                auto flx_im1 = tmp1 * (deref(in) - deref(shift(in, i, -1))) > 0 ? 0 : tmp1;

                auto tmp2 = lap_jp1 - lap;
                auto fly = tmp2 * (deref(shift(in, j, 1)) - deref(in)) > 0 ? 0 : tmp2;
                auto tmp3 = lap - lap_jm1;
                auto fly_jm1 = tmp3 * (deref(in) - deref(shift(in, j, -1))) > 0 ? 0 : tmp3;
                return deref(in) - deref(coeff) * (flx - flx_im1 + fly - fly_jm1);
            };
        }
    };

    GT_REGRESSION_TEST(fn_cartesian_horizontal_diffusion, test_environment<2>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;
        horizontal_diffusion_repository repo(TypeParam::d(0), TypeParam::d(1), TypeParam::d(2));
        auto out = TypeParam::make_storage();
        auto fencil = [&](int i, int j, int k, auto &out, auto const &in, auto const &coeff) {
            using sizes_t = hymap::keys<dim::i, dim::j, dim::k>::values<int, int, int>;
            auto be = fn_backend_t();
            auto full_domain = cartesian_domain(sizes_t{i, j, k});
            auto full_domain_backend = make_backend(be, full_domain);

            auto alloc = tmp_allocator(be);
            auto lap = allocate_global_tmp<float_t>(alloc, sizes_t{i, j, k});
            auto flx = allocate_global_tmp<float_t>(alloc, sizes_t{i, j, k});
            auto fly = allocate_global_tmp<float_t>(alloc, sizes_t{i, j, k});

            auto domain = cartesian_domain(sizes_t{i - 2, j - 2, k}, sizes_t{1, 1, 0});
            auto backend = make_backend(fn_backend_t(), domain);

            backend.stencil_executor()().arg(lap).arg(in).assign(0_c, laplacian(), 1_c).execute();
            backend.stencil_executor()()
                .arg(flx)
                .arg(fly)
                .arg(in)
                .arg(lap)
                .assign(0_c, flux<dim::i>(), 2_c, 3_c)
                .assign(1_c, flux<dim::j>(), 2_c, 3_c)
                .execute();
            backend.stencil_executor()()
                .arg(out)
                .arg(in)
                .arg(coeff)
                .arg(flx)
                .arg(fly)
                .assign(0_c, hdiff(), 1_c, 2_c, 3_c, 4_c)
                .execute();
        };
        auto comp =
            [&, coeff = TypeParam::make_const_storage(repo.coeff), in = TypeParam::make_const_storage(repo.in)] {
                fencil(TypeParam::d(0), TypeParam::d(1), TypeParam::d(2), out, in, coeff);
            };
        comp();
        TypeParam::verify(repo.out, out);
        TypeParam::benchmark("fn_cartesian_horizontal_diffusion", comp);
    }

    GT_REGRESSION_TEST(fn_cartesian_horizontal_diffusion_fused, test_environment<2>, fn_backend_t) {
        horizontal_diffusion_repository repo(TypeParam::d(0), TypeParam::d(1), TypeParam::d(2));
        auto out = TypeParam::make_storage();
        auto fencil = [&](int i, int j, int k, auto &out, auto const &in, auto const &coeff) {
            using sizes_t = hymap::keys<dim::i, dim::j, dim::k>::values<int, int, int>;
            auto domain = cartesian_domain(sizes_t{i - 4, j - 4, k}, sizes_t{2, 2, 0});
            auto backend = make_backend(fn_backend_t(), domain);

            backend.stencil_executor()().arg(out).arg(in).arg(coeff).assign(0_c, hdiff_fused(), 1_c, 2_c).execute();
        };
        auto comp =
            [&, coeff = TypeParam::make_const_storage(repo.coeff), in = TypeParam::make_const_storage(repo.in)] {
                fencil(TypeParam::d(0), TypeParam::d(1), TypeParam::d(2), out, in, coeff);
            };
        comp();
        TypeParam::verify(repo.out, out);
        TypeParam::benchmark("fn_cartesian_horizontal_diffusion_fused", comp);
    }
} // namespace
