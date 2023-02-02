/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/unstructured.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend/naive.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;

        template <class C, int MaxNeighbors>
        struct stencil {
            GT_FUNCTION constexpr auto operator()() const {
                return [](auto const &in) {
                    int tmp = 0;
                    tuple_util::host_device::for_each(
                        [&](auto i) {
                            auto shifted = shift(in, C(), i);
                            if (can_deref(shifted))
                                tmp += deref(shifted);
                        },
                        meta::rename<tuple, meta::make_indices_c<MaxNeighbors>>());
                    return tmp;
                };
            }
        };

        struct v2v {};
        struct v2e {};

        TEST(unstructured, v2v_sum) {
            auto apply_stencil = [](auto &&executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil<v2v, 3>(), 1_c).execute();
            };
            auto fencil = [&](auto const &v2v_table, int nvertices, int nlevels, auto &out, auto const &in) {
                auto v2v_conn = connectivity<v2v>(v2v_table);
                auto domain = unstructured_domain({nvertices, nlevels}, {}, v2v_conn);
                auto backend = make_backend(backend::naive(), domain);
                apply_stencil(backend.stencil_executor(), out, in);
            };

            std::array<int, 3> v2v_table[3] = {{1, 2, -1}, {0, 2, -1}, {0, 1, -1}};

            int in[3][5], out[3][5] = {};
            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k)
                    in[v][k] = 5 * v + k;

            fencil(&v2v_table[0], 3, 5, out, in);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = 0;
                    for (int i = 0; i < 3; ++i) {
                        int nb = v2v_table[v][i];
                        if (nb != -1)
                            nbsum += in[nb][k];
                    }
                    EXPECT_EQ(out[v][k], nbsum);
                }
        }

        TEST(unstructured, v2e_sum) {
            auto apply_stencil = [](auto &&executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil<v2e, 2>(), 1_c).execute();
            };
            auto fencil = [&](auto const &v2e_table, int nvertices, int nlevels, auto &out, auto const &in) {
                auto v2e_conn = connectivity<v2e>(v2e_table);
                auto domain = unstructured_domain({nvertices, nlevels}, {}, v2e_conn);
                auto backend = make_backend(backend::naive(), domain);
                apply_stencil(backend.stencil_executor(), out, in);
            };

            std::array<int, 2> v2e_table[3] = {{0, 2}, {0, 1}, {1, 2}};

            int in[3][5], out[3][5] = {};
            for (int e = 0; e < 3; ++e)
                for (int k = 0; k < 5; ++k)
                    in[e][k] = 5 * e + k;

            fencil(&v2e_table[0], 3, 5, out, in);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = 0;
                    for (int i = 0; i < 2; ++i) {
                        int nb = v2e_table[v][i];
                        nbsum += in[nb][k];
                    }
                    EXPECT_EQ(out[v][k], nbsum);
                }
        }

    } // namespace
} // namespace gridtools::fn
