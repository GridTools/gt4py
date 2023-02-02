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

#include <gridtools/fn/unstructured.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct zavg_stencil {
        constexpr auto operator()() const {
            return [](auto const &pp, auto const &s) {
                std::decay_t<decltype(deref(pp))> tmp = 0;
                tuple_util::host_device::for_each(
                    [&](auto i) {
                        auto shifted_pp = shift(pp, e2v(), i);
                        if (can_deref(shifted_pp))
                            tmp += deref(shifted_pp);
                    },
                    meta::rename<tuple, meta::make_indices_c<2>>());
                tmp /= 2;
                auto ss = deref(s);
                return make_tuple(tmp * tuple_get(0_c, ss), tmp * tuple_get(1_c, ss));
            };
        }
    };

    struct nabla_stencil {
        constexpr auto operator()() const {
            return [](auto const &zavg, auto const &sign, auto const &vol) {
                using float_t = std::decay_t<decltype(deref(vol))>;
                auto signs = deref(sign);
                tuple<float_t, float_t> tmp(0, 0);
                tuple_util::host_device::for_each(
                    [&](auto i) {
                        auto shifted_zavg = shift(zavg, v2e(), i);
                        if (can_deref(shifted_zavg)) {
                            tuple_get(0_c, tmp) += tuple_get(0_c, deref(shifted_zavg)) * get<i.value>(signs);
                            tuple_get(1_c, tmp) += tuple_get(1_c, deref(shifted_zavg)) * get<i.value>(signs);
                        }
                    },
                    meta::rename<tuple, meta::make_indices_c<6>>());
                auto v = deref(vol);
                return make_tuple(tuple_get(0_c, tmp) / v, tuple_get(1_c, tmp) / v);
            };
        }
    };

    struct nabla_stencil_fused {
        constexpr auto operator()() const {
            return [](auto const &sign, auto const &vol, auto const &pp, auto const &s) {
                using float_t = std::decay_t<decltype(deref(vol))>;
                auto signs = deref(sign);
                tuple<float_t, float_t> tmp(0, 0);
                tuple_util::host_device::for_each(
                    [&](auto i) {
                        auto shifted_s = shift(s, v2e(), i);
                        if (can_deref(shifted_s)) {
                            float_t tmp2 = 0;
                            tuple_util::host_device::for_each(
                                [&](auto const &j) {
                                    auto shifted_pp = shift(pp, v2e(), i, e2v(), j);
                                    if (can_deref(shifted_pp))
                                        tmp2 += deref(shifted_pp);
                                },
                                meta::rename<tuple, meta::make_indices_c<2>>());
                            tmp2 /= 2;
                            auto ss = deref(shifted_s);
                            auto zavg = tuple(tmp2 * tuple_get(0_c, ss), tmp2 * tuple_get(1_c, ss));

                            tuple_get(0_c, tmp) += tuple_get(0_c, zavg) * get<i.value>(signs);
                            tuple_get(1_c, tmp) += tuple_get(1_c, zavg) * get<i.value>(signs);
                        }
                    },
                    meta::rename<tuple, meta::make_indices_c<6>>());
                auto v = deref(vol);
                return make_tuple(tuple_get(0_c, tmp) / v, tuple_get(1_c, tmp) / v);
            };
        }
    };

    constexpr inline auto pp = [](int vertex, int k) { return (vertex + k) % 19; };
    constexpr inline auto sign = [](int vertex) { return array<int, 6>{0, 1, vertex % 2, 1, (vertex + 1) % 2, 0}; };
    constexpr inline auto vol = [](int vertex) { return vertex % 13 + 1; };
    constexpr inline auto s = [](int edge, int k) { return tuple((edge + k) % 17, (edge + k) % 7); };
    constexpr inline auto zavg = [](auto const &e2v) {
        return [&e2v](int edge, int k) {
            double tmp = 0.0;
            for (int neighbor = 0; neighbor < 2; ++neighbor)
                tmp += pp(e2v(edge)[neighbor], k);
            tmp /= 2.0;
            return tuple{tmp * get<0>(s(edge, k)), tmp * get<1>(s(edge, k))};
        };
    };

    constexpr inline auto expected = [](auto const &v2e, auto const &e2v) {
        return [&v2e, zavg = zavg(e2v)](int vertex, int k) {
            auto res = tuple(0.0, 0.0);
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                int edge = v2e(vertex)[neighbor];
                if (edge != -1) {
                    get<0>(res) += get<0>(zavg(edge, k)) * sign(vertex)[neighbor];
                    get<1>(res) += get<1>(zavg(edge, k)) * sign(vertex)[neighbor];
                }
            }
            get<0>(res) /= vol(vertex);
            get<1>(res) /= vol(vertex);
            return res;
        };
    };

    constexpr inline auto apply_zavg = [](auto &&executor, auto &zavg, auto const &pp, auto const &s) {
        executor().arg(zavg).arg(pp).arg(s).assign(0_c, zavg_stencil(), 1_c, 2_c).execute();
    };
    constexpr inline auto apply_nabla =
        [](auto executor, auto &nabla, auto const &zavg, auto const &sign, auto const &vol) {
            executor().arg(nabla).arg(zavg).arg(sign).arg(vol).assign(0_c, nabla_stencil(), 1_c, 2_c, 3_c).execute();
        };
    constexpr inline auto apply_nabla_fused =
        [](auto executor, auto &nabla, auto const &sign, auto const &vol, auto const &pp, auto const &s) {
            executor()
                .arg(nabla)
                .arg(sign)
                .arg(vol)
                .arg(pp)
                .arg(s)
                .assign(0_c, nabla_stencil_fused(), 1_c, 2_c, 3_c, 4_c)
                .execute();
        };

    constexpr inline auto fencil = [](auto backend,
                                       int nvertices,
                                       int nedges,
                                       int nlevels,
                                       auto const &v2e_table,
                                       auto const &e2v_table,
                                       auto &nabla,
                                       auto const &pp,
                                       auto const &s,
                                       auto const &sign,
                                       auto const &vol) {
        using float_t = std::remove_const_t<sid::element_type<decltype(pp)>>;
        auto v2e_conn = connectivity<v2e>(v2e_table);
        auto e2v_conn = connectivity<e2v>(e2v_table);
        auto edge_domain = unstructured_domain({nedges, nlevels}, {}, e2v_conn);
        auto vertex_domain = unstructured_domain({nvertices, nlevels}, {}, v2e_conn);
        auto edge_backend = make_backend(backend, edge_domain);
        auto vertex_backend = make_backend(backend, vertex_domain);
        auto alloc = tmp_allocator(backend);
        auto zavg = allocate_global_tmp<tuple<float_t, float_t>>(alloc, edge_domain.sizes());
        apply_zavg(edge_backend.stencil_executor(), zavg, pp, s);
        apply_nabla(vertex_backend.stencil_executor(), nabla, zavg, sign, vol);
    };

    constexpr inline auto fencil_fused = [](auto backend,
                                             int nvertices,
                                             int nlevels,
                                             auto const &v2e_table,
                                             auto const &e2v_table,
                                             auto &nabla,
                                             auto const &pp,
                                             auto const &s,
                                             auto const &sign,
                                             auto const &vol) {
        auto v2e_conn = connectivity<v2e>(v2e_table);
        auto e2v_conn = connectivity<e2v>(e2v_table);
        auto vertex_domain = unstructured_domain({nvertices, nlevels}, {}, v2e_conn, e2v_conn);
        auto vertex_backend = make_backend(backend, vertex_domain);
        apply_nabla_fused(vertex_backend.stencil_executor(), nabla, sign, vol, pp, s);
    };

    constexpr inline auto make_comp = [](auto backend, auto const &mesh, auto &nabla) {
        return [backend,
                   &nabla,
                   nvertices = mesh.nvertices(),
                   nedges = mesh.nedges(),
                   nlevels = mesh.nlevels(),
                   v2e_table = mesh.v2e_table(),
                   e2v_table = mesh.e2v_table(),
                   pp = mesh.make_const_storage(pp, mesh.nvertices(), mesh.nlevels()),
                   sign = mesh.template make_const_storage<array<float_t, 6>>(sign, mesh.nvertices()),
                   vol = mesh.make_const_storage(vol, mesh.nvertices()),
                   s = mesh.template make_const_storage<tuple<float_t, float_t>>(s, mesh.nedges(), mesh.nlevels())] {
            auto v2e_ptr = v2e_table->get_const_target_ptr();
            auto e2v_ptr = e2v_table->get_const_target_ptr();
            fencil(backend, nvertices, nedges, nlevels, v2e_ptr, e2v_ptr, nabla, pp, s, sign, vol);
        };
    };

    constexpr inline auto make_comp_fused = [](auto backend, auto const &mesh, auto &nabla) {
        return [backend,
                   &nabla,
                   nvertices = mesh.nvertices(),
                   nlevels = mesh.nlevels(),
                   v2e_table = mesh.v2e_table(),
                   e2v_table = mesh.e2v_table(),
                   pp = mesh.make_const_storage(pp, mesh.nvertices(), mesh.nlevels()),
                   sign = mesh.template make_const_storage<array<float_t, 6>>(sign, mesh.nvertices()),
                   vol = mesh.make_const_storage(vol, mesh.nvertices()),
                   s = mesh.template make_const_storage<tuple<float_t, float_t>>(s, mesh.nedges(), mesh.nlevels())] {
            auto v2e_ptr = v2e_table->get_const_target_ptr();
            auto e2v_ptr = e2v_table->get_const_target_ptr();
            fencil_fused(backend, nvertices, nlevels, v2e_ptr, e2v_ptr, nabla, pp, s, sign, vol);
        };
    };

    constexpr inline auto make_expected = [](auto const &mesh) {
        return [v2e_table = mesh.v2e_table(), e2v_table = mesh.e2v_table()](int vertex, int k) {
            auto v2e = v2e_table->const_host_view();
            auto e2v = e2v_table->const_host_view();
            return expected(v2e, e2v)(vertex, k);
        };
    };

    GT_REGRESSION_TEST(fn_unstructured_nabla_field_of_tuples, test_environment<>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;

        auto mesh = TypeParam::fn_unstructured_mesh();
        auto nabla = mesh.template make_storage<tuple<float_t, float_t>>(mesh.nvertices(), mesh.nlevels());
        auto comp = make_comp(fn_backend_t(), mesh, nabla);
        comp();
        auto expected = make_expected(mesh);
        TypeParam::verify(expected, nabla);
        TypeParam::benchmark("fn_unstructured_nabla_field_of_tuples", comp);
    }

    GT_REGRESSION_TEST(fn_unstructured_nabla_fused_field_of_tuples, test_environment<>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;

        auto mesh = TypeParam::fn_unstructured_mesh();
        auto nabla = mesh.template make_storage<tuple<float_t, float_t>>(mesh.nvertices(), mesh.nlevels());
        auto comp = make_comp_fused(fn_backend_t(), mesh, nabla);
        comp();
        auto expected = make_expected(mesh);
        TypeParam::verify(expected, nabla);
        TypeParam::benchmark("fn_unstructured_nabla_fused_field_of_tuples", comp);
    }

    GT_REGRESSION_TEST(fn_unstructured_nabla_tuple_of_fields, test_environment<>, fn_backend_t) {
        auto mesh = TypeParam::fn_unstructured_mesh();
        auto nabla0 = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        auto nabla1 = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        auto nabla =
            sid::composite::keys<integral_constant<int, 0>, integral_constant<int, 1>>::make_values(nabla0, nabla1);

        auto comp = make_comp(fn_backend_t(), mesh, nabla);
        comp();
        auto expected = make_expected(mesh);
        TypeParam::verify([&](int vertex, int k) { return get<0>(expected(vertex, k)); }, nabla0);
        TypeParam::verify([&](int vertex, int k) { return get<1>(expected(vertex, k)); }, nabla1);
        TypeParam::benchmark("fn_unstructured_nabla_tuple_of_fields", comp);
    }
} // namespace
