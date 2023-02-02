/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ostream>
#include <string>
#include <typeinfo>

#include <boost/core/demangle.hpp>
#include <nlohmann/json.hpp>

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "be_api.hpp"
#include "common/caches.hpp"
#include "common/extent.hpp"
#include "core/execution_types.hpp"
#include "core/functor_metafunctions.hpp"
#include "core/interval.hpp"
#include "core/is_tmp_arg.hpp"
#include "core/level.hpp"

namespace gridtools {
    namespace stencil {
        namespace dump_backend {
            using nlohmann::json;

            template <class T>
            auto get_type_name() {
                return boost::core::demangle(typeid(T).name());
            }

            inline auto from(core::parallel) { return "parallel"; }
            inline auto from(core::backward) { return "backward"; }
            inline auto from(core::forward) { return "forward"; }
            inline auto from(cache_type::ij) { return "ij"; }
            inline auto from(cache_type::k) { return "k"; }
            inline auto from(cache_io_policy::fill) { return "fill"; }
            inline auto from(cache_io_policy::flush) { return "flush"; }

            inline json dim_extent(int_t minus, int_t plus) { return {{"minus", minus}, {"plus", plus}}; }

            template <int_t IMinus, int_t IPlus, int_t JMinus, int_t JPlus, int_t KMinus, int_t KPlus>
            json from(extent<IMinus, IPlus, JMinus, JPlus, KMinus, KPlus>) {
                return {{"i", dim_extent(IMinus, IPlus)},
                    {"j", dim_extent(JMinus, JPlus)},
                    {"k", dim_extent(KMinus, KPlus)}};
            }

            template <uint_t Splitter, int_t Offset, int_t Limit>
            json from(core::level<Splitter, Offset, Limit>) {
                return {{"splitter", Splitter}, {"offset", Offset}};
            }

            template <class From, class To>
            json from(core::interval<From, To>) {
                return {{"from", from(From())}, {"to", from(To())}};
            }

            template <class Plh>
            auto from_plh(Plh) {
                return (core::is_tmp_arg<Plh>() ? "tmp" : "arg") + std::to_string(Plh::value);
            }

            template <template <class...> class L,
                template <class...>
                class LL,
                class Plh,
                class... Caches,
                class IsTmp,
                class Data,
                class NumColors,
                class IsConst,
                class Extent,
                class... CacheIoPolicies>
            json from(
                be_api::plh_info<L<Plh, Caches...>, IsTmp, Data, NumColors, IsConst, Extent, LL<CacheIoPolicies...>>) {
                json res = {{"plh", from_plh(Plh())},
                    {"caches", json::array({from(Caches())...})},
                    {"is_tmp", IsTmp::value},
                    {"data", get_type_name<Data>()},
                    {"is_const", IsConst::value},
                    {"extent", from(Extent())},
                    {"cache_io_policies", json::array({from(CacheIoPolicies())...})}};
                if (IsTmp::value)
                    res["num_colors"] = NumColors::value;
                return res;
            }

            template <template <class...> class L, class Plh, class... Caches>
            json from_arg(L<Plh, Caches...>) {
                return {{"plh", from_plh(Plh())}, {"caches", json::array({from(Caches())...})}};
            }

            template <class F>
            json from_fun(F) {
                return {{"functor", get_type_name<F>()}};
            }

            template <class F, class Interval>
            json from_fun(core::bound_functor<F, Interval>) {
                return {{"functor", get_type_name<F>()}, {"interval", from(Interval())}};
            }

            template <template <class...> class L, template <class...> class LL, class Fun, class... Args>
            json from_fun_call(L<Fun, LL<Args...>>) {
                json res = from_fun(Fun());
                res["args"] = json::array({from_arg(Args())...});
                return res;
            }

            template <template <class...> class L,
                template <class...>
                class LL,
                class... FunCalls,
                class Interval,
                class... PlhInfos,
                class Extent,
                class Execution,
                class NeedSync>
            json from(be_api::cell<L<FunCalls...>, Interval, LL<PlhInfos...>, Extent, Execution, NeedSync>) {
                return {{"fun_calls", json::array({from_fun_call(FunCalls())...})},
                    {"interval", from(Interval())},
                    {"plh_infos", json::array({from(PlhInfos())...})},
                    {"extent", from(Extent())},
                    {"execution", from(Execution())},
                    {"need_sync", NeedSync::value}};
            }

            template <template <class...> class L, class... Cells>
            auto from_row(L<Cells...>) {
                return json::array({from(Cells())...});
            }

            template <template <class...> class L, class... Rows>
            auto from_matrix(L<Rows...>) {
                return json::array({from_row(Rows())...});
            }

            template <template <class...> class L, class... Matrices>
            auto from_matrices(L<Matrices...>) {
                return json::array({from_matrix(Matrices())...});
            }

            struct dump {
                std::ostream &m_sink;
                template <class Spec, class... Ts>
                friend void gridtools_backend_entry_point(dump obj, Spec, Ts &&...) {
                    obj.m_sink << from_matrices(be_api::make_fused_view<Spec>()) << std::endl;
                }
            };
        } // namespace dump_backend
        using dump_backend::dump;
    } // namespace stencil
} // namespace gridtools
