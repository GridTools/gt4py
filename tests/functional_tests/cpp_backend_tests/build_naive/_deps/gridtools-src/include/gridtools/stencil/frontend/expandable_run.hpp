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

#include <type_traits>
#include <vector>

#include "../../meta.hpp"
#include "run.hpp"

namespace gridtools {
    namespace stencil {
        namespace expandalble_frontend_impl_ {
            template <class T>
            struct expandable : T {};

            template <size_t, class T>
            struct expanded : T {};

            template <size_t I>
            struct arg : integral_constant<size_t, I> {};

            template <size_t, size_t>
            struct expanded_arg {};

            template <size_t I, class Field>
            using make_arg =
                meta::if_<meta::is_instantiation_of<std::vector, std::decay_t<Field>>, expandable<arg<I>>, arg<I>>;

            template <class Field>
            size_t get_field_size(Field const &) {
                return size_t(-1);
            }
            template <class T, class A>
            size_t get_field_size(std::vector<T, A> const &field) {
                return field.size();
            }

            inline size_t get_expandable_size_impl() { return (size_t)-1; }

            template <class... Sizes>
            size_t get_expandable_size_impl(size_t first, Sizes... rest) {
                if (first == (size_t)-1)
                    return get_expandable_size_impl(rest...);
                assert(get_expandable_size_impl(rest...) == first || get_expandable_size_impl(rest...) == (size_t)-1);
                return first;
            }

            template <class... Fields>
            size_t get_expandable_size(Fields const &...fields) {
                size_t res = get_expandable_size_impl(get_field_size(fields)...);
                return res == size_t(-1) ? 1 : res;
            }

            namespace lazy {
                template <class...>
                struct convert_plh;

                template <class I, class Plh>
                struct convert_plh<I, Plh> {
                    using type = Plh;
                };

                template <class I, class Plh>
                struct convert_plh<I, expandable<Plh>> {
                    using type = expanded<I::value, Plh>;
                };

            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(convert_plh, class... Ts, Ts...);

            template <class Esf>
            struct convert_esf {
                template <class I>
                using apply = core::esf_replace_args<Esf,
                    meta::transform<meta::curry<convert_plh, I>::template apply, typename Esf::args_t>>;
            };

            template <class Factor>
            struct expand_esf_f {
                template <class Esf>
                using apply = meta::transform<convert_esf<Esf>::template apply, meta::make_indices<Factor>>;
            };

            template <class I, class Cache>
            struct convert_cache;

            template <class I, class Plh, class... Params>
            struct convert_cache<I, core::cache_info<Plh, Params...>> {
                using type = core::cache_info<convert_plh<I, Plh>, Params...>;
            };

            template <class IndexAndCache>
            using convert_cache_f =
                typename convert_cache<meta::first<IndexAndCache>, meta::second<IndexAndCache>>::type;

            template <class...>
            struct convert_mss;

            template <class Factor, class ExecutionType, class Esfs, class Caches>
            struct convert_mss<Factor, core::mss_descriptor<ExecutionType, Esfs, Caches>> {
                using esfs_t = meta::flatten<meta::transform<expand_esf_f<Factor>::template apply, Esfs>>;

                using indices_and_caches_t = meta::cartesian_product<meta::make_indices<Factor>, Caches>;
                using caches_t = meta::dedup<meta::transform<convert_cache_f, indices_and_caches_t>>;

                using type = core::mss_descriptor<ExecutionType, esfs_t, caches_t>;
            };

            template <class Factor, class Spec>
            using expand_spec =
                meta::transform<meta::curry<meta::force<convert_mss>::apply, Factor>::template apply, Spec>;

            template <class Plh, size_t... Js, class T, class A>
            meta::rename<hymap::keys<expanded<Js, Plh>...>::template values,
                meta::repeat_c<sizeof...(Js), meta::list<T const &>>>
            make_data_store_map_impl(std::index_sequence<Js...>, size_t offset, std::vector<T, A> const &field) {
                return {field[offset + Js]...};
            }

            template <size_t Factor, class Plh, class T, class A>
            auto make_data_store_map2(size_t offset, std::vector<T, A> const &field) {
                return make_data_store_map_impl<Plh>(std::make_index_sequence<Factor>(), offset, field);
            }

            template <size_t Factor, class Plh, class Field>
            typename hymap::keys<Plh>::template values<Field const &> make_data_store_map2(size_t, Field const &field) {
                return {field};
            }

            template <size_t Factor, size_t... Is, class... Fields>
            auto make_data_store_map(size_t offset, Fields const &...fields) {
                return hymap::concat(make_data_store_map2<Factor, arg<Is>>(offset, fields)...);
            }

            template <size_t Factor, class Spec, size_t... Is, class Backend, class Grid, class... Fields>
            void expanded_run(Backend be, Grid const &grid, size_t offset, const Fields &...fields) {
                core::call_entry_point_f<expand_spec<std::integral_constant<size_t, Factor>, Spec>>()(
                    std::forward<Backend>(be), grid, make_data_store_map<Factor, Is...>(offset, fields...));
            }

            template <size_t Factor, class Comp, class Backend, class Grid, class... Fields, size_t... Is>
            auto run_impl(Comp comp, Backend &&be, Grid const &grid, std::index_sequence<Is...>, Fields &&...fields)
                -> std::void_t<decltype(comp(make_arg<Is, Fields>()...))> {
                using spec_t = decltype(comp(make_arg<Is, Fields>()...));
                static_assert(meta::is_instantiation_of<frontend_impl_::spec, spec_t>::value,
                    "Invalid stencil composition specification.");
                static_assert(
                    meta::is_instantiation_of<core::interval, typename Grid::interval_t>::value, "Invalid grid.");
                using functors_t = meta::transform<meta::first, meta::flatten<meta::transform<meta::second, spec_t>>>;
                static_assert(
                    meta::all_of<frontend_impl_::check_valid_apply_overloads<typename Grid::interval_t>::template apply,
                        functors_t>::value,
                    "Invalid stencil operator detected.");

                size_t size = get_expandable_size(fields...);
                size_t offset = 0;
                for (; size - offset >= Factor; offset += Factor)
                    expanded_run<Factor, spec_t, Is...>(be, grid, offset, fields...);
                for (; offset < size; ++offset)
                    expanded_run<1, spec_t, Is...>(be, grid, offset, fields...);
            }

            template <size_t, class... Ts>
            void run_impl(Ts...) {
                static_assert(sizeof...(Ts) < 0, "Unexpected gridtools::stencil::expandable_run first argument.");
            }

            template <size_t Factor, class Comp, class Backend, class Grid, class... Fields>
            void expandable_run(Comp comp, Backend &&be, Grid const &grid, Fields &&...fields) {
                run_impl<Factor>(comp,
                    std::forward<Backend>(be),
                    grid,
                    std::index_sequence_for<Fields...>(),
                    std::forward<Fields>(fields)...);
            }
        } // namespace expandalble_frontend_impl_
        using expandalble_frontend_impl_::expandable;
        using expandalble_frontend_impl_::expandable_run;
    } // namespace stencil
} // namespace gridtools
