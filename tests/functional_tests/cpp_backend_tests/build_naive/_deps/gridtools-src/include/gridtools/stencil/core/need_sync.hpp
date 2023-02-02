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

#include "../../meta.hpp"
#include "../common/caches.hpp"
#include "esf_metafunctions.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace need_sync_impl_ {
                template <class DirtyPlhs>
                struct is_dirty_f {
                    template <class Item, class Extent = typename meta::second<Item>::extent_t>
                    using apply = std::bool_constant<meta::st_contains<DirtyPlhs, meta::first<Item>>::value &&
                                                     (Extent::iminus::value != 0 || Extent::iplus::value != 0 ||
                                                         Extent::jminus::value != 0 || Extent::jplus::value != 0)>;
                };

                template <class Esf, class DirtyPlhs>
                using has_dirty_args = typename meta::any_of<is_dirty_f<DirtyPlhs>::template apply,
                    meta::zip<typename Esf::args_t, esf_param_list<Esf>>>::type;

                template <class State,
                    class Esf,
                    class DirtyPlhs = meta::second<State>,
                    class NeedSync = has_dirty_args<Esf, DirtyPlhs>,
                    class OutPlhs = esf_get_w_args_per_functor<Esf>,
                    class NewDirtys = meta::if_<NeedSync, OutPlhs, meta::dedup<meta::concat<DirtyPlhs, OutPlhs>>>>
                using folding_fun = meta::list<meta::push_back<meta::first<State>, NeedSync>, NewDirtys>;

                template <class CacheInfo>
                using get_cache_types = typename CacheInfo::cache_types_t;

                template <class CacheMap>
                using has_ij_cache =
                    meta::st_contains<meta::dedup<meta::flatten<meta::transform<get_cache_types, CacheMap>>>,
                        cache_type::ij>;
            } // namespace need_sync_impl_

            template <class Esfs,
                class CacheMap,
                class InitialState = meta::list<meta::list<>, meta::list<>>,
                class FinalState = meta::foldl<need_sync_impl_::folding_fun, InitialState, Esfs>>
            using need_sync = meta::replace_at_c<meta::first<FinalState>, 0, need_sync_impl_::has_ij_cache<CacheMap>>;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
