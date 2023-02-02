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
#include "../common/extent.hpp"
#include "esf_metafunctions.hpp"
#include "mss.hpp"

/** @file
    This file implements the metafunctions to perform data dependency analysis on a multi-stage computation (MSS).
    The idea is to assign to each placeholder used in the computation an extent that represents the values that need
    to be accessed by the stages of the computation in each iteration point. This "assignment" is done by using
    a compile time between placeholders and extents.
 */

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace compute_extents_metafunctions_impl_ {
                template <class Map, class Arg>
                using lookup_extent_map =
                    meta::rename<enclosing_extent, meta::pop_front<meta::mp_find<Map, Arg, meta::list<Arg, extent<>>>>>;

                template <class Map>
                struct lookup_extent_map_f {
                    template <class Arg>
                    using apply = lookup_extent_map<Map, Arg>;
                };

                template <intent Intent>
                struct has_intent {
                    template <class Item, class Param = meta::second<Item>>
                    using apply = std::bool_constant<Param::intent_v == Intent>;
                };

                template <class Esf>
                using get_arg_param_pairs = meta::zip<typename Esf::args_t, esf_param_list<Esf>>;

                namespace lazy {
                    template <class ArgParamPair>
                    struct get_out_arg : meta::lazy::first<ArgParamPair> {
                        using extent_t = typename meta::lazy::second<ArgParamPair>::type::extent_t;
                        static_assert(extent_t::iminus::value == 0 && extent_t::iplus::value == 0 &&
                                          extent_t::jminus::value == 0 && extent_t::jplus::value == 0,
                            "Horizontal extents of the outputs of ESFs are not all empty. All outputs must have empty "
                            "(horizontal) extents");
                    };
                } // namespace lazy
                GT_META_DELEGATE_TO_LAZY(get_out_arg, class T, T);

                template <class Extent>
                struct make_item_f {
                    template <class ArgParamPair, class Param = meta::second<ArgParamPair>>
                    using apply = meta::list<meta::first<ArgParamPair>, sum_extent<Extent, typename Param::extent_t>>;
                };

                namespace lazy {
                    template <class Esf, class ExtentMap, class Extent = typename Esf::extent_t>
                    struct get_esf_extent {
                        using type = Extent;
                    };

                    template <class Esf, class ExtentMap>
                    struct get_esf_extent<Esf, ExtentMap, void> {
                        using arg_param_pairs_t = get_arg_param_pairs<Esf>;
                        using out_args_t = meta::transform<compute_extents_metafunctions_impl_::get_out_arg,
                            meta::filter<has_intent<intent::inout>::apply, arg_param_pairs_t>>;
                        using extents_t = meta::transform<lookup_extent_map_f<ExtentMap>::template apply, out_args_t>;
                        using type = meta::rename<enclosing_extent, extents_t>;
                    };

                    template <class ExtentMap, class Esf, class Extent = typename Esf::extent_t>
                    struct process_esf {
                        using arg_param_pairs_t = get_arg_param_pairs<Esf>;
                        using new_items_t = meta::transform<make_item_f<Extent>::template apply, arg_param_pairs_t>;
                        using type = meta::foldl<meta::mp_insert, ExtentMap, new_items_t>;
                    };

                    template <class ExtentMap, class Esf>
                    struct process_esf<ExtentMap, Esf, void> {
                        using esf_extent_t = typename get_esf_extent<Esf, ExtentMap>::type;
                        using arg_param_pairs_t = get_arg_param_pairs<Esf>;
                        using new_items_t =
                            meta::transform<make_item_f<esf_extent_t>::template apply, arg_param_pairs_t>;
                        using type = meta::foldl<meta::mp_insert, ExtentMap, new_items_t>;
                    };
                } // namespace lazy
                GT_META_DELEGATE_TO_LAZY(get_esf_extent, (class Esf, class ExtentMap), (Esf, ExtentMap));
                GT_META_DELEGATE_TO_LAZY(process_esf, (class ExtentMap, class Esfs), (ExtentMap, Esfs));

                template <class Mss>
                using get_esfs_from_mss = typename Mss::esf_sequence_t;

                template <class Esfs>
                using get_extent_map_from_esfs = meta::foldr<process_esf, meta::list<>, Esfs>;

                template <class Mss>
                using get_extent_map_from_mss = get_extent_map_from_esfs<get_esfs_from_mss<Mss>>;

                template <class Msses>
                using get_extent_map_from_msses =
                    get_extent_map_from_esfs<meta::flatten<meta::transform<get_esfs_from_mss, Msses>>>;

            } // namespace compute_extents_metafunctions_impl_

            using compute_extents_metafunctions_impl_::get_esf_extent;
            using compute_extents_metafunctions_impl_::get_extent_map_from_mss;
            using compute_extents_metafunctions_impl_::get_extent_map_from_msses;
            using compute_extents_metafunctions_impl_::lookup_extent_map;
            using compute_extents_metafunctions_impl_::lookup_extent_map_f;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
