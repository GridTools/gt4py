/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor in the grid point that IteratorDomain points to.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

#pragma once

#include <type_traits>
#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../../meta.hpp"
#include "../../../sid/multi_shift.hpp"
#include "../../common/extent.hpp"
#include "../../common/intent.hpp"
#include "expressions/expr_base.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace stage_impl_ {
                struct default_deref_f {
                    template <class Key, class T>
                    GT_FUNCTION decltype(auto) operator()(Key, T ptr) const {
                        return *ptr;
                    }
                };

                template <class Ptr, class Strides, class Keys, class Deref>
                struct evaluator {
                    Ptr const &m_ptr;
                    Strides const &m_strides;

                    template <class Accessor>
                    GT_FUNCTION decltype(auto) operator()(Accessor acc) const {
                        using key_t = meta::at_c<Keys, Accessor::index_t::value>;
                        return apply_intent<Accessor::intent_v>(Deref()(key_t(),
                            sid::multi_shifted<key_t>(host_device::at_key<key_t>(m_ptr), m_strides, std::move(acc))));
                    }

                    template <class Op, class... Ts>
                    GT_FUNCTION auto operator()(expr<Op, Ts...> arg) const {
                        return expressions::evaluation::value(*this, std::move(arg));
                    }
                };

                template <class Functor, class PlhMap>
                struct stage {
                    template <class Deref = void, class Ptr, class Strides>
                    GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides) const {
                        using deref_t = meta::if_<std::is_void<Deref>, default_deref_f, Deref>;
                        using eval_t = evaluator<Ptr, Strides, PlhMap, deref_t>;
                        eval_t eval{ptr, strides};
                        Functor::template apply<eval_t &>(eval);
                    }
                };
            } // namespace stage_impl_
            template <class... Ts>
            meta::curry<stage_impl_::stage> get_stage(Ts &&...);
        } // namespace cartesian
    }     // namespace stencil
} // namespace gridtools
