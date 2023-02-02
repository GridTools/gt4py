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
#include <utility>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        namespace multi_shift_impl_ {
            template <class Dims>
            struct for_each_dim;

            template <template <class...> class L, class... Dims>
            struct for_each_dim<L<Dims...>> {
                template <class Ptr, class Strides, class Offsets>
                GT_FUNCTION for_each_dim(Ptr &ptr, Strides const &strides, Offsets offsets) {
                    (..., shift(ptr, get_stride<Dims>(strides), gridtools::host_device::at_key<Dims>(offsets)));
                }
            };

            template <template <class...> class L>
            struct for_each_dim<L<>> {
                template <class Ptr, class Strides, class Offsets>
                GT_FUNCTION for_each_dim(Ptr &, Strides const &, Offsets) {}
            };

            template <class Arg, class Dims>
            struct for_each_dim_a;

            template <class Arg, template <class...> class L, class... Dims>
            struct for_each_dim_a<Arg, L<Dims...>> {
                template <class Ptr, class Strides, class Offsets>
                GT_FUNCTION for_each_dim_a(Ptr &ptr, Strides const &strides, Offsets offsets) {
                    (...,
                        shift(ptr,
                            get_stride_element<Arg, Dims>(strides),
                            gridtools::host_device::at_key<Dims>(offsets)));
                }
            };

            template <class Arg, template <class...> class L>
            struct for_each_dim_a<Arg, L<>> {
                template <class Ptr, class Strides, class Offsets>
                GT_FUNCTION for_each_dim_a(Ptr &, Strides const &, Offsets) {}
            };
        } // namespace multi_shift_impl_

        /**
         *   A helper the invokes `sid::shift` in several dimensions.
         *   `offsets` should be a hymap of individual offsets.
         */
        template <class Ptr, class Strides, class Offsets>
        GT_FUNCTION void multi_shift(Ptr &ptr, Strides const &strides, Offsets offsets) {
            multi_shift_impl_::for_each_dim<get_keys<Offsets>>(ptr, strides, std::move(offsets));
        }

        template <class Ptr, class Strides, class Offsets>
        GT_FUNCTION Ptr multi_shifted(Ptr ptr, Strides const &strides, Offsets offsets) {
            multi_shift(ptr, strides, std::move(offsets));
            return ptr;
        }

        /**
         *   Variation of multi_shift that works with the strides of composite sid.
         */
        template <class Arg, class Ptr, class Strides, class Offsets>
        GT_FUNCTION void multi_shift(Ptr &ptr, Strides const &strides, Offsets offsets) {
            multi_shift_impl_::for_each_dim_a<Arg, get_keys<Offsets>>(ptr, strides, std::move(offsets));
        }

        template <class Arg, class Ptr, class Strides, class Offsets>
        GT_FUNCTION Ptr multi_shifted(Ptr ptr, Strides const &strides, Offsets offsets) {
            multi_shift<Arg>(ptr, strides, std::move(offsets));
            return ptr;
        }
    } // namespace sid
} // namespace gridtools
