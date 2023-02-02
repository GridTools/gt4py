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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "dim.hpp"

namespace gridtools {
    namespace stencil {
        /**
         * Class to specify access extents for stencil functions
         */
        template <int_t IMinus = 0,
            int_t IPlus = 0,
            int_t JMinus = 0,
            int_t JPlus = 0,
            int_t KMinus = 0,
            int_t KPlus = 0>
        struct extent {
            using iminus = integral_constant<int_t, IMinus>;
            using iplus = integral_constant<int_t, IPlus>;
            using jminus = integral_constant<int_t, JMinus>;
            using jplus = integral_constant<int_t, JPlus>;
            using kminus = integral_constant<int_t, KMinus>;
            using kplus = integral_constant<int_t, KPlus>;

            static GT_FUNCTION iminus minus(dim::i) { return {}; }
            static GT_FUNCTION jminus minus(dim::j) { return {}; }
            static GT_FUNCTION kminus minus(dim::k) { return {}; }

            static GT_FUNCTION iplus plus(dim::i) { return {}; }
            static GT_FUNCTION jplus plus(dim::j) { return {}; }
            static GT_FUNCTION kplus plus(dim::k) { return {}; }

            template <class Dim, class Size>
            static GT_FUNCTION auto extend(Dim dim, Size size) {
                return size - minus(dim) + plus(dim);
            }
        };

        /**
         * Metafunction to check if a type is a extent
         */
        template <class>
        struct is_extent : std::false_type {};

        template <int_t... Is>
        struct is_extent<extent<Is...>> : std::true_type {};

        /**
         * Metafunction taking extents and yielding an extent containing them
         */
        namespace lazy {
            template <class...>
            struct enclosing_extent;
        }
        GT_META_DELEGATE_TO_LAZY(enclosing_extent, class... Extents, Extents...);
        namespace lazy {
            template <>
            struct enclosing_extent<> {
                using type = extent<>;
            };

            template <int_t... Is>
            struct enclosing_extent<extent<Is...>> {
                using type = extent<Is...>;
            };

            template <int_t IMinus1,
                int_t IPlus1,
                int_t JMinus1,
                int_t JPlus1,
                int_t KMinus1,
                int_t KPlus1,
                int_t IMinus2,
                int_t IPlus2,
                int_t JMinus2,
                int_t JPlus2,
                int_t KMinus2,
                int_t KPlus2>
            struct enclosing_extent<extent<IMinus1, IPlus1, JMinus1, JPlus1, KMinus1, KPlus1>,
                extent<IMinus2, IPlus2, JMinus2, JPlus2, KMinus2, KPlus2>> {
                using type = extent<(IMinus1 < IMinus2) ? IMinus1 : IMinus2,
                    (IPlus1 < IPlus2) ? IPlus2 : IPlus1,
                    (JMinus1 < JMinus2) ? JMinus1 : JMinus2,
                    (JPlus1 < JPlus2) ? JPlus2 : JPlus1,
                    (KMinus1 < KMinus2) ? KMinus1 : KMinus2,
                    (KPlus1 < KPlus2) ? KPlus2 : KPlus1>;
            };

            template <class... Extents>
            struct enclosing_extent : meta::lazy::combine<stencil::enclosing_extent, meta::list<Extents...>> {};

        } // namespace lazy

        template <class Extent>
        using to_horizontal_extent =
            extent<Extent::iminus::value, Extent::iplus::value, Extent::jminus::value, Extent::jplus::value>;

        /**
         * Metafunction taking two extents and yielding a extent which is the extension of one another
         */
        template <typename Extent1, typename Extent2>
        using sum_extent = extent<Extent1::iminus::value + Extent2::iminus::value,
            Extent1::iplus::value + Extent2::iplus::value,
            Extent1::jminus::value + Extent2::jminus::value,
            Extent1::jplus::value + Extent2::jplus::value,
            Extent1::kminus::value + Extent2::kminus::value,
            Extent1::kplus::value + Extent2::kplus::value>;
    } // namespace stencil
} // namespace gridtools
