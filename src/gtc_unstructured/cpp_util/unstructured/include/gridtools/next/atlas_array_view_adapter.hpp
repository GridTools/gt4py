#pragma once

#include <atlas/array.h>
#include <cstddef>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/simple_ptr_holder.hpp>
#include <utility>

/**
 * SID adapters for atlas::ArrayView.
 *
 * Probably not useful for unstructured as we cannot check the location type.
 * TODO: proper kind
 */

namespace atlas::array {
    namespace sid_adapter_impl_ {
        // TODO use more meta/tu helpers as an exercise (instead of direct
        // implementation)...

        template <class>
        struct to_hymap;

        template <std::size_t... I>
        struct to_hymap<std::index_sequence<I...>> {
            template <std::size_t II>
            struct intifier {
                using type = int;
            };

            auto operator()(idx_t const *strides) const {
                using keys_t = gridtools::hymap::keys<gridtools::integral_constant<int, I>...>;
                using hymap_t = typename keys_t::template values<typename intifier<I>::type...>;
                return hymap_t(strides[I]...);
            }
        };

        template <class>
        struct lower_bounds;

        template <std::size_t... I>
        struct lower_bounds<std::index_sequence<I...>> {
            template <std::size_t II>
            struct zero {
                using type = gridtools::integral_constant<int, 0>;
            };

            using keys_t = gridtools::hymap::keys<gridtools::integral_constant<int, I>...>;
            using type = typename keys_t::template values<typename zero<I>::type...>;
        };

        template <class T>
        using lower_bounds_t = typename lower_bounds<T>::type;
    } // namespace sid_adapter_impl_

    template <typename Value, int Rank>
    auto sid_get_origin(ArrayView<Value, Rank> &view) {
        return gridtools::sid::make_simple_ptr_holder(view.data());
    }

    template <typename Value, int Rank>
    auto sid_get_strides(ArrayView<Value, Rank> const &view) {
        return sid_adapter_impl_::to_hymap<std::make_index_sequence<Rank>>{}(view.strides());
    }

    template <typename Value, int Rank>
    std::tuple<Value, std::integral_constant<int, Rank>> /* TODO this is not a proper kind as we don't
                                                            guarantee same strides */
    sid_get_strides_kind(ArrayView<Value, Rank> const &);

    template <typename Value, int Rank>
    sid_adapter_impl_::lower_bounds_t<std::make_index_sequence<Rank>> sid_get_lower_bounds(
        ArrayView<Value, Rank> const &) {
        return {};
    }

    template <typename Value, int Rank>
    auto sid_get_upper_bounds(ArrayView<Value, Rank> const &view) {
        return sid_adapter_impl_::to_hymap<std::make_index_sequence<Rank>>{}(view.shape());
    }

} // namespace atlas::array
