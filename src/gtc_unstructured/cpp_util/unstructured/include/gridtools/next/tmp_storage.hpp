#include "unstructured.hpp"
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/contiguous.hpp>

namespace gridtools {
    namespace next {
        namespace tmp_impl_ {
            template <class LocationType>
            auto sizes(int_t unstructured_size, int_t k_size) {
                using value_t = typename hymap::keys<LocationType, dim::k>::template values<int_t, int_t>;
                return value_t(unstructured_size, k_size);
            }
        } // namespace tmp_impl_

        template <class LocationType, class Data, class Allocator>
        auto make_simple_tmp_storage(int_t unstructured_size, int_t k_size, Allocator &alloc) {
            return sid::make_contiguous<Data, int_t>(alloc, tmp_impl_::sizes<LocationType>(unstructured_size, k_size));
        }
    } // namespace next
} // namespace gridtools
