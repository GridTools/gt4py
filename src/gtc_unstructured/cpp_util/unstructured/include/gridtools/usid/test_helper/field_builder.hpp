#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <type_traits>

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools::usid::test_helper {

    namespace make_field_impl_ {

        template <class T>
        struct builder {
            auto operator()(std::size_t size) const {
                return gridtools::storage::builder<storage_trait>.template type<T>().template dimensions(size);
            }
        };
    } // namespace make_field_impl_

    template <class T>
    auto make_field(std::size_t size) {
        return make_field_impl_::builder<T>{}(size)();
    }

    namespace make_sparse_field_impl_ {

        template <class T>
        struct builder {
            auto operator()(std::size_t size, std::size_t max_neighbors) const {
                return gridtools::storage::builder<storage_trait>.template type<T>().template dimensions(
                    size, max_neighbors);
            }
        };
    } // namespace make_sparse_field_impl_

    template <class T>
    auto make_sparse_field(std::size_t size, std::size_t max_neighbors) {
        return sid::rename_dimensions<gridtools::integral_constant<int_t, 1>, dim::s>(
            make_sparse_field_impl_::builder<T>{}(size, max_neighbors)());
    }
} // namespace gridtools::usid::test_helper
