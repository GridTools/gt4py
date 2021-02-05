#include "../unstructured.hpp"
#include "gridtools/next/mesh.hpp"
#include <gridtools/sid/rename_dimensions.hpp>
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

namespace gridtools::next::test_helper {

    namespace make_field_impl_ {

        template <class T>
        struct builder {
            auto operator()(std::size_t size) const {
                return gridtools::storage::builder<storage_trait>.template type<T>().template dimensions(size);
            }
        };
    } // namespace make_field_impl_

    template <class T, class Location, class Mesh>
    auto make_field(Mesh const &mesh) {
        // TODO seems it would be nice if we can adapt the data_store builder to create SIDs with named dimension
        return sid::rename_numbered_dimensions<Location> // TODO
            (make_field_impl_::builder<T>{}(connectivity::size(mesh::connectivity<meta::list<Location>>(mesh)))());
    }
} // namespace gridtools::next::test_helper
