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

    template <class T, class... Sizes>
    auto make_field(Sizes... sizes) {
        return gridtools::storage::builder<storage_trait>.template type<T>().template dimensions(sizes...)();
    }
} // namespace gridtools::usid::test_helper
