#include "gridtools/common/integral_constant.hpp"
#include <gridtools/next/mesh.hpp>

#include <gtest/gtest.h>
#include <type_traits>

// TODO test concept

using namespace gridtools::next;

namespace {
    namespace fake {
        struct fake_connectivity {
            inline friend std::size_t connectivity_size(fake_connectivity const &) { return 1; }
            inline friend gridtools::integral_constant<int, 2> connectivity_max_neighbors(fake_connectivity const &) {
                return {};
            }
            inline friend gridtools::integral_constant<int, -1> connectivity_skip_value(fake_connectivity const &) {
                return {};
            }
        };
    } // namespace fake

    TEST(connectivity, make_info) {
        auto info = connectivity::extract_info(fake::fake_connectivity{});
        ASSERT_EQ(1, info.size);
        static_assert(std::is_same_v<gridtools::integral_constant<int, 2>, decltype(info.max_neighbors)>);
        static_assert(std::is_same_v<gridtools::integral_constant<int, -1>, decltype(info.skip_value)>);
    }
} // namespace
