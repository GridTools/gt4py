#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

namespace gridtools::usid {
    using namespace gridtools::usid::test_helper;

    TEST(regression, cell2cell) {
        std::size_t k_size = 1;

        auto in = test_helper::make_field<double>(simple_mesh::cells, k_size);

        auto view = in->host_view();
        // 1 1 1
        // 1 2 1
        // 1 1 1
        for (std::size_t i = 0; i < simple_mesh::cells; ++i)
            for (std::size_t k = 0; k < k_size; ++k)
                view(i, k) = 1;
        for (std::size_t k = 0; k < k_size; ++k)
            view(4, k) = 2;

        auto out = test_helper::make_field<double>(simple_mesh::cells, k_size);
        sten({-1, -1, test_helper::simple_mesh::cells, k_size}, simple_mesh{}.c2c())(in, out);

        // 8  9  8
        // 9 12  9
        // 8  9  8
        auto out_view = out->const_host_view();
        for (std::size_t k = 0; k < k_size; ++k) {
            EXPECT_DOUBLE_EQ(8, out_view(0, k));
            EXPECT_DOUBLE_EQ(9, out_view(1, k));
            EXPECT_DOUBLE_EQ(8, out_view(2, k));
            EXPECT_DOUBLE_EQ(9, out_view(3, k));
            EXPECT_DOUBLE_EQ(12, out_view(4, k));
            EXPECT_DOUBLE_EQ(9, out_view(5, k));
            EXPECT_DOUBLE_EQ(8, out_view(6, k));
            EXPECT_DOUBLE_EQ(9, out_view(7, k));
            EXPECT_DOUBLE_EQ(8, out_view(8, k));
        }
    }
} // namespace gridtools::usid
