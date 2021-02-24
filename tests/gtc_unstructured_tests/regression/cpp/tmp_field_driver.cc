#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

namespace gridtools::usid {

    using namespace gridtools::usid::test_helper;

    TEST(regression, temporary) {
        test_helper::simple_mesh mesh;

        auto in = test_helper::make_field<double>(simple_mesh::cells, 1);

        auto view = in->host_view();
        //  1   2   3
        //  4   5   6
        //  7   8   9
        for (std::size_t i = 0; i < 9; ++i)
            view(i, 0) = i;

        auto out = test_helper::make_field<double>(simple_mesh::cells, 1);
        sten({-1, -1, test_helper::simple_mesh::cells, 1})(in, out);

        //  1   2   3
        //  4   5   6
        //  7   8   9
        auto out_view = out->const_host_view();
        EXPECT_DOUBLE_EQ(0, out_view(0, 0));
        EXPECT_DOUBLE_EQ(1, out_view(1, 0));
        EXPECT_DOUBLE_EQ(2, out_view(2, 0));
        EXPECT_DOUBLE_EQ(3, out_view(3, 0));
        EXPECT_DOUBLE_EQ(4, out_view(4, 0));
        EXPECT_DOUBLE_EQ(5, out_view(5, 0));
        EXPECT_DOUBLE_EQ(6, out_view(6, 0));
        EXPECT_DOUBLE_EQ(7, out_view(7, 0));
        EXPECT_DOUBLE_EQ(8, out_view(8, 0));
    }
} // namespace gridtools::usid
