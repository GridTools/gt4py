#include "${STENCIL_IMPL_SOURCE}"
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

#include <gtest/gtest.h>
#include <tuple>

// ```python
// out: SparseField[E2V, dtype]
// for e in edges(mesh):
//     out = (in[v] for v in vertices(e))
// ```

namespace gridtools::usid {

    using namespace gridtools::usid::test_helper;

    TEST(regression, vertex2edge) {
        auto in = test_helper::make_field<double>(simple_mesh::vertices);

        auto view = in->host_view();
        //  1   1   1 (1)
        //  1   2   1 (1)
        //  1   1   1 (1)
        // (1) (1) (1)
        for (std::size_t i = 0; i < 9; ++i)
            view(i) = 1;
        view(4) = 2;

        auto out = test_helper::make_field<double>(simple_mesh::edges, 2);

        sten({-1, simple_mesh::edges, -1, 1}, simple_mesh{}.e2v())(
            in, sid::rename_dimensions<integral_constant<int, 1>, dim::s>(out));

        //   x (1,1) x (1,1) x (1,1)
        // (1,1)   (1,2)   (1,1)
        //   x (1,2) x (2,1) x (1,1)
        // (1,1)   (2,1)   (1,1)
        //   x (1,1) x (1,1) x (1,1)
        // (1,1)   (1,1)   (1,1)

        auto out_view = out->const_host_view();
        for (std::size_t i = 0; i < 18; ++i) {
            if (i == 3 || i == 4 || i == 10 || i == 13)
                continue;
            EXPECT_DOUBLE_EQ(1, out_view(i, 0)) << i;
            EXPECT_DOUBLE_EQ(1, out_view(i, 1)) << i;
        }
        EXPECT_DOUBLE_EQ(1, out_view(3, 0));
        EXPECT_DOUBLE_EQ(2, out_view(3, 1));
        EXPECT_DOUBLE_EQ(2, out_view(4, 0));
        EXPECT_DOUBLE_EQ(1, out_view(4, 1));
        EXPECT_DOUBLE_EQ(1, out_view(10, 0));
        EXPECT_DOUBLE_EQ(2, out_view(10, 1));
        EXPECT_DOUBLE_EQ(2, out_view(13, 0));
        EXPECT_DOUBLE_EQ(1, out_view(13, 1));
    }
} // namespace gridtools::usid
