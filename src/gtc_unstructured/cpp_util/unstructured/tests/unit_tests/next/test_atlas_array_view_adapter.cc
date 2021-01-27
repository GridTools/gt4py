#include <gridtools/common/tuple_util.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/sid/concept.hpp>

#include <gtest/gtest.h>

namespace {
    TEST(atlas_array_view, sid) {
        namespace gsid = gridtools::sid;
        namespace tu = gridtools::tuple_util;

        atlas::array::ArrayT<int> arr(3, 4, 5, 6);

        auto testee = atlas::array::make_view<int, 4>(arr);

        using testee_t = decltype(testee);

        static_assert(gsid::concept_impl_::is_sid<testee_t>(), "");
        static_assert(std::is_same<gsid::ptr_type<testee_t>, int *>(), "");
        // TODO do we care?
        //   static_assert(std::is_same<gsid::ptr_diff_type<testee_t>, long>(), "");

        // TODO need to define a proper strides_kind
        // static_assert(std::is_same<gsid::strides_kind<testee_t>,
        //                            typename testee_t::element_type::kind_t>(),
        //               "");

        using strides_t = gsid::strides_type<testee_t>;

        static_assert(tu::size<strides_t>() == 4, "");

        EXPECT_EQ(&testee(0, 0, 0, 0), gsid::get_origin(testee)());

        auto strides = gsid::get_strides(testee);
        auto &&expected_strides = testee.strides();

        EXPECT_EQ(expected_strides[0], tu::get<0>(strides));
        EXPECT_EQ(expected_strides[1], tu::get<1>(strides));
        EXPECT_EQ(expected_strides[2], tu::get<2>(strides));
        EXPECT_EQ(expected_strides[3], tu::get<3>(strides));

        auto lower_bounds = gsid::get_lower_bounds(testee);
        EXPECT_EQ(0, tu::get<0>(lower_bounds));
        EXPECT_EQ(0, tu::get<1>(lower_bounds));
        EXPECT_EQ(0, tu::get<2>(lower_bounds));
        EXPECT_EQ(0, tu::get<3>(lower_bounds));

        auto &&lengths = testee.shape();
        auto upper_bounds = gsid::get_upper_bounds(testee);
        EXPECT_EQ(lengths[0], tu::get<0>(upper_bounds));
        EXPECT_EQ(lengths[1], tu::get<1>(upper_bounds));
        EXPECT_EQ(lengths[2], tu::get<2>(upper_bounds));
        EXPECT_EQ(lengths[3], tu::get<3>(upper_bounds));
    }
} // namespace
