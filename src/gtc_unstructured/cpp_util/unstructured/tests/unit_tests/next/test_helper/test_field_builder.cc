#include <array>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/test_helper/field_builder.hpp>
#include <gridtools/next/test_helper/simple_mesh.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <utility>

namespace {

    TEST(field_builder, cell_field) {
        gridtools::next::test_helper::simple_mesh mesh;

        auto field = gridtools::next::test_helper::make_field<double, cell>(mesh);

        static_assert(std::is_same<double *, gridtools::sid::ptr_type<decltype(field)>>{});
        auto c2c = gridtools::next::mesh::connectivity<std::tuple<cell, cell>>(mesh);
        ASSERT_EQ(
            gridtools::next::connectivity::size(c2c), gridtools::at_key<cell>(gridtools::sid::get_upper_bounds(field)));
    }

    // TODO consider moving to a different file
    // TODO we need to understand where we should use SID and where data_store concepts
    using namespace gridtools::sid;
    using namespace gridtools::next;
    using gridtools::at_key;
    namespace tu = gridtools::tuple_util;

    struct in_tag;
    struct out_tag;
    struct connectivity_tag;

    template <class DimTag, class Sid, class Fun>
    void make_full_loop(Sid &&sid, Fun &&fun) {
        auto ptr = get_origin(std::forward<Sid>(sid))();
        auto strides = get_strides(std::forward<Sid>(sid));

        make_loop<DimTag>(at_key<DimTag>(get_upper_bounds(std::forward<Sid>(sid))))(std::forward<Fun>(fun))(
            ptr, strides);
    }

    TEST(simple_mesh_regression, cell2cell_reduction) {
        test_helper::simple_mesh mesh;
        auto c2c = mesh::connectivity<std::tuple<cell, cell>>(mesh);

        auto in = test_helper::make_field<double, cell>(mesh);
        auto out = gridtools::next::test_helper::make_field<double, cell>(mesh);

        make_full_loop<cell>(in, [](auto ptr, auto const &) { *ptr = 1; });

        auto primary_fields =
            tu::make<composite::keys<out_tag, connectivity_tag>::values>(out, connectivity::neighbor_table(c2c));
        {
            auto ptrs = get_origin(primary_fields)();
            auto strides = get_strides(primary_fields);

            // variant 1:
            // for (std::size_t c = 0; c < at_key<cell>(get_upper_bounds(out)); ++c) {
            //     *at_key<out_tag>(ptrs) = 0;
            //     for (std::size_t i = 0; i < connectivity::max_neighbors(c2c); ++i) {
            //         auto in_ptr = get_origin(in)();
            //         auto in_strides = get_strides(in);
            //         shift(in_ptr, at_key<cell>(in_strides), *at_key<connectivity_tag>(ptrs));
            //         *at_key<out_tag>(ptrs) += *in_ptr;
            //         shift(ptrs, gridtools::at_key<neighbor>(strides), 1);
            //     }
            //     shift(ptrs, at_key<neighbor>(strides), -connectivity::max_neighbors(c2c));
            //     shift(ptrs, at_key<cell>(strides), 1);
            // }

            // variant 2:
            make_loop<cell>(gridtools::at_key<cell>(get_upper_bounds(out)))(
                [&c2c, &in](auto ptrs, auto const &strides) {
                    *at_key<out_tag>(ptrs) = 0;
                    for (std::size_t i = 0; i < connectivity::max_neighbors(c2c); ++i) {
                        auto in_ptr = get_origin(in)();
                        auto in_strides = get_strides(in);
                        shift(in_ptr, at_key<cell>(in_strides), *at_key<connectivity_tag>(ptrs));
                        *at_key<out_tag>(ptrs) += *in_ptr;
                        shift(ptrs, gridtools::at_key<neighbor>(strides), 1);
                    }
                })(ptrs, strides);
        }

        // variant 3:
        // TODO requires changes in the make_full_loop
        // auto loop_body = [&c2c, &in](auto ptrs, auto const &strides) {
        //     *at_key<out_tag>(ptrs) = 0;
        //     for (std::size_t i = 0; i < connectivity::max_neighbors(c2c); ++i) {
        //         auto in_ptr = get_origin(in)();
        //         auto in_strides = get_strides(in);
        //         shift(in_ptr, at_key<cell>(in_strides), *at_key<connectivity_tag>(ptrs));
        //         *at_key<out_tag>(ptrs) += *in_ptr;
        //         shift(ptrs, gridtools::at_key<neighbor>(strides), 1);
        //     }
        // };
        // make_full_loop<cell>(primary_fields, loop_body);

        make_full_loop<cell>(out, [](auto ptr, auto const &) { ASSERT_EQ(4, *ptr); });
    }
} // namespace
