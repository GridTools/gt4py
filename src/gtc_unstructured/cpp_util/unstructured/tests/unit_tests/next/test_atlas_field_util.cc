#include <gridtools/next/atlas_field_util.hpp>

#include "util/atlas_util.hpp"
#include <array_fwd.h>
#include <gridtools/sid/concept.hpp>

#include <gtest/gtest.h>

namespace {
    TEST(atlas_field_as_sid, edge_k_field) {
        namespace tu = gridtools::tuple_util;

        auto mesh = atlas_util::make_mesh();

        auto k_size = 10;
        auto edge_field = atlas_util::make_edge_field<double>(mesh, k_size);
        //   auto view = atlas::array::make_view<double, 3>(edge_field);

        auto testee = gridtools::next::atlas_util::as<edge, dim::k>::with_type<double>{}(edge_field);
        using testee_t = decltype(testee);

        static_assert(gridtools::sid::concept_impl_::is_sid<testee_t>{});

        using strides_t = gridtools::sid::strides_type<testee_t>;
        static_assert(tu::size<strides_t>() == 2, "");
        static_assert(gridtools::has_key<strides_t, edge>{});
        static_assert(gridtools::has_key<strides_t, dim::k>{});

        [[maybe_unused]] auto strides = gridtools::sid::get_strides(testee);
    }

    TEST(atlas_field_as_sid, invalid_location_type) {
        auto mesh = atlas_util::make_mesh();

        auto k_size = 10;
        auto edge_field = atlas_util::make_edge_field<double>(mesh, k_size);

        ASSERT_ANY_THROW((gridtools::next::atlas_util::as<cell, dim::k>::with_type<double>{}(edge_field)));
    }

    TEST(atlas_field_as_data_store, edge_k_field) {
        namespace tu = gridtools::tuple_util;

        auto mesh = atlas_util::make_mesh();

        auto k_size = 10;
        auto edge_field = atlas_util::make_edge_field<double>(mesh, k_size);

        auto testee = gridtools::next::atlas_util::as_data_store<edge, dim::k>::with_type<double>{}(edge_field);
        using testee_t = decltype(testee);

        static_assert(gridtools::sid::concept_impl_::is_sid<testee_t>{});

        using strides_t = gridtools::sid::strides_type<testee_t>;
        static_assert(tu::size<strides_t>() == 2, "");
        static_assert(gridtools::has_key<strides_t, edge>{});
        static_assert(gridtools::has_key<strides_t, dim::k>{});

        [[maybe_unused]] auto strides = gridtools::sid::get_strides(testee);
    }

} // namespace
