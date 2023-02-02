/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/gcl/halo_exchange.hpp>

#include <type_traits>
#include <vector>

#include <mpi.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/array_addons.hpp>
#include <gridtools/common/for_each.hpp>
#include <gridtools/common/layout_map.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/storage/builder.hpp>

#include <gcl_select.hpp>
#include <storage_select.hpp>

using namespace gridtools;

constexpr size_t num_dims = 3;
constexpr size_t num_fields = 3;
constexpr size_t num_halos = 2;

struct test_spec {
    int dims[num_dims];
    int halos[num_fields][num_dims][num_halos];
    int mpi_dims[num_dims];
};

using value_type = array<int, num_dims + 1>;

value_type none() { return {-1, -1, -1, -1}; }

template <class Testee, class... Fields>
void exchange(std::false_type, Testee &testee, Fields const &... fields) {
    testee.pack(fields...);
    testee.exchange();
    testee.unpack(fields...);
}

template <class Testee, class... Fields>
void exchange(std::true_type, Testee &testee, Fields const &... fields) {
    std::vector<std::common_type_t<Fields...>> vec{fields...};
    testee.pack(vec);
    testee.exchange();
    testee.unpack(vec);
}

class halo_exchange_3D_test : public testing::TestWithParam<test_spec> {
    int mpi_dims[num_dims];
    int coords[num_dims] = {};

    value_type initial_state(int i, int j, int k, int field_no) const {
        auto val = [&](int i, int d) {
            auto size = GetParam().dims[d];
            i -= GetParam().halos[field_no][d][0];
            int c = coords[d];
            if (c == 0 && i < 0)
                c = mpi_dims[d];
            else if (c == mpi_dims[d] - 1 && i >= size)
                c = -1;
            return c * size + i;
        };
        return {val(i, 0), val(j, 1), val(k, 2), field_no};
    }

    template <int... Is>
    auto make_storages(layout_map<Is...>) const {
        auto make_storage = [&](int field_no) {
            auto size = [&](int d) {
                auto &&halos = GetParam().halos[field_no][d];
                return GetParam().dims[d] + halos[0] + halos[1];
            };
            auto in_halo = [&](int i, int d) {
                i -= GetParam().halos[field_no][d][0];
                return i < 0 || i >= GetParam().dims[d];
            };
            return storage::builder<storage_traits_t>
                    .template type<value_type>()
                    .template layout<Is...>()
                    .dimensions(size(0), size(1), size(2))
                    .initializer([&](int i, int j, int k) {
                        return in_halo(i, 0) || in_halo(j, 1) || in_halo(k, 2)
                            ? none()
                            : initial_state(i, j, k, field_no);
                    })
                    .build();
        };
        return array{make_storage(0), make_storage(1), make_storage(2)};
    }

    template <class Storages>
    void verify(Storages const &storages, std::array<bool, 3> periodicity) const {
        for (int f = 0; f != num_fields; ++f) {
            auto view = storages[f]->const_host_view();
            auto is_border = [&](int i, int d) {
                if (periodicity[d])
                    return false;
                i -= GetParam().halos[f][d][0];
                return (i < 0 && coords[d] == 0) || (i >= GetParam().dims[d] && coords[d] + 1 == mpi_dims[d]);
            };
            auto &&lengths = view.lengths();
            for (int i = 0; i != lengths[0]; ++i)
                for (int j = 0; j != lengths[1]; ++j)
                    for (int k = 0; k != lengths[2]; ++k)
                        EXPECT_EQ(view(i, j, k),
                            is_border(i, 0) || is_border(j, 1) || is_border(k, 2) ? none() : initial_state(i, j, k, f))
                            << "pid:" << gcl::pid() << " f:" << f << " i:" << i << " j:" << j << " k:" << k;
        }
    }

  public:
    MPI_Comm CartComm;

    halo_exchange_3D_test() {
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        for (int i = 0; i != num_dims; ++i)
            mpi_dims[i] = GetParam().mpi_dims[i];
        MPI_Dims_create(nprocs, num_dims, mpi_dims);
        int period[num_dims] = {1, 1, 1};
        MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_dims, period, false, &CartComm);
        MPI_Cart_get(CartComm, 3, mpi_dims, period, coords);
    }

    template <class Storages>
    auto make_halo_descriptors(Storages const &storages, int field_no) const {
        array<halo_descriptor, num_dims> res;
        auto total_lengths = make_total_lengths(*storages[field_no]);
        for (size_t d = 0; d != num_dims; ++d) {
            auto &&halos = GetParam().halos[field_no][d];
            res[d] = halo_descriptor(halos[0], halos[1], halos[0], GetParam().dims[d] + halos[0] - 1, total_lengths[d]);
        }
        return res;
    }

    template <class F>
    void run_exchanges(F f) {
        using layouts_t = meta::list<layout_map<0, 1, 2>,
            layout_map<0, 2, 1>,
            layout_map<1, 0, 2>,
            layout_map<1, 2, 0>,
            layout_map<2, 0, 1>,
            layout_map<2, 1, 0>>;
        using bools_t = meta::list<std::true_type, std::false_type>;
        for_each<layouts_t>([&](auto layout) {
            for_each<bools_t>([&](auto use_vector_interface) {
                for_each<bools_t>([&](auto p0) {
                    for_each<bools_t>([&](auto p1) {
                        for_each<bools_t>([&](auto p2) {
                            auto storages = make_storages(layout);
                            f(layout, use_vector_interface, storages, p0, p1, p2);
                            verify(storages, {p0, p1, p2});
                        });
                    });
                });
            });
        });
    }
};

struct halo_exchange_3D_all : halo_exchange_3D_test {};

TEST_P(halo_exchange_3D_all, test) {
    run_exchanges([&](auto layout, auto use_vector_interface, auto &&storages, auto... periodicity) {
        using testing::ContainerEq;
        auto &&halos = GetParam().halos;
        ASSERT_THAT(halos[1], ContainerEq(halos[0]));
        ASSERT_THAT(halos[2], ContainerEq(halos[0]));

        using testee_t = gcl::halo_exchange_dynamic_ut<decltype(layout), layout_map<0, 1, 2>, value_type, gcl_arch_t>;
        testee_t testee({periodicity...}, CartComm);
        auto halo_descriptors = make_halo_descriptors(storages, 0);
        for_each<meta::make_indices_c<num_fields>>(
            [&](auto f) { testee.template add_halo<decltype(f)::value>(halo_descriptors[f.value]); });
        testee.setup(3);
        auto field = [&](int f) { return storages[f]->get_target_ptr(); };
        exchange(use_vector_interface, testee, field(0), field(1), field(2));
    });
}

INSTANTIATE_TEST_SUITE_P(tests,
    halo_exchange_3D_all,
    testing::Values(test_spec{.dims = {123, 56, 76},
                        .halos = {{{2, 3}, {1, 2}, {2, 1}}, {{2, 3}, {1, 2}, {2, 1}}, {{2, 3}, {1, 2}, {2, 1}}},
                        .mpi_dims = {}},
        test_spec{.dims = {23, 12, 7},
            .halos = {{{2, 2}, {4, 4}, {3, 3}}, {{2, 2}, {4, 4}, {3, 3}}, {{2, 2}, {4, 4}, {3, 3}}},
            .mpi_dims = {2, 1}},
        test_spec{.dims = {23, 12, 7},
            .halos = {{{2, 2}, {4, 4}, {3, 3}}, {{2, 2}, {4, 4}, {3, 3}}, {{2, 2}, {4, 4}, {3, 3}}},
            .mpi_dims = {1, 2}},
        test_spec{.dims = {12, 12, 12},
            .halos = {{{2, 2}, {2, 2}, {2, 2}}, {{2, 2}, {2, 2}, {2, 2}}, {{2, 2}, {2, 2}, {2, 2}}},
            .mpi_dims = {2, 1}}));

struct halo_exchange_3D_generic : halo_exchange_3D_test {
    array<halo_descriptor, num_dims> make_enclosed_halo_descriptor() {
        array<halo_descriptor, num_dims> res;
        for (size_t d = 0; d != num_dims; ++d) {
            auto &&all_halos = GetParam().halos;
            int halos[2];
            for (size_t h = 0; h != num_halos; ++h)
                halos[h] = std::max({all_halos[0][d][h], all_halos[1][d][h], all_halos[2][d][h]});
            auto dims = GetParam().dims[d];
            res[d] = halo_descriptor(halos[0], halos[1], halos[0], dims + halos[0] - 1, dims + halos[0] + halos[1]);
        }
        return res;
    }
};

TEST_P(halo_exchange_3D_generic, test) {
    run_exchanges([&](auto layout, auto use_vector_interface, auto &&storages, auto... periodicity) {
        using layout_t = decltype(layout);
        using testee_t = gcl::halo_exchange_generic<layout_map<0, 1, 2>, gcl_arch_t>;
        testee_t testee({periodicity...}, CartComm);
        testee.setup(3,
            gcl::field_on_the_fly<int, layout_t, testee_t::traits>(nullptr, make_enclosed_halo_descriptor()),
            sizeof(value_type));
        auto field = [&](int f) {
            return gcl::field_on_the_fly<value_type, layout_t, testee_t::traits>(
                storages[f]->get_target_ptr(), make_halo_descriptors(storages, f));
        };
        exchange(use_vector_interface, testee, field(0), field(1), field(2));
    });
}

INSTANTIATE_TEST_SUITE_P(tests,
    halo_exchange_3D_generic,
    testing::Values(test_spec{.dims = {98, 54, 87},
                        .halos = {{{0, 1}, {2, 3}, {2, 1}}, {{0, 1}, {2, 3}, {2, 1}}, {{0, 1}, {2, 3}, {0, 1}}},
                        .mpi_dims = {}},
        test_spec{.dims = {89, 45, 104},
            .halos = {{{3, 3}, {1, 1}, {2, 2}}, {{3, 3}, {1, 1}, {2, 2}}, {{3, 3}, {1, 1}, {2, 2}}},
            .mpi_dims = {}}));
