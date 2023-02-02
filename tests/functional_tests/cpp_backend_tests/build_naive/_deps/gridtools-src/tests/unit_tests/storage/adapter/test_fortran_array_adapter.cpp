/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <cpp_bindgen/fortran_array_view.hpp>
#include <gridtools/storage/adapter/fortran_array_adapter.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/cpu_kfirst.hpp>

const auto builder = gridtools::storage::builder<gridtools::storage::cpu_kfirst>.type<double>();

TEST(FortranArrayAdapter, TransformAdapterIntoDataStore) {
    constexpr size_t x_size = 6;
    constexpr size_t y_size = 5;
    constexpr size_t z_size = 4;
    double fortran_array[z_size][y_size][x_size];

    bindgen_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = x_size;
    descriptor.dims[1] = y_size;
    descriptor.dims[2] = z_size;
    descriptor.type = bindgen_fk_Double;
    descriptor.data = fortran_array;
    descriptor.is_acc_present = false;

    auto data_store = builder.dimensions(x_size, y_size, z_size)();

    int i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                fortran_array[z][y][x] = i;

    // transform adapter into data_store
    gridtools::fortran_array_adapter<decltype(data_store)>{descriptor}.transform_to(data_store);

    i = 0;
    auto view = data_store->host_view();
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                EXPECT_EQ(view(x, y, z), i);
}

TEST(FortranArrayAdapter, TransformDataStoreIntoAdapter) {
    constexpr size_t x_size = 6;
    constexpr size_t y_size = 5;
    constexpr size_t z_size = 4;
    double fortran_array[z_size][y_size][x_size];

    bindgen_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = x_size;
    descriptor.dims[1] = y_size;
    descriptor.dims[2] = z_size;
    descriptor.type = bindgen_fk_Double;
    descriptor.data = fortran_array;
    descriptor.is_acc_present = false;

    auto data_store = builder.dimensions(x_size, y_size, z_size)();

    auto view = data_store->host_view();

    int i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                view(x, y, z) = i;

    // transform data_store into adapter
    gridtools::fortran_array_adapter<decltype(data_store)>{descriptor}.transform_from(data_store);

    i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                EXPECT_EQ(fortran_array[z][y][x], i);
}
