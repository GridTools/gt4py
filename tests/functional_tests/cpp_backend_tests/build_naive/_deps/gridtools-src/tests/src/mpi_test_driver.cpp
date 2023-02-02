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
#include <mpi.h>

#include <gridtools/gcl/GCL.hpp>

#include "mpi_listener.hpp"

#ifdef GT_HAS_CUDA

#include <cstdlib>

#include <gridtools/common/cuda_runtime.hpp>
#include <gridtools/common/cuda_util.hpp>

namespace {
    int get_local_rank() {
        for (auto var : {"MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"})
            if (auto *str = std::getenv(var))
                return std::atoi(str);
        return 0;
    }

    int dev_device_count() {
        if (auto *str = std::getenv("NUM_GPU_DEVICES"))
            return std::atoi(str);
        int res;
        GT_CUDA_CHECK(cudaGetDeviceCount(&res));
        return res;
    }
} // namespace

#endif

int main(int argc, char **argv) {

#ifdef GT_HAS_CUDA
    GT_CUDA_CHECK(cudaSetDevice(get_local_rank() % dev_device_count()));
#endif

    gridtools::gcl::init(argc, argv);

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto &listeners = testing::UnitTest::GetInstance()->listeners();
    // first delete the original printer
    delete listeners.Release(listeners.default_result_printer());
    // now add our custom printer
    listeners.Append(new mpi_listener("results_global_communication"));

    // record the local return value for tests run on this mpi rank
    //      0 : success
    //      1 : failure
    auto result = RUN_ALL_TESTS();
    decltype(result) global_result{};

    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    gridtools::gcl::finalize();
    // perform global collective, to ensure that all ranks return the same exit code
    return global_result;
}
