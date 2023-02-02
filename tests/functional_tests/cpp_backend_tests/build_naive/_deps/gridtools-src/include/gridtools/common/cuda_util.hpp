/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "cuda_runtime.hpp"
#include "defs.hpp"

#define GT_CUDA_CHECK(expr)                                                                    \
    do {                                                                                       \
        cudaError_t err = expr;                                                                \
        if (err != cudaSuccess)                                                                \
            ::gridtools::cuda_util::_impl::on_error(err, #expr, __func__, __FILE__, __LINE__); \
    } while (false)

namespace gridtools {
    namespace cuda_util {
        namespace _impl {
            inline void on_error(cudaError_t err, const char snippet[], const char fun[], const char file[], int line) {
                std::ostringstream strm;
                strm << "cuda failure: \"" << cudaGetErrorString(err) << "\" [" << cudaGetErrorName(err) << "(" << err
                     << ")] in \"" << snippet << "\" function: " << fun << ", location: " << file << "(" << line << ")";
                throw std::runtime_error(strm.str());
            }

            struct cuda_free {
                template <class T>
                void operator()(T *ptr) const {
                    cudaFree(const_cast<std::remove_cv_t<T> *>(ptr));
                }
            };

        } // namespace _impl

        template <class T>
        using unique_cuda_ptr = std::unique_ptr<T, _impl::cuda_free>;

        template <class Arr, class T = std::remove_extent_t<Arr>>
        unique_cuda_ptr<Arr> cuda_malloc(size_t size) {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
            return unique_cuda_ptr<Arr>{ptr};
        }

        template <class T, std::enable_if_t<!std::is_array_v<T>, int> = 0>
        unique_cuda_ptr<T> cuda_malloc() {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T)));
            return unique_cuda_ptr<T>{ptr};
        }

        template <class T, std::enable_if_t<std::is_trivially_copy_constructible_v<T>, int> = 0>
        unique_cuda_ptr<T> make_clone(T const &src) {
            unique_cuda_ptr<T> res = cuda_malloc<T>();
            GT_CUDA_CHECK(cudaMemcpy(res.get(), &src, sizeof(T), cudaMemcpyHostToDevice));
            return res;
        }

        template <class T, std::enable_if_t<std::is_trivially_copy_constructible_v<T>, int> = 0>
        T from_clone(unique_cuda_ptr<T> const &clone) {
            T res;
            GT_CUDA_CHECK(cudaMemcpy(&res, clone.get(), sizeof(T), cudaMemcpyDeviceToHost));
            return res;
        }

#ifdef GT_CUDACC
        template <class Kernel, class... Args>
        void launch(dim3 const &blocks,
            dim3 const &threads,
            size_t shared_memory_size,
            cudaStream_t stream,
            Kernel kernel,
            Args... args) {
#ifndef __HIPCC__
            GT_CUDA_CHECK(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
#endif
            kernel<<<blocks, threads, shared_memory_size, stream>>>(std::move(args)...);
            GT_CUDA_CHECK(cudaGetLastError());
#ifndef NDEBUG
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#endif
        }
#endif
    } // namespace cuda_util
} // namespace gridtools
