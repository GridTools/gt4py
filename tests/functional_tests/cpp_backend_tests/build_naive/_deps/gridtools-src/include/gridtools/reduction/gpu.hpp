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

#include <cassert>
#include <cstdlib>
#include <numeric>
#include <type_traits>

#include "../common/ct_dispatch.hpp"
#include "../common/cuda_runtime.hpp"
#include "../common/cuda_util.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../meta.hpp"
#include "functions.hpp"

namespace gridtools {
    namespace reduction {
        namespace gpu_backend {
            struct tiny_kernel {
                template <class F, class T>
                GT_FUNCTION_DEVICE void operator()(F f, T const *__restrict__ in, T *__restrict__ out) const {
                    in += blockIdx.x * 2;
                    *(out + blockIdx.x) = f(*in, *(in + 1));
                }
            };

            template <size_t BlockSize,
                size_t Step,
                class F,
                class T,
                std::enable_if_t<(BlockSize >= 2 * Step), int> = 0>
            GT_FUNCTION_DEVICE void shmem_reduce(F f, T *val) {
                __syncthreads();
                if (threadIdx.x < Step)
                    *val = f(*val, *(val + Step));
            }
            template <size_t BlockSize,
                size_t Step,
                class F,
                class T,
                std::enable_if_t<(BlockSize < 2 * Step), int> = 0>
            GT_FUNCTION_DEVICE void shmem_reduce(F, T *) {}

            template <size_t BlockSize>
            struct shmem_kernel {
                static_assert(BlockSize > 1, GT_INTERNAL_ERROR);
                static_assert(BlockSize <= 1024, GT_INTERNAL_ERROR);
                template <class F, class T>
                GT_FUNCTION_DEVICE void operator()(F f, T const *__restrict__ in, T *__restrict__ out) const {
                    in += blockIdx.x * BlockSize * 2 + threadIdx.x;
                    __shared__ T buff[BlockSize];
                    T *val = buff + threadIdx.x;
                    *val = f(*in, *(in + BlockSize));
                    shmem_reduce<BlockSize, 512>(f, val);
                    shmem_reduce<BlockSize, 256>(f, val);
                    shmem_reduce<BlockSize, 128>(f, val);
                    shmem_reduce<BlockSize, 64>(f, val);
                    shmem_reduce<BlockSize, 32>(f, val);
                    shmem_reduce<BlockSize, 16>(f, val);
                    shmem_reduce<BlockSize, 8>(f, val);
                    shmem_reduce<BlockSize, 4>(f, val);
                    shmem_reduce<BlockSize, 2>(f, val);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        *(out + blockIdx.x) = f(*val, *(val + 1));
                }
            };

#ifdef __HIP__
            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 64>, T val) {
                val = f(val, __shfl_down(val, 32));
                val = f(val, __shfl_down(val, 16));
                val = f(val, __shfl_down(val, 8));
                val = f(val, __shfl_down(val, 4));
                val = f(val, __shfl_down(val, 2));
                return f(val, __shfl_down(val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 32>, T val) {
                val = f(val, __shfl_down(val, 16));
                val = f(val, __shfl_down(val, 8));
                val = f(val, __shfl_down(val, 4));
                val = f(val, __shfl_down(val, 2));
                return f(val, __shfl_down(val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 16>, T val) {
                val = f(val, __shfl_down(val, 8));
                val = f(val, __shfl_down(val, 4));
                val = f(val, __shfl_down(val, 2));
                return f(val, __shfl_down(val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 8>, T val) {
                val = f(val, __shfl_down(val, 4));
                val = f(val, __shfl_down(val, 2));
                return f(val, __shfl_down(val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 4>, T val) {
                val = f(val, __shfl_down(val, 2));
                return f(val, __shfl_down(val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 2>, T val) {
                return f(val, __shfl_down(val, 1));
            }
#else
            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 64>, T val) {
                val = f(val, __shfl_down_sync((unsigned)-1, val, 32));
                val = f(val, __shfl_down_sync(0xFFFFFFF, val, 16));
                val = f(val, __shfl_down_sync(0xFFFF, val, 8));
                val = f(val, __shfl_down_sync(0xFF, val, 4));
                val = f(val, __shfl_down_sync(0xF, val, 2));
                return f(val, __shfl_down_sync(0x3, val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 32>, T val) {
                val = f(val, __shfl_down_sync(0xFFFFFFF, val, 16));
                val = f(val, __shfl_down_sync(0xFFFF, val, 8));
                val = f(val, __shfl_down_sync(0xFF, val, 4));
                val = f(val, __shfl_down_sync(0xF, val, 2));
                return f(val, __shfl_down_sync(0x3, val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 16>, T val) {
                val = f(val, __shfl_down_sync(0xFFFF, val, 8));
                val = f(val, __shfl_down_sync(0xFF, val, 4));
                val = f(val, __shfl_down_sync(0xF, val, 2));
                return f(val, __shfl_down_sync(0x3, val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 8>, T val) {
                val = f(val, __shfl_down_sync(0xFF, val, 4));
                val = f(val, __shfl_down_sync(0xF, val, 2));
                return f(val, __shfl_down_sync(0x3, val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 4>, T val) {
                val = f(val, __shfl_down_sync(0xF, val, 2));
                return f(val, __shfl_down_sync(0x3, val, 1));
            }

            template <class F, class T>
            GT_FUNCTION_DEVICE T warp_reduce(F f, std::integral_constant<size_t, 2>, T val) {
                return f(val, __shfl_down_sync(0x3, val, 1));
            }

#if __CUDA_ARCH__ >= 800
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(plus, std::integral_constant<size_t, Width>, int res) {
                return __reduce_add_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(min, std::integral_constant<size_t, Width>, int res) {
                return __reduce_min_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(max, std::integral_constant<size_t, Width>, int res) {
                return __reduce_max_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(plus, std::integral_constant<size_t, Width>, unsigned res) {
                return __reduce_add_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(min, std::integral_constant<size_t, Width>, unsigned res) {
                return __reduce_min_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(max, std::integral_constant<size_t, Width>, unsigned res) {
                return __reduce_max_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(bitwise_and, std::integral_constant<size_t, Width>, unsigned res) {
                return __reduce_and_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(bitwise_or, std::integral_constant<size_t, Width>, unsigned res) {
                return __reduce_or_sync((1 << Width) - 1, res);
            }
            template <size_t Width>
            GT_FUNCTION_DEVICE auto warp_reduce(bitwise_xor, std::integral_constant<size_t, Width>, unsigned res) {
                return __reduce_xor_sync((1 << Width) - 1, res);
            }
#endif
#endif

            template <size_t WarpSize, size_t BlockSize, size_t ShmemSize = BlockSize / WarpSize>
            struct big_warp_kernel {
                static_assert(BlockSize > WarpSize, GT_INTERNAL_ERROR);
                static_assert(ShmemSize <= WarpSize, GT_INTERNAL_ERROR);
                template <class F, class T>
                GT_FUNCTION_DEVICE void operator()(F f, T const *__restrict__ in, T *__restrict__ out) const {
                    __shared__ T buff[ShmemSize];
                    in += blockIdx.x * BlockSize * 2 + threadIdx.x;
                    T res = warp_reduce(f, std::integral_constant<size_t, WarpSize>(), f(*in, *(in + BlockSize)));
                    if (threadIdx.x % WarpSize == 0)
                        buff[threadIdx.x / WarpSize] = res;
                    __syncthreads();
                    if (threadIdx.x >= ShmemSize)
                        return;
                    res = warp_reduce(f, std::integral_constant<size_t, ShmemSize>(), buff[threadIdx.x]);
                    if (threadIdx.x == 0)
                        *(out + blockIdx.x) = res;
                }
            };

            template <size_t WarpSize, size_t BlockSize>
            struct small_warp_kernel {
                static_assert(BlockSize > 1, GT_INTERNAL_ERROR);
                static_assert(BlockSize <= WarpSize, GT_INTERNAL_ERROR);
                template <class F, class T>
                GT_FUNCTION_DEVICE void operator()(F f, T const *__restrict__ in, T *__restrict__ out) const {
                    in += blockIdx.x * BlockSize * 2 + threadIdx.x;
                    T res = warp_reduce(f, std::integral_constant<size_t, BlockSize>(), f(*in, *(in + BlockSize)));
                    if (threadIdx.x == 0)
                        *(out + blockIdx.x) = res;
                }
            };

            template <class T>
            using is_shufflable = meta::st_contains<
                meta::list<int, unsigned, long, unsigned long, long long, unsigned long long, float, double>,
                T>;

            template <size_t WarpSize, size_t BlockSize, class T>
            using choose_kernel = typename meta::if_c<(BlockSize == 1),
                // no need to use neither shared memory nor intrinsics
                meta::lazy::id<tiny_kernel>,
                meta::if_<is_shufflable<T>,
                    meta::if_c<(BlockSize > WarpSize),
#if defined(__HIP__) && HIP_VERSION <= 400
                        // compiler bug appearing in big_warp_kernel, thus using shared mem
                        meta::lazy::id<shmem_kernel<BlockSize>>,
#else
                        // use shared memory and two rounds of warp level intrinsics
                        meta::lazy::id<big_warp_kernel<WarpSize, BlockSize>>,
#endif
                        // use a single round of warp level intrinsics, no shared memory needed
                        meta::lazy::id<small_warp_kernel<WarpSize, BlockSize>>>,
                    // use shared memory only
                    meta::lazy::id<shmem_kernel<BlockSize>>>>::type;

            template <class Kernel, class F, class T>
            __global__ void launch(Kernel kernel, F f, T const *__restrict__ in, T *__restrict__ out) {
                kernel(f, in, out);
            }

            inline constexpr bool is_pow2(size_t x) { return (x & x - 1) == 0; }

            inline constexpr size_t next_pow2(size_t x) {
                --x;
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                x |= x >> 8;
                x |= x >> 16;
                x |= x >> 32;
                return ++x;
            }

            inline constexpr size_t int_log2(size_t val) {
                assert(val);
                size_t ret = 0;
                for (; val > 1; val /= 2)
                    ret++;
                return ret;
            }

            // TODO(anstaf): benchmark that
            using cpu_final_threshold_t = integral_constant<size_t, 8>;

            static_assert(cpu_final_threshold_t::value > 1, GT_INTERNAL_ERROR);
            static_assert(is_pow2(cpu_final_threshold_t::value), GT_INTERNAL_ERROR);

            inline auto fecth_device_properties() {
                int device;
                GT_CUDA_CHECK(cudaGetDevice(&device));
                cudaDeviceProp res;
                GT_CUDA_CHECK(cudaGetDeviceProperties(&res, device));
                return res;
            }

            inline auto const &get_device_properties() {
                static auto res = fecth_device_properties();
                return res;
            }

            inline size_t max_threads() { return get_device_properties().maxThreadsPerBlock; }

            inline size_t get_threads(size_t n) {
                assert(is_pow2(max_threads()));
                assert(n % 2 == 0);
                return std::gcd(n / 2, max_threads());
            }

            template <class T>
            double kahan_sum(T const *data, size_t n) {
                assert(n);
                double sum = data[0];
                double c = 0;
                for (int i = 1; i < n; i++) {
                    double y = data[i] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                return sum;
            }

            template <class F, class T>
            T reduce_cpu(F f, T const *buff, size_t n) {
                assert(n);
                T res = buff[0];
                for (size_t i = 1; i != n; i++)
                    res = f(res, buff[i]);
                return res;
            }

            inline auto reduce_cpu(plus, float const *buff, size_t n) { return kahan_sum(buff, n); }

            inline auto reduce_cpu(plus, double const *buff, size_t n) { return kahan_sum(buff, n); }

            template <class F, class T>
            void reduce_step(F f, size_t threads, size_t blocks, size_t warp_size, T const *in, T *out) {
                constexpr size_t max_threads = 1024;
                assert(threads >= 1);
                assert(threads <= max_threads);
                assert(warp_size == 32 || warp_size == 64);
                ct_dispatch<2>(
                    [=](auto w) {
                        constexpr size_t warp_size = decltype(w)::value ? 64 : 32;
                        ct_dispatch<int_log2(max_threads) + 1>(
                            [=](auto n) {
                                constexpr size_t block_size = 1 << decltype(n)::value;
                                using kernel_t = choose_kernel<warp_size, block_size, T>;
                                launch<<<blocks, threads>>>(kernel_t(), f, in, out);
                                GT_CUDA_CHECK(cudaGetLastError());
                            },
                            int_log2(threads));
                    },
                    warp_size / 64);
            }

            template <class T>
            __global__ void fill(T *dst, T val) {
                dst[blockIdx.x * blockDim.x + threadIdx.x] = val;
            }

            struct gpu {};

            template <class F, class T>
            auto reduction_reduce(gpu, T, F f, T *ptr, size_t size) {
                T *in = ptr;
                size_t n = size;
                auto warp_size = get_device_properties().warpSize;
                while (n > cpu_final_threshold_t::value) {
                    size_t threads = get_threads(n);
                    size_t blocks = n / (2 * threads);
                    T *out = in + n;
                    reduce_step(f, threads, blocks, warp_size, in, out);
                    n = blocks;
                    in = out;
                }
                T buff[cpu_final_threshold_t::value];
                GT_CUDA_CHECK(cudaMemcpy(buff, in, n * sizeof(T), cudaMemcpyDeviceToHost));
                return reduce_cpu(f, buff, n);
            }

            inline size_t reduction_round_size(gpu, size_t size) {
                auto chunk = next_pow2(size) / cpu_final_threshold_t::value;
                return chunk ? (size + chunk - 1) / chunk * chunk : size;
            }

            inline size_t reduction_allocation_size(gpu, size_t n) {
                size_t res = 0;
                do {
                    res += n;
                    n /= 2 * get_threads(n);
                } while (n > cpu_final_threshold_t::value);
                res += n;
                return res;
            }

            template <class T>
            void reduction_fill(gpu, T const &val, T *dst, size_t data_size, size_t rounded_size, bool has_holes) {
                if (!has_holes && data_size == rounded_size)
                    return;
                auto threads = std::gcd(rounded_size, max_threads());
                fill<<<rounded_size / threads, threads>>>(dst, val);
                GT_CUDA_CHECK(cudaGetLastError());
            }
        } // namespace gpu_backend
        using gpu_backend::gpu;
    } // namespace reduction
} // namespace gridtools
