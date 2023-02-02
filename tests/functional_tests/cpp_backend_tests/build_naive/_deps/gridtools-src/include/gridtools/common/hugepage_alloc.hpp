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

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <tuple>

#ifdef __linux__
#include <cstdio>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace gridtools {
    namespace hugepage_alloc_impl_ {
        inline std::size_t ilog2(std::size_t i) {
            std::size_t log = 0;
            while (i >>= 1)
                ++log;
            return log;
        }

        enum class hugepage_mode { disabled, transparent, explicit_allocation };

#ifdef __linux__
        inline std::size_t get_sysinfo(const char *info, std::size_t default_value) {
            int fd = open(info, O_RDONLY);
            if (fd != -1) {
                char buffer[16];
                auto size = read(fd, buffer, sizeof(buffer));
                if (size > 0)
                    default_value = std::atoll(buffer);
                close(fd);
            }
            return default_value;
        }

        inline std::size_t get_meminfo(const char *pattern, std::size_t default_value) {
            auto *fp = std::fopen("/proc/meminfo", "r");
            if (fp) {
                char *line = nullptr;
                size_t line_length;
                while (getline(&line, &line_length, fp) != -1) {
                    if (sscanf(line, pattern, &default_value) == 1)
                        break;
                }
                free(line);
                std::fclose(fp);
            }
            return default_value;
        }

        inline std::size_t cache_line_size() {
            static const std::size_t value =
                get_sysinfo("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", 64);
            return value;
        }

        inline std::size_t cache_sets() {
            static const std::size_t value =
                get_sysinfo("/sys/devices/system/cpu/cpu0/cache/index0/number_of_sets", 64);
            return value;
        }

        inline std::size_t hugepage_size() {
            static const std::size_t value = get_meminfo("Hugepagesize: %lu kB", 2 * 1024) * 1024;
            return value;
        }

        inline std::size_t page_size() {
            static const std::size_t value = sysconf(_SC_PAGESIZE);
            return value;
        }

        inline std::pair<void *, std::size_t> allocate(std::size_t size, hugepage_mode mode) {
            void *ptr;
            switch (mode) {
            case hugepage_mode::disabled:
                // here we just align to small/normal page size
                size = ((size + page_size() - 1) / page_size()) * page_size();
                if (posix_memalign(&ptr, page_size(), size))
                    throw std::bad_alloc();
                // explicitly forbid usage of huge pages
                madvise(ptr, size, MADV_NOHUGEPAGE);
                break;
            case hugepage_mode::transparent:
                // here we try to get transparent huge pages
                size = ((size + hugepage_size() - 1) / hugepage_size()) * hugepage_size();
                ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (ptr == MAP_FAILED)
                    throw std::bad_alloc();
                madvise(ptr, size, MADV_HUGEPAGE);
                break;
            case hugepage_mode::explicit_allocation:
                // here we force huge page allocation (fails with a bus error if none are available)
                size = ((size + hugepage_size() - 1) / hugepage_size()) * hugepage_size();
                ptr = mmap(nullptr,
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_NORESERVE,
                    -1,
                    0);
                if (ptr == MAP_FAILED)
                    throw std::bad_alloc();
                break;
            }
            return {ptr, size};
        }

        inline void deallocate(void *ptr, std::size_t size, hugepage_mode mode) {
            switch (mode) {
            case hugepage_mode::disabled:
                free(ptr);
                break;
            default:
                if (munmap(ptr, size))
                    throw std::bad_alloc();
                break;
            }
        }
#else
        inline std::size_t cache_line_size() {
            return 64; // default value for x86-64 archs
        }

        inline std::size_t cache_sets() {
            return 64; // default value for (most?) x86-64 archs
        }

        inline std::size_t hugepage_size() {
            return 2 * 1024 * 1024; // 2MB is the default on most systems
        }

        inline std::size_t page_size() {
            return 4 * 1024; // 4kB is the default on most systems
        }

        inline std::tuple<void *, std::size_t> allocate(std::size_t size, hugepage_mode) {
            // here we hope that aligning to hugepage_size will return transparent huge pages
            void *ptr;
            size = ((size + hugepage_size() - 1) / hugepage_size()) * hugepage_size();
            if (posix_memalign(&ptr, hugepage_size(), size))
                throw std::bad_alloc();
            return {ptr, size};
        }

        inline void deallocate(void *ptr, std::size_t, hugepage_mode) { free(ptr); }
#endif

        inline std::size_t allocation_offset() {
            static std::atomic<std::size_t> s_offset(0);
            return ((s_offset++ % cache_sets()) << ilog2(cache_line_size())) + cache_line_size();
        }

        inline hugepage_mode hugepage_mode_from_env() {
            const char *env_value = std::getenv("GT_HUGEPAGE_MODE");
            if (!env_value || std::strcmp(env_value, "transparent") == 0)
                return hugepage_mode::transparent;
            if (std::strcmp(env_value, "disable") == 0)
                return hugepage_mode::disabled;
            if (std::strcmp(env_value, "explicit") == 0)
                return hugepage_mode::explicit_allocation;
            std::fprintf(stderr, "warning: env variable GT_HUGEPAGE_MODE set to invalid value '%s'\n", env_value);
            return hugepage_mode::transparent;
        }

        struct ptr_metadata {
            std::size_t offset, full_size;
            hugepage_mode mode;
        };

    } // namespace hugepage_alloc_impl_

    /**
     * @brief Allocates huge page memory (if GT_NO_HUGETLB is not defined) and shifts allocations by some bytes to
     * reduce cache set conflicts.
     */
    inline void *hugepage_alloc(std::size_t size) {
        // get allocation offset to reduce L1 cache conflicts
        std::size_t offset = hugepage_alloc_impl_::allocation_offset();
        assert(offset >= sizeof(hugepage_alloc_impl_::ptr_metadata));

        // get allocation mode from environment
        auto mode = hugepage_alloc_impl_::hugepage_mode_from_env();

        // allocate memory with additional space for offsetting
        void *ptr;
        std::tie(ptr, size) = hugepage_alloc_impl_::allocate(size + offset, mode);

        // offset pointer and write pointer metadata required for deallocation
        ptr = static_cast<char *>(ptr) + offset;
        static_cast<hugepage_alloc_impl_::ptr_metadata *>(ptr)[-1] = {offset, size, mode};
        return ptr;
    }

    /**
     * @brief Frees memory allocated by hugepage_alloc.
     */
    inline void hugepage_free(void *ptr) {
        if (!ptr)
            return;
        // read pointer metadata and compute originally allocated ptr value
        auto &metadata = static_cast<hugepage_alloc_impl_::ptr_metadata *>(ptr)[-1];
        // free originally allocated pointer
        hugepage_alloc_impl_::deallocate(static_cast<char *>(ptr) - metadata.offset, metadata.full_size, metadata.mode);
    }

} // namespace gridtools
