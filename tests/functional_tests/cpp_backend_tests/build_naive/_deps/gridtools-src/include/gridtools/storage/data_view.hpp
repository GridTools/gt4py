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

#include <utility>

#include "../common/array.hpp"

namespace gridtools {
    namespace storage {
        template <class T, class Info>
        struct host_view {
            T *m_ptr;
            Info const *m_info;

            auto *data() const { return m_ptr; }
            auto const &info() const { return *m_info; }

            decltype(auto) length() const { return m_info->length(); }
            decltype(auto) lengths() const { return m_info->lengths(); }
            decltype(auto) strides() const { return m_info->strides(); }
            decltype(auto) native_lengths() const { return m_info->native_lengths(); }
            decltype(auto) native_strides() const { return m_info->native_strides(); }

            template <class... Args>
            auto operator()(Args &&... args) const -> decltype(m_ptr[m_info->index(std::forward<Args>(args)...)]) {
                return m_ptr[m_info->index(std::forward<Args>(args)...)];
            }

            decltype(auto) operator()(array<int, Info::ndims> const &arg) const {
                return m_ptr[m_info->index_from_tuple(arg)];
            }
        };

        template <class T, class Info>
        host_view<T, Info> make_host_view(T *ptr, Info const &info) {
            return {ptr, &info};
        }
    } // namespace storage
} // namespace gridtools
