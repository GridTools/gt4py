/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_SID_ALLOCATOR_HPP_
#define GT_SID_ALLOCATOR_HPP_

#include <map>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../meta.hpp"
#include "simple_ptr_holder.hpp"

/**
 *
 *  Allocator concept
 *  -----------------
 *
 *  For any allocator type function `allocate` should be avaliable by ADL if it is called like this:
 *
 *  Allocator allocator;
 *
 *  auto ptr_holder = allocate(allocator, meta:::lazy:id<T>, size);
 *
 *  The return value of `allocate` should be a ptr holder in the sid concept sense.
 *  Allocator should keep the ownership of the allocated resources.
 *
 *
 *  API
 *  ---
 *
 *  The library provides two types that model the concept:
 *    - `allocator`,
 *    - `cached_allocator`.
 *
 *  Both are templated with the functor that takes the size in bytes and returns `std::unique_ptr`
 *
 *  Semantics:
 *    - `allocator` keeps the resources that are allocated and releases them in dtor.
 *    - `cached_allocator` keeps resources during its lifetime. On dtor it stashes the resources in the internal static
 *      storage. The newly created instances of `cached_allocator` will attempt to reuse the stashed resources.
 *
 *  To make the simplest possible allocator one can do:
 *    `auto alloc = allocator(&std::make_unique<char[]>);`
 *
 */

namespace gridtools {
    namespace sid {
        namespace allocator_impl_ {

            template <class Impl, class Ptr = decltype(std::declval<Impl const>()(size_t{}))>
            struct cached_proxy_f;

            template <class Impl, class T, class Deleter>
            struct cached_proxy_f<Impl, std::unique_ptr<T, Deleter>> {
                using ptr_t = std::unique_ptr<T, Deleter>;
                using stack_t = std::stack<ptr_t>;

                struct deleter_f {
                    using pointer = typename ptr_t::pointer;
                    Deleter m_deleter;
                    stack_t &m_stack;

                    void operator()(pointer ptr) const { m_stack.emplace(ptr, m_deleter); }
                };
                using cached_ptr_t = std::unique_ptr<T, deleter_f>;

                Impl m_impl;

                cached_ptr_t operator()(size_t size) const {
                    static thread_local std::map<size_t, stack_t> stack_map;
                    auto &stack = stack_map[size];
                    ptr_t ptr;
                    if (stack.empty()) {
                        ptr = m_impl(size);
                    } else {
                        ptr = std::move(stack.top());
                        stack.pop();
                    }
                    return {ptr.release(), {ptr.get_deleter(), stack}};
                }
            };
        } // namespace allocator_impl_
    }     // namespace sid
} // namespace gridtools

#define GT_FILENAME <gridtools/sid/allocator.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    namespace sid {
        GT_TARGET_NAMESPACE {
            template <class Impl, class Ptr = decltype(std::declval<Impl const>()(size_t{}))>
            class allocator;

            template <class Impl, class T, class Deleter>
            class allocator<Impl, std::unique_ptr<T, Deleter>> {
                Impl m_impl;
                std::vector<std::unique_ptr<T, Deleter>> m_buffers;

              public:
                allocator() = default;
                allocator(Impl impl) : m_impl(std::move(impl)) {}

                template <class LazyT>
                friend auto allocate(allocator &self, LazyT, size_t size) {
                    using type = typename LazyT::type;
                    auto ptr = self.m_impl(sizeof(type) * size);
                    self.m_buffers.push_back(self.m_impl(sizeof(type) * size));
                    return simple_ptr_holder(reinterpret_cast<type *>(self.m_buffers.back().get()));
                }
            };

            template <class Impl>
            allocator(Impl) -> allocator<Impl>;

            template <class Impl>
            struct cached_allocator : allocator<allocator_impl_::cached_proxy_f<Impl>> {
                cached_allocator() = default;
                cached_allocator(Impl impl) : cached_allocator::allocator({std::move(impl)}) {}
            };
        }
    } // namespace sid
} // namespace gridtools

#endif
