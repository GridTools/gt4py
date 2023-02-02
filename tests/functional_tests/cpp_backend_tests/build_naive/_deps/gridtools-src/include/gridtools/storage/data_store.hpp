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
#include <string>
#include <type_traits>
#include <utility>

#include "../common/array.hpp"
#include "../common/array_addons.hpp"
#include "../common/defs.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "data_view.hpp"
#include "info.hpp"
#include "traits.hpp"

namespace gridtools {
    namespace storage {

        struct uninitialized {};

        namespace data_store_impl_ {
            template <class Traits, class T, class Info, class Kind>
            class base {
                static constexpr size_t byte_alignment = traits::byte_alignment<Traits>;
                static_assert(byte_alignment > 0, GT_INTERNAL_ERROR);

                using alignment_t = integral_constant<int, traits::elem_alignment<Traits, T>>;
                using strides_t = decltype(std::declval<Info const &>().native_strides());

                using mutable_data_t = std::remove_const_t<T>;

                std::string m_name;
                Info m_info;
                traits::target_ptr_type<Traits, mutable_data_t> m_target_ptr_holder;
                mutable_data_t *m_target_ptr;

              public:
                using layout_t = traits::layout_type<Traits, Info::ndims>;
                using data_t = T;
                using kind_t = Kind;
                static constexpr size_t ndims = Info::ndims;

                auto const &name() const { return m_name; }
                auto const &info() const { return m_info; }
                decltype(auto) native_lengths() const { return m_info.native_lengths(); }
                decltype(auto) native_strides() const { return m_info.native_strides(); }
                decltype(auto) lengths() const { return m_info.lengths(); }
                decltype(auto) strides() const { return m_info.strides(); }
                decltype(auto) length() const { return m_info.length(); }

              protected:
                template <class Halos>
                base(std::string name, Info info, Halos const &halos)
                    : m_name(std::move(name)), m_info(std::move(info)),
                      m_target_ptr_holder(traits::allocate<Traits, mutable_data_t>(m_info.length() + alignment_t())) {
                    auto offset_to_align = m_info.index_from_tuple(halos);
                    auto byte_offset = offset_to_align * sizeof(T);
                    auto address_to_align = reinterpret_cast<std::uintptr_t>(m_target_ptr_holder.get()) + byte_offset;
                    m_target_ptr = reinterpret_cast<mutable_data_t *>(
                        (address_to_align + byte_alignment - 1) / byte_alignment * byte_alignment - byte_offset);
                }

                auto raw_target_ptr() const { return m_target_ptr; }
            };

            template <class Traits,
                class T,
                class Info,
                class Kind,
                bool = std::is_const_v<T>,
                bool = traits::is_host_referenceable<Traits>>
            class data_store;

            template <class Traits, class T, class Info, class Kind>
            class data_store<Traits, T, Info, Kind, false, false> : public base<Traits, T, Info, Kind> {
                enum state { synced, invalid_host, invalid_target };
                state m_state;
                std::unique_ptr<T[]> m_host_ptr;

                void update_target() {
                    if (m_state != invalid_target)
                        return;
                    traits::update_target<Traits>(this->raw_target_ptr(), m_host_ptr.get(), this->info().length());
                    m_state = synced;
                }

                void update_host() {
                    if (m_state != invalid_host)
                        return;
                    traits::update_host<Traits>(m_host_ptr.get(), this->raw_target_ptr(), this->info().length());
                    m_state = synced;
                }

              public:
                template <class Halos>
                data_store(std::string name, Info info, Halos const &halos, uninitialized const &)
                    : data_store::base(std::move(name), std::move(info), halos), m_state(synced),
                      m_host_ptr(std::make_unique<T[]>(this->info().length())) {}

                template <class Initializer, class Halos>
                data_store(std::string name, Info info, Halos const &halos, Initializer const &initializer)
                    : data_store::base(std::move(name), std::move(info), halos), m_state(invalid_target),
                      m_host_ptr(std::make_unique<T[]>(this->info().length())) {
                    initializer(m_host_ptr.get(), typename data_store::layout_t(), this->info());
                }

                T *get_target_ptr() {
                    update_target();
                    m_state = invalid_host;
                    return this->raw_target_ptr();
                }

                T const *get_const_target_ptr() {
                    update_target();
                    return this->raw_target_ptr();
                }

                T *get_host_ptr() {
                    update_host();
                    m_state = invalid_target;
                    return m_host_ptr.get();
                }

                T const *get_const_host_ptr() {
                    update_host();
                    return m_host_ptr.get();
                }

                auto host_view() { return make_host_view(get_host_ptr(), this->info()); }
                auto const_host_view() { return make_host_view(get_const_host_ptr(), this->info()); }

                auto target_view() { return traits::make_target_view<Traits>(get_target_ptr(), this->info()); }
                auto const_target_view() {
                    return traits::make_target_view<Traits>(get_const_target_ptr(), this->info());
                }
            };

            template <class Traits, class T, class Info, class Kind>
            class data_store<Traits, T, Info, Kind, false, true> : public base<Traits, T, Info, Kind> {
              public:
                template <class Halos>
                data_store(std::string name, Info info, Halos const &halos, uninitialized const &)
                    : data_store::base(std::move(name), std::move(info), halos) {}

                template <class Initializer, class Halos>
                data_store(std::string name, Info info, Halos const &halos, Initializer const &initializer)
                    : data_store::base(std::move(name), std::move(info), halos) {
                    initializer(this->raw_target_ptr(), typename data_store::layout_t(), this->info());
                }

                T *get_target_ptr() const { return this->raw_target_ptr(); }
                T const *get_const_target_ptr() const { return this->raw_target_ptr(); }

                auto target_view() const { return traits::make_target_view<Traits>(get_target_ptr(), this->info()); }
                auto const_target_view() const {
                    return traits::make_target_view<Traits>(get_const_target_ptr(), this->info());
                }

                T *get_host_ptr() { return get_target_ptr(); }
                T const *get_const_host_ptr() { return get_const_target_ptr(); }
                auto host_view() const { return target_view(); }
                auto const_host_view() const { return const_target_view(); }
            };

            template <class Traits, class T, class Info, class Kind, bool IsHostRefrenceable>
            class data_store<Traits, T const, Info, Kind, true, IsHostRefrenceable>
                : public base<Traits, T const, Info, Kind> {

                template <class>
                struct is_host_refrenceable : std::bool_constant<IsHostRefrenceable> {};

                template <class Initializer, std::enable_if_t<!is_host_refrenceable<Initializer>::value, int> = 0>
                void init(Initializer const &initializer) {
                    auto host_ptr = std::make_unique<T[]>(this->info().length());
                    initializer(host_ptr.get(), typename data_store::layout_t(), this->info());
                    traits::update_target<Traits>(this->raw_target_ptr(), host_ptr.get(), this->info().length());
                }

                template <class Initializer, std::enable_if_t<is_host_refrenceable<Initializer>::value, int> = 0>
                void init(Initializer const &initializer) {
                    initializer(this->raw_target_ptr(), typename data_store::layout_t(), this->info());
                }

              public:
                template <class Halos>
                data_store(std::string, Info, Halos const &, uninitialized const &) = delete;

                template <class Initializer, class Halos>
                data_store(std::string name, Info info, Halos const &halos, Initializer const &initializer)
                    : base<Traits, T const, Info, Kind>(std::move(name), std::move(info), halos) {
                    init(initializer);
                }
                T const *get_target_ptr() const { return this->raw_target_ptr(); }
                auto target_view() const { return traits::make_target_view<Traits>(get_target_ptr(), this->info()); }
                auto get_const_target_ptr() const { return get_target_ptr(); }
                auto const_target_view() const { return target_view(); }
            };

            template <class>
            struct is_data_store : std::false_type {};

            template <class Traits, class T, class Info, class Id>
            struct is_data_store<data_store<Traits, T, Info, Id>> : std::true_type {};

            template <class>
            struct is_data_store_ptr : std::false_type {};

            template <class Traits, class T, class Info, class Id>
            struct is_data_store_ptr<std::shared_ptr<data_store<Traits, T, Info, Id>>> : std::true_type {};

            template <class Traits, class T, class Kind, class Info, class Halos, class Initializer>
            auto make_data_store_helper(
                std::string name, Info info, Halos const &halos, Initializer const &initializer) {
                return std::make_shared<data_store<Traits, T, Info, Kind>>(
                    std::move(name), std::move(info), halos, initializer);
            }

            template <class Traits, class T, class Id, class Lengths, class Halos, class Initializer>
            auto make_data_store(
                std::string name, Lengths const &lengths, Halos const &halos, Initializer const &initializer) {
                return make_data_store_helper<Traits, T, traits::strides_kind<Traits, T, Lengths, Id>>(
                    std::move(name), traits::make_info<Traits, T>(lengths), halos, initializer);
            }

            template <class DataStore>
            auto make_total_lengths(DataStore const &ds) {
                auto &&strides = ds.strides();
                using layout_t = typename DataStore::layout_t;
                std::decay_t<decltype(ds.lengths())> res;
                for (int i = 0; i != DataStore::ndims; ++i) {
                    auto n = layout_t::at(i);
                    res[i] = n == -1 ? 1 : n == 0 ? ds.lengths()[i] : strides[layout_t::find(n - 1)] / strides[i];
                }
                return res;
            }
        } // namespace data_store_impl_
        using data_store_impl_::is_data_store;
        using data_store_impl_::is_data_store_ptr;
        using data_store_impl_::make_data_store;
        using data_store_impl_::make_total_lengths;
    } // namespace storage
} // namespace gridtools
