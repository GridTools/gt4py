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
#include <type_traits>
#include <utility>

#include "../common/defs.hpp"
#include "../common/for_each.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "concept.hpp"
#include "unknown_kind.hpp"

namespace gridtools {
    namespace sid {
        namespace composite {
            namespace impl_ {
                namespace lazy {
                    template <class State, class Kind>
                    struct make_map_helper {
                        using map_t = meta::first<State>;
                        using kinds_t = meta::second<State>;
                        using pos_t = typename meta::length<map_t>::type;
                        using kind_pos_t = typename meta::find<kinds_t, Kind>::type;
                        using new_map_t = meta::push_back<map_t, tuple<pos_t, kind_pos_t>>;
                        using new_kinds_t = meta::if_c<kind_pos_t::value == meta::length<kinds_t>::value,
                            meta::push_back<kinds_t, Kind>,
                            kinds_t>;
                        using type = meta::list<new_map_t, new_kinds_t>;
                    };
                } // namespace lazy
                GT_META_DELEGATE_TO_LAZY(make_map_helper, (class State, class Kind), (State, Kind));

                template <class Kinds>
                using make_index_map =
                    meta::first<meta::foldl<make_map_helper, meta::list<tuple<>, meta::list<>>, Kinds>>;

                template <class... Items>
                struct check_strides_of_the_same_kind : std::true_type {
                    static_assert(meta::are_same<meta::second<Items>...>() ||
                                      std::conjunction<std::is_same<meta::first<Items>, unknown_kind>...>(),
                        "SIDs with the same strides kinds must have the same strides types");
                };

                template <size_t>
                struct unique_kind {};

                template <class Kind, class I>
                using replace_unknown_kind = meta::if_<std::is_same<Kind, unknown_kind>, unique_kind<I::value>, Kind>;

                template <class Kinds>
                using replace_unknown_kinds =
                    meta::transform<replace_unknown_kind, Kinds, meta::make_indices_for<Kinds>>;

                /**
                 *  `maybe_equal(lhs, rhs)` is a functional equivalent of the following pseudo code:
                 *   `<no_equal_operator_exists> || lhs == rhs;`
                 *
                 *   It is implemented as following:
                 *   - the first overload can be chosen only if `lhs == rhs` defined [SFINAE: `decltype(lhs == rhs)` is
                 * a part of the signature]
                 *   - the second overload can be chosen only if the first failed.
                 */
                template <class T>
                auto maybe_equal(T const &lhs, T const &rhs) -> decltype(lhs == rhs) {
                    return lhs == rhs;
                }

                inline bool maybe_equal(...) { return true; }

                template <class PrimaryValue, class Tup>
                bool are_secondaries_equal_to_primary(PrimaryValue const &, Tup const &) {
                    return true;
                }

                template <class SecondaryIndex, class... SecondaryIndices, class PrimaryValue, class Tup>
                bool are_secondaries_equal_to_primary(PrimaryValue const &primary_value, Tup const &tup) {
                    return are_secondaries_equal_to_primary<SecondaryIndices...>(primary_value, tup) &&
                           maybe_equal(tuple_util::get<SecondaryIndex::value>(tup), primary_value);
                }

                template <class>
                struct item_generator;

                template <template <class...> class L, class Key, class PrimaryIndex, class... SecondaryIndices>
                struct item_generator<L<Key, PrimaryIndex, SecondaryIndices...>> {
                    template <class Args, class Res = tuple_util::element<PrimaryIndex::value, Args>>
                    decltype(auto) operator()(Args const &args) const noexcept {
                        static_assert(std::conjunction_v<
                                          std::is_same<tuple_util::element<SecondaryIndices::value, Args>, Res>...>,
                            GT_INTERNAL_ERROR);
                        assert((are_secondaries_equal_to_primary<SecondaryIndices...>(
                            tuple_util::get<PrimaryIndex::value>(args), args)));
                        return tuple_util::get<PrimaryIndex::value>(args);
                    }
                };

                template <class ObjTup, class StrideTup, class Offset>
                GT_FUNCTION void composite_shift_impl(ObjTup &obj_tup, StrideTup &&stride_tup, Offset offset) {
                    tuple_util::host_device::for_each(
                        [offset](auto &obj, auto &&stride)
                            GT_FORCE_INLINE_LAMBDA { shift(obj, std::forward<decltype(stride)>(stride), offset); },
                        obj_tup,
                        std::forward<StrideTup>(stride_tup));
                }

                template <class Key, class Strides, class I = meta::st_position<get_keys<Strides>, Key>>
                using normalized_stride_type =
                    typename std::conditional_t<(I::value < tuple_util::size<Strides>::value),
                        tuple_util::lazy::element<I::value, Strides>,
                        meta::lazy::id<default_stride>>::type;

                template <class Keys>
                struct normalize_strides_f;

                template <template <class...> class L, class... Keys>
                struct normalize_strides_f<L<Keys...>> {
                    template <class Sid, class Strides = strides_type<Sid>>
                    tuple<normalized_stride_type<Keys, std::decay_t<Strides>>...> operator()(Sid const &sid) const {
                        return {get_stride<Keys>(get_strides(sid))...};
                    }
                };

                struct ctor_tag {};

                struct sum {
                    template <class Lhs, class Rhs>
                    GT_FUNCTION constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const {
                        return std::forward<Lhs>(lhs) + std::forward<Rhs>(rhs);
                    }
                };
            } // namespace impl_

            /**
             *  This class models both `SID` and `hymap` concepts at the same time
             *
             *  All derived types of the `composite` are `hymap`'s of the correspondent types of the original ones.
             *  Example:
             *    say you have two sids: `s1` and `s2` of types `S1` and `S2`
             *    you can compose them: `composite::keys<a, b>::values<S1, S2> c = {s1, s2};`
             *    now `c` is a `SID` as well, you can call `get_origin(c)`, `get_strides(c)` etc.
             *    you have an access to `s1` and `s2` via `get_key`: `get_key<a>(c)` is the same as `s1`
             *
             *  The way how the composite deals with strides is easy to illustrate with example:
             *  Say the sid `s1` has strides {dim_i:3, dim_j:15} (here I use pseudo code to express the map)
             *  and `s2` has strides {dim_k:4, dim_j:12}.
             *  We create a composite `c`: `composite::keys<a, b>::values<S1, S2> c = {s1, s2}`
             *  Now `c` has strides : {dim_i:{a:3, b:0}, dim_j:{a:15, b:12}, dim_k:{a:0, b:4}}
             *
             * Internally `composite` utilizes the fact that the strides of the same kind are always the same.
             * This  means that the composite stride doesn't hold duplicate strides.
             *
             */
            template <class...>
            struct keys {
                template <class...>
                struct composite_ptr;
                template <class...>
                struct composite_ptr_holder;
                template <class>
                struct compressed;
                template <class...>
                struct values;
#if !defined(__NVCC__) && defined(__clang__) && __clang_major__ <= 15
                template <class... Sids>
                values(Sids const &...) -> values<Sids...>;
#endif
                // NVCC 11 fails to do class template deduction in the case of nested templates
                template <class... Args>
                static constexpr GT_FUNCTION values<Args...> make_values(Args &&...args) {
                    return {std::forward<Args>(args)...};
                }
            };

            template <class... Keys>
            template <class... Ptrs>
            struct keys<Keys...>::composite_ptr {
                static_assert(sizeof...(Keys) == sizeof...(Ptrs), GT_INTERNAL_ERROR);

                tuple<Ptrs...> m_vals;

                template <class... Args,
                    std::enable_if_t<std::conjunction_v<std::is_constructible<Ptrs, Args>...>, int> = 0>
                constexpr GT_FUNCTION composite_ptr(Args &&...args) noexcept : m_vals{std::forward<Args>(args)...} {}

                constexpr GT_FUNCTION composite_ptr(Ptrs const &...args) noexcept : m_vals(args...) {}

                constexpr GT_FUNCTION composite_ptr(tuple<Ptrs...> &&args) noexcept : m_vals(std::move(args)) {}
                constexpr GT_FUNCTION composite_ptr(tuple<Ptrs...> const &args) noexcept : m_vals(args) {}

                composite_ptr() = default;
                composite_ptr(composite_ptr const &) = default;
                composite_ptr(composite_ptr &&) = default;
                composite_ptr &operator=(composite_ptr const &) = default;
                composite_ptr &operator=(composite_ptr &&) = default;

                GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite_ptr, m_vals);

                constexpr GT_FUNCTION decltype(auto) operator*() const {
                    return tuple_util::host_device::convert_to<hymap::keys<Keys...>::template values>(
                        tuple_util::host_device::transform([](auto const &ptr)
// Workaround for GCC 9 bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90333
// The failure is observed with 9.3 as well even though they say it was already fixed there.
// gcc 10.1, 10.2 fails here as well. Disabling for all gcc 9 and 10 versions...
#if defined(__clang__) || !defined(__GNUC__) || (__GNUC__ != 9 && __GNUC__ != 10)
                                                               GT_FORCE_INLINE_LAMBDA
#endif
                            -> decltype(auto) { return *ptr; },
                            m_vals));
                }

                friend keys hymap_get_keys(composite_ptr const &) { return {}; }
            };

            template <class... Keys>
            template <class... PtrHolders>
            struct keys<Keys...>::composite_ptr_holder {
                static_assert(sizeof...(Keys) == sizeof...(PtrHolders), GT_INTERNAL_ERROR);

                tuple<PtrHolders...> m_vals;

                template <class... Args,
                    std::enable_if_t<std::conjunction_v<std::is_constructible<PtrHolders, Args>...>, int> = 0>
                constexpr GT_FUNCTION composite_ptr_holder(Args &&...args) noexcept
                    : m_vals{std::forward<Args>(args)...} {}

                constexpr GT_FUNCTION composite_ptr_holder(PtrHolders const &...args) noexcept : m_vals(args...) {}

                constexpr GT_FUNCTION composite_ptr_holder(tuple<PtrHolders...> &&args) noexcept
                    : m_vals(std::move(args)) {}
                constexpr GT_FUNCTION composite_ptr_holder(tuple<PtrHolders...> const &args) noexcept : m_vals(args) {}

                composite_ptr_holder() = default;
                composite_ptr_holder(composite_ptr_holder const &) = default;
                composite_ptr_holder(composite_ptr_holder &&) = default;
                composite_ptr_holder &operator=(composite_ptr_holder const &) = default;
                composite_ptr_holder &operator=(composite_ptr_holder &&) = default;

                GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(composite_ptr_holder, m_vals);

                constexpr GT_FUNCTION auto operator()() const {
                    return tuple_util::host_device::convert_to<composite_ptr>(tuple_util::host_device::transform(
                        [](auto const &obj) GT_FORCE_INLINE_LAMBDA { return obj(); }, m_vals));
                }

                friend keys hymap_get_keys(composite_ptr_holder const &) { return {}; }
            };

            /**
             *  Implements strides and ptr_diffs compression based on skipping the objects of the
             *  same kind.
             *
             *  `composite` objects pretend to be a tuples of `uncompressed` types (aka external tuple), but
             * internally they store a tuple of `compressed` types (aka internal tuple).
             *
             *  @tparam Map - compile time map from the index in external tuple to the index of internal tuple.
             *                `compile time map` here is a tuple of tuples: `tuple<tuple<Key, Value>...>`
             */
            template <class... Keys>
            template <class Map>
            struct keys<Keys...>::compressed {
                /**
                 *  Inverse map is like that: tuple<tuple<Value, Keys...>...>, where values and keys are taken
                 *  from the source.
                 *
                 *  In the concrete case of the map of indices it would be:
                 *  tuple<tuple<CompressedIndex, UncompressedIndices...>>
                 *
                 *  Note that it could be several UncompressedIndices per one CompressedIndex
                 *
                 */
                using inversed_map_t = meta::mp_inverse<Map>;
                /**
                 *  A tuple of the first of UncompressedIndices per each CompressedIndex
                 */
                using primary_indices_t = meta::transform<meta::second, inversed_map_t>;

                using generators_t = meta::transform<impl_::item_generator, inversed_map_t>;

                template <size_t I,
                    class T,
                    class Item = meta::mp_find<Map, std::integral_constant<size_t, I>>,
                    class Pos = meta::second<Item>>
                static constexpr GT_FUNCTION decltype(auto) get(T &&obj) {
                    return tuple_util::host_device::get<Pos::value>(std::forward<T>(obj).m_vals);
                }

                template <class... Ts>
                struct composite_entity {
                    static_assert(sizeof...(Keys) == sizeof...(Ts), GT_INTERNAL_ERROR);

                    template <class I>
                    using get_compressed_type = meta::at<meta::list<Ts...>, I>;

                    using vals_t = meta::transform<get_compressed_type, primary_indices_t>;

                    vals_t m_vals;

                    template <template <class...> class L, class... Args>
                    composite_entity(impl_::ctor_tag, L<Args...> &&tup) noexcept
                        : m_vals{tuple_util::generate<generators_t, vals_t>(std::move(tup))} {
                        static_assert(sizeof...(Args) == sizeof...(Keys), GT_INTERNAL_ERROR);
                    }

                    composite_entity() = default;
                    composite_entity(composite_entity const &) = default;
                    composite_entity(composite_entity &&) noexcept = default;
                    composite_entity &operator=(composite_entity const &) = default;
                    composite_entity &operator=(composite_entity &&) noexcept = default;

                    friend compressed tuple_getter(composite_entity const &) { return {}; }

                    template <class... Ptrs>
                    friend constexpr GT_FUNCTION composite_ptr<Ptrs...> operator+(
                        composite_ptr<Ptrs...> const &lhs, composite_entity const &rhs) {
                        return tuple_util::host_device::transform(impl_::sum(), lhs, rhs);
                    }

                    template <class... PtrHolders>
                    friend composite_ptr_holder<PtrHolders...> operator+(
                        composite_ptr_holder<PtrHolders...> const &lhs, composite_entity const &rhs) {
                        return tuple_util::transform(impl_::sum(), lhs, rhs);
                    }

                    template <class... Ptrs, class Offset>
                    friend GT_FUNCTION void sid_shift(
                        composite_ptr<Ptrs...> &ptr, composite_entity const &stride, Offset offset) {
                        impl_::composite_shift_impl(ptr.m_vals, stride, offset);
                    }

                    template <class... Ptrs, class Offset>
                    friend GT_FUNCTION void sid_shift(
                        composite_ptr<Ptrs...> &ptr, composite_entity &&stride, Offset offset) {
                        impl_::composite_shift_impl(ptr.m_vals, std::move(stride), offset);
                    }

                    friend keys hymap_get_keys(composite_entity const &) { return {}; }
                };

                template <class... PtrDiffs, class... Strides, class Offset>
                friend GT_FUNCTION void sid_shift(composite_entity<PtrDiffs...> &ptr_diff,
                    composite_entity<Strides...> const &stride,
                    Offset offset) {
                    impl_::composite_shift_impl(ptr_diff.m_vals, stride.m_vals, offset);
                }

                template <class... PtrDiffs, class... Strides, class Offset>
                friend GT_FUNCTION void sid_shift(
                    composite_entity<PtrDiffs...> &ptr_diff, composite_entity<Strides...> &&stride, Offset offset) {
                    impl_::composite_shift_impl(ptr_diff.m_vals, std::move(stride.m_vals), offset);
                }

                struct convert_f {
                    template <template <class...> class L, class... Ts>
                    composite_entity<std::remove_reference_t<Ts>...> operator()(L<Ts...> &&tup) const {
                        return {impl_::ctor_tag(), std::move(tup)};
                    }
                };
            };

            template <class... Keys>
            template <class... Sids>
            struct keys<Keys...>::values {
                static_assert(sizeof...(Keys) == sizeof...(Sids), GT_INTERNAL_ERROR);
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 1
#else
                static_assert(std::conjunction<is_sid<Sids>...>::value, GT_INTERNAL_ERROR);
#endif
                static_assert(meta::all<meta::mp_make<impl_::check_strides_of_the_same_kind,
                        meta::list<meta::list<strides_kind<Sids>, strides_type<Sids>>...>>>());

                tuple<Sids...> m_sids;

                // Extracted lists of raw kinds (uncompressed)
                using strides_kinds_t = meta::list<strides_kind<Sids>...>;

                // The index map that is needed to build compressed composite objects
                using map_t = impl_::make_index_map<impl_::replace_unknown_kinds<strides_kinds_t>>;
                using compressed_t = compressed<map_t>;

                template <class... Ts>
                using compress = typename compressed_t::template composite_entity<Ts...>;

                using stride_keys_t = meta::dedup<meta::concat<get_keys<strides_type<Sids>>...>>;

                using stride_hymap_keys_t = meta::rename<hymap::keys, stride_keys_t>;

                template <class... Values>
                using stride_hymap_ctor = typename stride_hymap_keys_t::template values<Values...>;

                // A helper for generating strides_t
                // It is a meta function from the stride key to the stride type
                template <class Key>
                using get_stride_type =
                    compress<impl_::normalized_stride_type<Key, std::decay_t<strides_type<Sids>>>...>;

                // all `SID` types are here
                using ptr_holder_t = composite_ptr_holder<ptr_holder_type<Sids>...>;
                using ptr_t = composite_ptr<ptr_type<Sids>...>;
                using strides_t = meta::rename<stride_hymap_ctor, meta::transform<get_stride_type, stride_keys_t>>;
                using ptr_diff_t = compress<ptr_diff_type<Sids>...>;

                // Here the `SID` concept is modeled

                friend ptr_holder_t sid_get_origin(values &obj) {
                    return tuple_util::transform(
                        [](auto &obj) GT_FORCE_INLINE_LAMBDA { return get_origin(obj); }, obj.m_sids);
                }

                template <class U = strides_t, std::enable_if_t<!meta::is_empty<U>::value, int> = 0>
                friend strides_t sid_get_strides(values const &obj) {
                    return tuple_util::transform(typename compressed_t::convert_f(),
                        tuple_util::transpose(
                            tuple_util::transform(impl_::normalize_strides_f<stride_keys_t>(), obj.m_sids)));
                }

                template <class U = strides_t, std::enable_if_t<meta::is_empty<U>::value, int> = 0>
                friend strides_t sid_get_strides(values const &) {
                    return {};
                }

                friend ptr_diff_t sid_get_ptr_diff(values const &) { return {}; }

                friend unknown_kind sid_get_strides_kind(values const &) { return {}; }

                // Here the `tuple_like` concept is modeled
                struct getter {
                    template <size_t I>
                    static decltype(auto) get(values const &obj) noexcept {
                        return tuple_util::get<I>(obj.m_sids);
                    }
                    template <size_t I>
                    static decltype(auto) get(values &obj) noexcept {
                        return tuple_util::get<I>(obj.m_sids);
                    }
                    template <size_t I>
                    static decltype(auto) get(values &&obj) noexcept {
                        return tuple_util::get<I>(std::move(obj).m_sids);
                    }
                };
                friend getter tuple_getter(values const &) { return {}; }

                template <class... Args,
                    std::enable_if_t<std::conjunction_v<std::is_constructible<Sids, Args>...>, int> = 0>
                constexpr values(Args &&...args) noexcept : m_sids{std::forward<Args>(args)...} {}

                constexpr values(Sids const &...args) noexcept : m_sids(args...) {}

                constexpr values(tuple<Sids...> &&args) noexcept : m_sids(std::move(args)) {}
                constexpr values(tuple<Sids...> const &args) noexcept : m_sids(args) {}

                values() = default;
                values(values const &) = default;
                values(values &&) = default;
                values &operator=(values const &) = default;
                values &operator=(values &&) = default;

                // hymap concept
                friend keys hymap_get_keys(values const &) { return {}; }
            };

            template <>
            template <>
            struct keys<>::values<> {
                constexpr GT_FUNCTION values operator()() const { return {}; }
                constexpr GT_FUNCTION values operator*() const { return {}; }
                friend constexpr GT_FUNCTION values operator+(values, values) { return {}; }
                friend values sid_get_origin(values) { return {}; }
                friend values sid_get_ptr_diff(values) { return {}; }
                friend values tuple_getter(values) { return {}; }
            };
        } // namespace composite
    }     // namespace sid
} // namespace gridtools
