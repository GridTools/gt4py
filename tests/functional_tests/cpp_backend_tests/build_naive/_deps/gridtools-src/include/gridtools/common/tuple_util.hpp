/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \addtogroup common
    @{
*/
/** \addtogroup tupleutils Utilities for Tuples
    @{
*/

/**
 *  @file
 *
 *  Here is a set of algorithms that are defined on "tuple like" structures
 *
 *  The formal definition of the "tuple like" concept:
 *  A type `%T` satisfies the concept if:
 *    - it is move constructible;
 *
 *    - it can be constructed element wise using brace initializer syntax;
 *      [example: `my_triple<T1, T2, T3> val = {elem0, elem1, elem2};` ]
 *
 *    - a function `tuple_to_types(T)` can be found from the `::gridtools::tuple_util::traits` namespace [via ADL or
 *      directly in this namespace]. This function should return a type, which is instantiation of a template
 *      parameterized on types. The actual parameters of that instantiation is interpreted as a types of elements of the
 *      "tuple like". Simply speaking `tuple_to_types` returns a type list of the types of `T` elements.
 *      Note that this function (and for others in this concept definition as well) will be never called. It is enough
 *      to just declare it. Example:
 *      \code
 *      // for the simple "tuple_like"'s it is enough to return itself from tuple_to_types
 *      template <class T, class U, class Q>
 *      my_triple<T, U, Q> tuple_to_types(my_triple<T, U, Q>);
 *      \endcode
 *
 *    - a function `tuple_from_types(T)` can be found from the `::gridtools::tuple_util::traits` namespace [via ADL or
 *      directly in this namespace]. This function should return a type which is a meta class [in the terms of `meta`
 *      library]. This meta class should contain a meta function that takes types elements that we are going to pass to
 *      brace initializer and returns a type [satisfying the same concept] that can accept such a list. Example:
 *      \code
 *      struct my_triple_from_types {
 *          template <class T, class U, class Q>
 *          using apply = my_triple<T, U, Q>;
 *      };
 *      template <class T, class U, class Q>
 *      my_triple_from_types tuple_from_types(my_triple<T, U, Q>);
 *      \endcode
 *
 *    - a function `tuple_getter(T)` can be found from the `::gridtools::tuple_util::traits` namespace [via ADL or
 *      directly in this namespace]. This function should return a type that has a static template (on size_t) member
 *      function called `get<N>` that accepts `T` by some reference and returns the Nth element of `T`. Example:
 *      \code
 *      struct my_triple_getter {
 *          template <size_t N, class T, class U, class Q, std::enable_if_t<N == 0, int> = 0>
 *          static T get(my_triple<T, U, Q> obj) { return obj.first; }
 *          ...
 *      };
 *      template <class T, class U, class Q>
 *      my_triple_getter tuple_getter(my_triple<T, U, Q>);
 *      \endcode
 *
 *  If the opposite is not mentioned explicitly, the algorithms produce tuples of references. L-value or R-value
 *  depending on algorithm input.
 *
 *  Almost all algorithms are defined in two forms:
 *    1) conventional template functions;
 *    2) functions that return generic functors
 *
 *  For example you can do:
 *  \code
 *    auto ints = transform([](int x) {return x;}, input);
 *  \endcode
 *  or you can:
 *  \code
 *    auto convert_to_ints = transform([](int x) {return x;});
 *    auto ints = convert_to_ints(input);
 *  \endcode
 *
 *  The second form is more composable. For example if the input is a tuple of tuples of whatever and you need a
 *  tuple of tuple of tuple of integers you can do it in one expression:
 *  \code
 *  auto out = transform(transform([](int x) {return x;}), input);
 *  \endcode
 *
 *
 *  TODO list
 *  =========
 *  - add filter
 *
 */

#ifndef GT_TARGET_ITERATING
//// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_TUPLE_UTIL_HPP_
#define GT_COMMON_TUPLE_UTIL_HPP_

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "../meta.hpp"
#include "defs.hpp"
#include "functional.hpp"
#include "host_device.hpp"

#define GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(class_name, member_name)                        \
    struct class_name##_tuple_util_getter {                                                    \
        template <::std::size_t I>                                                             \
        static constexpr GT_FUNCTION decltype(auto) get(class_name const &obj) {               \
            return ::gridtools::tuple_util::host_device::get<I>(obj.member_name);              \
        }                                                                                      \
        template <::std::size_t I>                                                             \
        static constexpr GT_FUNCTION decltype(auto) get(class_name &obj) {                     \
            return ::gridtools::tuple_util::host_device::get<I>(obj.member_name);              \
        }                                                                                      \
        template <::std::size_t I>                                                             \
        static constexpr GT_FUNCTION decltype(auto) get(class_name &&obj) {                    \
            return ::gridtools::tuple_util::host_device::get<I>(::std::move(obj).member_name); \
        }                                                                                      \
    };                                                                                         \
    friend class_name##_tuple_util_getter tuple_getter(class_name const &) { return {}; }      \
    static_assert(1)

#define GT_STRUCT_TUPLE_IMPL_DECL_(r, data, elem) BOOST_PP_TUPLE_ELEM(0, elem) BOOST_PP_TUPLE_ELEM(1, elem);
#define GT_STRUCT_TUPLE_IMPL_TYPE_(s, data, elem) BOOST_PP_TUPLE_ELEM(0, elem)
#define GT_STRUCT_TUPLE_IMPL_GETS_(s, name, i, elem)                                                        \
    template <::std::size_t I, ::std::enable_if_t<I == i, int> = 0, class T = BOOST_PP_TUPLE_ELEM(0, elem)> \
    static constexpr GT_FUNCTION T const &get(name const &obj) {                                            \
        return obj.BOOST_PP_TUPLE_ELEM(1, elem);                                                            \
    }                                                                                                       \
    template <::std::size_t I, ::std::enable_if_t<I == i, int> = 0, class T = BOOST_PP_TUPLE_ELEM(0, elem)> \
    static constexpr GT_FUNCTION T &get(name &obj) {                                                        \
        return obj.BOOST_PP_TUPLE_ELEM(1, elem);                                                            \
    }                                                                                                       \
    template <::std::size_t I, ::std::enable_if_t<I == i, int> = 0, class T = BOOST_PP_TUPLE_ELEM(0, elem)> \
    static constexpr GT_FUNCTION T &&get(name &&obj) {                                                      \
        return static_cast<T &&>(obj.BOOST_PP_TUPLE_ELEM(1, elem));                                         \
    }
#define GT_STRUCT_TUPLE_IMPL_(name, members)                                                                          \
    BOOST_PP_SEQ_FOR_EACH(GT_STRUCT_TUPLE_IMPL_DECL_, _, members)                                                     \
    struct gt_##name##_tuple_getter {                                                                                 \
        BOOST_PP_SEQ_FOR_EACH_I(GT_STRUCT_TUPLE_IMPL_GETS_, name, members)                                            \
    };                                                                                                                \
    friend gt_##name##_tuple_getter tuple_getter(name);                                                               \
    friend ::gridtools::meta::list<BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_TRANSFORM(GT_STRUCT_TUPLE_IMPL_TYPE_, _, members))> \
        tuple_to_types(name);                                                                                         \
    friend ::gridtools::meta::always<name> tuple_from_types(name)

/*
 * Struct adapter to tuple_like
 *
 * Usage:
 * Declaring a struct like this, makes it tuple-like:
 * ```
 *  struct foo {
 *     GT_STRUCT_TUPLE(foo,
 *          (int a),
 *          (double b)
 *     );
 *     // some methods can be declared here as well
 *  };
 * ```
 * I.e. one can access the members both by name and by `get` accessor:
 * ```
 *    foo obj;
 *    obj.a = 42;
 *    assert(tuple_util::get<1>(obj) == 42);
 * ```
 * Also tuple algorithms works with `foo` as expected:
 * ```
 *    auto x = tuple_util::trasnform([](auto x) { return x * 2; }, foo{1, 2.5});
 *    assert(x.a == 2);
 *    assert(x.b == 5.);
 * ```
 */
#define GT_STRUCT_TUPLE(name, ...) GT_STRUCT_TUPLE_IMPL_(name, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

namespace gridtools {
    namespace tuple_util {

        /// @cond
        namespace traits {
            namespace _impl {

                template <class... Ts>
                struct deduce_array_type : std::common_type<Ts...> {};

                template <>
                struct deduce_array_type<> {
                    using type = meta::lazy::id<void>;
                };

                /// implementation of the `from_types`  for `std::array`
                //
                struct array_from_types {
                    template <class... Ts>
                    using apply = std::array<typename deduce_array_type<Ts...>::type, sizeof...(Ts)>;
                };

                /// getter for the standard "tuple like" entities: `std::tuple`, `std::pair` and `std::array`
                //
                struct std_getter {
                    template <size_t I, class T>
                    GT_FORCE_INLINE static constexpr decltype(auto) get(T &&obj) noexcept {
                        return std::get<I>(std::forward<T>(obj));
                    }
                };
            } // namespace _impl

            // start of builtin adaptations

            // to_types

            // generic `tuple_to_types` that works for `std::tuple`, `std::pair` and its clones.
            // It just returns an argument;
            template <template <class...> class L, class... Ts>
            L<Ts...> tuple_to_types(L<Ts...>);

            // `std::array` specialization. Returns the type from array repeated N times.
            template <class T, size_t N>
            meta::repeat_c<N, meta::list<T>> tuple_to_types(std::array<T, N>);

            // from_types

            template <class T>
            meta::always<T> tuple_from_types(T);

            // generic `tuple_from_types` that works for `std::tuple`, `std::pair` and its clones.
            // meta constructor 'L' is extracted and used to build the new "tuple like"
            template <template <class...> class L, class... Ts>
            meta::ctor<L<Ts...>> tuple_from_types(L<Ts...>);

            // arrays specialization.
            template <class T, size_t N>
            _impl::array_from_types tuple_from_types(std::array<T, N>);

            // getter

            // all `std` "tuple_like"s use `std::get`
            template <class... Ts>
            _impl::std_getter tuple_getter(std::tuple<Ts...>);
            template <class T, class U>
            _impl::std_getter tuple_getter(std::pair<T, U>);
            template <class T, size_t N>
            _impl::std_getter tuple_getter(std::array<T, N>);

            // end of builtin adaptations

            // Here ADL definitions of `tuple_*` functions are picked up
            // The versions in this namespace will be chosen if nothing is found by `ADL`.
            // it is important to have all builtins above this line.

            template <class T>
            decltype(tuple_getter(std::declval<T>())) get_getter(T);
            template <class T>
            decltype(tuple_to_types(std::declval<T>())) get_to_types(T);
            template <class T>
            decltype(tuple_from_types(std::declval<T>())) get_from_types(T);

            template <class T>
            using getter = decltype(::gridtools::tuple_util::traits::get_getter(std::declval<T>()));
            template <class T>
            using to_types = decltype(::gridtools::tuple_util::traits::get_to_types(std::declval<T>()));
            template <class T>
            using from_types = decltype(::gridtools::tuple_util::traits::get_from_types(std::declval<T>()));
        } // namespace traits
        /// @endcond

        ///  Generalization of std::tuple_size
        //
        template <class T>
        using size = meta::length<traits::to_types<T>>;

        ///  Generalization of std::tuple_element
        //
        namespace lazy {
            template <size_t I, class T>
            struct element {
                using type = meta::at_c<traits::to_types<T>, I>;
            };
            template <size_t I, class T>
            struct element<I, T const> {
                using type = std::add_const_t<meta::at_c<traits::to_types<T>, I>>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(element, (size_t I, class T), (I, T));

        template <class T, class = void>
        struct is_empty_or_tuple_of_empties : std::is_empty<T> {};

        template <class Tup, class Types = traits::to_types<Tup>>
        using is_tuple_of_empties = meta::all_of<is_empty_or_tuple_of_empties, Types>;

        template <class Tup>
        struct is_empty_or_tuple_of_empties<Tup, std::enable_if_t<is_tuple_of_empties<Tup>::value>> : std::true_type {};

        // Here goes the stuff that is common for all targets (meta functions)
        namespace _impl {

            template <class T>
            using to_types = meta::rename<meta::list, traits::to_types<std::decay_t<T>>>;

            template <class Sample, class Types, class FromTypesMetaClass = traits::from_types<std::decay_t<Sample>>>
            using from_types = meta::rename<FromTypesMetaClass::template apply, Types>;

            enum class ref_kind { rvalue, lvalue, const_lvalue };

            template <class>
            struct get_ref_kind : std::integral_constant<ref_kind, ref_kind::rvalue> {};

            template <class T>
            struct get_ref_kind<T &> : std::integral_constant<ref_kind, ref_kind::lvalue> {};

            template <class T>
            struct get_ref_kind<T const &> : std::integral_constant<ref_kind, ref_kind::const_lvalue> {};

            namespace lazy {
                template <ref_kind Kind, class Dst>
                struct add_ref;

                template <class T>
                struct add_ref<ref_kind::rvalue, T> {
                    using type = T;
                };

                template <class T>
                struct add_ref<ref_kind::lvalue, T> : std::add_lvalue_reference<T> {};

                template <class T>
                struct add_ref<ref_kind::const_lvalue, T> : std::add_lvalue_reference<std::add_const_t<T>> {};
            } // namespace lazy
            GT_META_DELEGATE_TO_LAZY(add_ref, (ref_kind Kind, class Dst), (Kind, Dst));

            template <ref_kind Kind>
            struct get_accessor {
                template <class T>
                using apply = add_ref<Kind, T>;
            };

            template <class Fun>
            struct get_fun_result_index {
                template <class I, class... Ts>
                using apply =
                    decltype(std::declval<Fun const &>().template operator()<I::value>(std::declval<Ts>()...));
            };

            template <class Fun>
            struct get_fun_result {
                template <class... Ts>
                using apply =
                    decltype(std::declval<std::add_lvalue_reference_t<std::add_const_t<Fun>>>()(std::declval<Ts>()...));
            };

            template <class Tup>
            using get_accessors =
                meta::transform<get_accessor<get_ref_kind<Tup>::value>::template apply, to_types<Tup>>;

            template <class D, class... Ts>
            struct make_array_helper {
                using type = D;
            };

            template <class... Ts>
            struct make_array_helper<void, Ts...> : std::common_type<Ts...> {};

            template <template <class...> class L>
            struct to_tuple_converter_helper {
                template <class... Ts>
                using apply = L<Ts...>;
            };

            template <template <class, size_t> class Arr, class D>
            struct to_array_converter_helper {
                template <class... Ts>
                using apply = Arr<typename make_array_helper<D, Ts...>::type, sizeof...(Ts)>;
            };

            struct default_concat_result_maker_f {
                template <class FlattenTypes, class Tuples>
                using apply = from_types<meta::first<Tuples>, FlattenTypes>;
            };

            template <template <class...> class Pred>
            struct group_predicate_proxy_f {
                template <class... TypesAndIndices>
                using apply = Pred<meta::first<TypesAndIndices>...>;
            };

            template <class... TypesAndIndices>
            using extract_indices = meta::list<meta::second<TypesAndIndices>...>;

            template <template <class...> class Pred, class Types>
            using group_indices = meta::group<group_predicate_proxy_f<Pred>::template apply,
                extract_indices,
                meta::zip<Types, meta::make_indices_for<Types>>>;

            template <class...>
            struct is_constructible_from_elements : std::false_type {};

            template <class T, template <class...> class L, class... Ts>
            struct is_constructible_from_elements<T, L<Ts...>> : std::is_same<decltype(T{std::declval<Ts>()...}), T> {};

            template <class...>
            struct is_getter_valid;

            template <class T, class Getter, class Types, class I>
            struct is_getter_valid<T, Getter, Types, I>
                : std::bool_constant<std::is_same_v<decltype(Getter::template get<I::value>(std::declval<T const &>())),
                                         meta::at<Types, I> const &> &&
                                     std::is_same_v<decltype(Getter::template get<I::value>(std::declval<T &>())),
                                         meta::at<Types, I> &> &&
                                     std::is_same_v<decltype(Getter::template get<I::value>(std::declval<T &&>())),
                                         meta::at<Types, I> &&>> {};

            template <class T,
                class Types = traits::to_types<T>,
                class FromTypes = traits::from_types<T>,
                class Getter = traits::getter<T>>
            struct is_tuple_like
                : std::bool_constant<
                      // `to_types` produces a type list
                      meta::is_list<Types>::value &&
                      // there are no `void`'s within the types
                      meta::all_of<meta::not_<std::is_void>::apply, Types>::value &&
                      // FromTypes metafunction is responsible for producing another `tuple_like` of the given kind from
                      // the list of types. We can not check it for all possible types but we at least can ensure that
                      // if we apply it to the current element types, we will get the current `tuple_like` type back.
                      std::is_same_v<meta::rename<FromTypes::template apply, Types>, T> &&
                      // we should be able to construct `tuple_like` element wise
                      is_constructible_from_elements<T, Types>::value &&
                      // iff element types are all move_constructible, `tuple_like` is move constructible
                      meta::all_of<std::is_move_constructible, Types>::value == std::is_move_constructible_v<T> &&
                      // the same for copy_constructible
                      meta::all_of<std::is_copy_constructible, Types>::value == std::is_copy_constructible_v<T> &&
                      // check that the getters produce expected types for all indices
                      meta::all_of<meta::curry<is_getter_valid, T, Getter, Types>::template apply,
                          meta::make_indices_for<Types>>::value> {};
        } // namespace _impl

        template <class, class = void>
        struct is_tuple_like : std::false_type {};

        template <class T>
        struct is_tuple_like<T, std::enable_if_t<_impl::is_tuple_like<T>::value>> : std::true_type {};
    } // namespace tuple_util

    using tuple_util::is_tuple_like;

#ifdef __cpp_concepts
    namespace concepts {
        template <class T>
        concept tuple_like = is_tuple_like<T>::value;

        template <class T, class... Ts>
        concept tuple_like_of = tuple_like<T> &&
            std::is_same_v<meta::rename<meta::list, tuple_util::traits::to_types<T>>, meta::list<Ts...>>;
    } // namespace concepts
#endif

} // namespace gridtools

// Now it's time to generate host/device/host_device stuff
#define GT_FILENAME <gridtools/common/tuple_util.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_TUPLE_UTIL_HPP_
#else  // GT_TARGET_ITERATING

#ifdef GT_TARGET_HAS_DEVICE

#define DEFINE_FUNCTOR_INSTANCE(name, functor)                                \
    template <class... Args>                                                  \
    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) name(Args &&...args) { \
        return functor()(std::forward<Args>(args)...);                        \
    }                                                                         \
    static_assert(1)

#ifdef __NVCC__
#define DEFINE_TEMPLATED_FUNCTOR_INSTANCE(name, functor) GT_DEVICE constexpr functor name = {}
#else
#define DEFINE_TEMPLATED_FUNCTOR_INSTANCE(name, functor) constexpr functor name = {}
#endif

#else

#define DEFINE_FUNCTOR_INSTANCE(name, functor) constexpr functor name = {}

#define DEFINE_TEMPLATED_FUNCTOR_INSTANCE(name, functor) constexpr functor name = {}

#endif

namespace gridtools {
    namespace tuple_util {
        GT_TARGET_NAMESPACE {
            /**
             * @brief Tuple element accessor like std::get.
             *
             * @tparam I Element index.
             * @tparam T Tuple-like type.
             * @param obj Tuple-like object.
             */
            template <size_t I, class T, class Getter = traits::getter<std::decay_t<T>>>
            GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) get(T && obj) noexcept {
                return Getter::template get<I>(std::forward<T>(obj));
            }

            template <size_t I>
            struct get_nth_f {
                template <class T, class Getter = traits::getter<std::decay_t<T>>>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(T &&obj) const noexcept {
                    return Getter::template get<I>(std::forward<T>(obj));
                }
            };

            // Let us use `detail` for internal namespace of the target dependent namespace.
            // This way we can refer `_impl::foo` for the entities that are independent on the target and
            // `detail::bar` for the target dependent ones.
            namespace detail {
                using _impl::from_types;
                using _impl::get_accessors;
                using _impl::get_fun_result;
                using _impl::to_types;

                template <size_t I>
                struct transform_elem_index_f {
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Fun &&fun, Tups &&...tups) const {
                        return std::forward<Fun>(fun).template operator()<I>(
                            GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tups>(tups))...);
                    }
                };

                template <class I>
                using get_transform_index_generator = transform_elem_index_f<I::value>;

                template <class GeneratorList, class Res>
                struct generate_f;
                template <template <class...> class L, class... Generators, class Res>
                struct generate_f<L<Generators...>, Res> {
                    template <class... Args>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Args &&...args) const {
                        return Res{Generators{}(std::forward<Args>(args)...)...};
                    }
                };

                template <class Fun>
                struct transform_index_f {
                    template <class... Args>
                    using get_results_t = meta::transform<_impl::get_fun_result_index<Fun>::template apply, Args...>;

                    Fun m_fun;

                    template <class Tup, class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr auto operator()(Tup &&tup, Tups &&...tups) const {
                        using indices_t = meta::make_indices<size<std::decay_t<Tup>>>;
                        using res_t =
                            from_types<Tup, get_results_t<indices_t, get_accessors<Tup>, get_accessors<Tups>...>>;
                        using generators_t = meta::transform<get_transform_index_generator, indices_t>;
                        return generate_f<generators_t, res_t>()(
                            m_fun, std::forward<Tup>(tup), std::forward<Tups>(tups)...);
                    }
                };

                template <class Fun>
                struct add_index_arg_f {
                    Fun m_fun;

                    template <size_t I, class... Args>
                    GT_TARGET GT_FORCE_INLINE constexpr auto operator()(Args &&...args) const
                        -> decltype(m_fun(std::forward<Args>(args)...)) {
                        return m_fun(std::forward<Args>(args)...);
                    }
                };

                template <class Fun>
                using transform_f = transform_index_f<add_index_arg_f<Fun>>;

                template <class Fun>
                struct for_each_adaptor_f {
                    Fun m_fun;
                    template <size_t I, class... Args>
                    GT_TARGET GT_FORCE_INLINE constexpr meta::lazy::id<void> operator()(Args &&...args) const {
                        m_fun(std::forward<Args>(args)...);
                        return {};
                    }
                };

                template <class Fun>
                struct for_each_index_adaptor_f {
                    Fun m_fun;
                    template <size_t I, class... Args>
                    GT_TARGET GT_FORCE_INLINE constexpr meta::lazy::id<void> operator()(Args &&...args) const {
                        m_fun.template operator()<I>(std::forward<Args>(args)...);
                        return {};
                    }
                };

                template <class Indices>
                struct apply_to_elements_f;

                template <template <class...> class L, class... Is>
                struct apply_to_elements_f<L<Is...>> {
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Fun &&fun, Tups &&...tups) const {
                        return std::forward<Fun>(fun)(
                            GT_TARGET_NAMESPACE_NAME::get<Is::value>(std::forward<Tups>(tups))...);
                    }
                };

                template <class>
                struct for_each_in_cartesian_product_impl_f;

                template <template <class...> class Outer, class... Inners>
                struct for_each_in_cartesian_product_impl_f<Outer<Inners...>> {
                    template <class Fun, class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr void operator()(Fun &&fun, Tups &&...tups) const {
                        using loop_t = int[sizeof...(Inners)];
                        void(loop_t{(
                            apply_to_elements_f<Inners>{}(std::forward<Fun>(fun), std::forward<Tups>(tups)...), 0)...});
                    }
                };

                template <class Fun>
                struct for_each_in_cartesian_product_f {
                    Fun m_fun;
                    template <class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr void operator()(Tups &&...tups) const {
                        for_each_in_cartesian_product_impl_f<
                            meta::cartesian_product<meta::make_indices_c<size<std::decay_t<Tups>>::value>...>>{}(
                            m_fun, std::forward<Tups>(tups)...);
                    }
                };

                struct skip_me {
                    template <class T>
                    GT_TARGET GT_FORCE_INLINE constexpr skip_me(T &&) {}
                };

                template <size_t>
                using skip_me_type = skip_me;

                template <class IndicesToSkip>
                struct select_arg_f;

                template <size_t... IndicesToSkip>
                struct select_arg_f<std::index_sequence<IndicesToSkip...>> {
                    template <class T, class... Ts>
                    GT_TARGET GT_FORCE_INLINE constexpr T &&operator()(
                        skip_me_type<IndicesToSkip> &&..., T &&obj, Ts &&...) const {
                        return std::forward<T>(obj);
                    }
                };

                template <class ResultMaker>
                struct concat_f {
                    template <class OuterI, class InnerI>
                    struct generator_f {
                        template <class... Tups>
                        GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Tups &&...tups) const {
                            return GT_TARGET_NAMESPACE_NAME::get<InnerI::value>(
                                select_arg_f<std::make_index_sequence<OuterI::value>>{}(std::forward<Tups>(tups)...));
                        }
                    };

                    template <class OuterI, class InnerTup>
                    using get_inner_generators = meta::transform<meta::curry<generator_f, OuterI>::template apply,
                        meta::make_indices_for<InnerTup>>;

                    template <class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr auto operator()(Tups &&...tups) const {
                        using accessors_t = meta::transform<get_accessors, meta::list<Tups...>>;
                        using res_t =
                            typename ResultMaker::template apply<meta::flatten<accessors_t>, meta::list<Tups...>>;
                        using generators_t = meta::flatten<
                            meta::transform<get_inner_generators, meta::make_indices_for<accessors_t>, accessors_t>>;
                        return generate_f<generators_t, res_t>{}(std::forward<Tups>(tups)...);
                    }
                };

                template <class ResultMaker>
                struct flatten_f {
                    template <class OuterI, class InnerI>
                    struct generator_f {
                        template <class Tup>
                        GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Tup &&tup) const {
                            return GT_TARGET_NAMESPACE_NAME::get<InnerI::value>(
                                GT_TARGET_NAMESPACE_NAME::get<OuterI::value>(std::forward<Tup>(tup)));
                        }
                    };

                    template <class OuterI, class InnerTup>
                    using get_inner_generators = meta::transform<meta::curry<generator_f, OuterI>::template apply,
                        meta::make_indices_for<InnerTup>>;

                    template <class Tup>
                    GT_TARGET GT_FORCE_INLINE constexpr auto operator()(Tup &&tup) const {
                        static_assert(size<Tup>::value != 0, "can not flatten empty tuple");
                        using accessors_t = meta::transform<get_accessors, get_accessors<Tup>>;
                        using res_t = typename ResultMaker::template apply<meta::flatten<accessors_t>, to_types<Tup>>;
                        using generators_t = meta::flatten<
                            meta::transform<get_inner_generators, meta::make_indices_for<accessors_t>, accessors_t>>;
                        return generate_f<generators_t, res_t>{}(std::forward<Tup>(tup));
                    }
                };

                template <size_t N>
                struct drop_front_f {
                    template <class I>
                    using get_drop_front_generator = get_nth_f<N + I::value>;

                    template <class Tup,
                        class Accessors = get_accessors<Tup>,
                        class Res = from_types<Tup, meta::drop_front_c<N, Accessors>>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        using generators =
                            meta::transform<get_drop_front_generator, meta::make_indices_c<size<Accessors>::value - N>>;
                        return generate_f<generators, Res>{}(std::forward<Tup>(tup));
                    }
                };

                template <class, class>
                struct push_back_impl_f;

                template <template <class T, T...> class L, class Int, Int... Is, class Res>
                struct push_back_impl_f<L<Int, Is...>, Res> {
                    template <class Tup, class... Args>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup, Args &&...args) const {
                        return Res{
                            GT_TARGET_NAMESPACE_NAME::get<Is>(std::forward<Tup>(tup))..., std::forward<Args>(args)...};
                    }
                };

                struct push_back_f {
                    template <class Tup,
                        class... Args,
                        class Accessors = get_accessors<Tup>,
                        class Res = from_types<Tup, meta::push_back<Accessors, Args &&...>>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup, Args &&...args) const {
                        return push_back_impl_f<std::make_index_sequence<size<Accessors>::value>, Res>{}(
                            std::forward<Tup>(tup), std::forward<Args>(args)...);
                    }
                };

                template <class, class>
                struct push_front_impl_f;

                template <template <class T, T...> class L, class Int, Int... Is, class Res>
                struct push_front_impl_f<L<Int, Is...>, Res> {
                    template <class Tup, class... Args>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup, Args &&...args) const {
                        return Res{
                            std::forward<Args>(args)..., GT_TARGET_NAMESPACE_NAME::get<Is>(std::forward<Tup>(tup))...};
                    }
                };

                struct push_front_f {
                    template <class Tup,
                        class... Args,
                        class Accessors = get_accessors<Tup>,
                        class Res = from_types<Tup, meta::push_front<Accessors, Args &&...>>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup, Args &&...args) const {
                        return push_front_impl_f<std::make_index_sequence<size<Accessors>::value>, Res>{}(
                            std::forward<Tup>(tup), std::forward<Args>(args)...);
                    }
                };

                template <class, class>
                struct pop_back_impl_f;

                template <template <class T, T...> class L, class Int, Int... Is, class Res>
                struct pop_back_impl_f<L<Int, Is...>, Res> {
                    template <class Tup>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        return Res{GT_TARGET_NAMESPACE_NAME::get<Is>(std::forward<Tup>(tup))...};
                    }
                };

                struct pop_back_f {
                    template <class Tup,
                        class Accessors = get_accessors<Tup>,
                        class Res = from_types<Tup, meta::pop_front<Accessors>>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        return pop_back_impl_f<std::make_index_sequence<size<Accessors>::value - 1>, Res>()(
                            std::forward<Tup>(tup));
                    }
                };

                template <class, class>
                struct pop_front_impl_f;

                template <template <class T, T...> class L, class Int, Int... Is, class Res>
                struct pop_front_impl_f<L<Int, Is...>, Res> {
                    template <class Tup>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        return Res{GT_TARGET_NAMESPACE_NAME::get<Is + 1>(std::forward<Tup>(tup))...};
                    }
                };

                struct pop_front_f {
                    template <class Tup,
                        class Accessors = get_accessors<Tup>,
                        class Res = from_types<Tup, meta::pop_front<Accessors>>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        return pop_front_impl_f<std::make_index_sequence<size<Accessors>::value - 1>, Res>()(
                            std::forward<Tup>(tup));
                    }
                };

                template <class Fun>
                struct fold_f {
                    template <class S, class T>
                    using meta_fun = typename get_fun_result<Fun>::template apply<S, T>;
                    Fun m_fun;

                    template <size_t I, size_t N, class State, class Tup, std::enable_if_t<I == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr State impl(State &&state, Tup &&) const {
                        return state;
                    }

                    template <size_t I, size_t N, class State, class Tup, std::enable_if_t<I + 1 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) impl(State &&state, Tup &&tup) const {
                        return m_fun(
                            std::forward<State>(state), GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tup>(tup)));
                    }

                    template <size_t I, size_t N, class State, class Tup, std::enable_if_t<I + 2 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) impl(State &&state, Tup &&tup) const {
                        return m_fun(
                            m_fun(std::forward<State>(state), GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tup>(tup))),
                            GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tup>(tup)));
                    }

                    template <size_t I, size_t N, class State, class Tup, std::enable_if_t<I + 3 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) impl(State &&state, Tup &&tup) const {
                        return m_fun(m_fun(m_fun(std::forward<State>(state),
                                               GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tup>(tup))),
                                         GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tup>(tup))),
                            GT_TARGET_NAMESPACE_NAME::get<I + 2>(std::forward<Tup>(tup)));
                    }

                    template <size_t I, size_t N, class State, class Tup, std::enable_if_t<I + 4 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) impl(State &&state, Tup &&tup) const {
                        return m_fun(m_fun(m_fun(m_fun(std::forward<State>(state),
                                                     GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tup>(tup))),
                                               GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tup>(tup))),
                                         GT_TARGET_NAMESPACE_NAME::get<I + 2>(std::forward<Tup>(tup))),
                            GT_TARGET_NAMESPACE_NAME::get<I + 3>(std::forward<Tup>(tup)));
                    }

                    template <size_t I,
                        size_t N,
                        class State,
                        class Tup,
                        class AllAccessors = get_accessors<Tup>,
                        class Accessors = meta::drop_front_c<I, AllAccessors>,
                        class Res = meta::foldl<meta_fun, State, Accessors>,
                        std::enable_if_t<(I + 4 < N), int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr Res impl(State &&state, Tup &&tup) const {
                        return impl<I + 5, N>(
                            m_fun(m_fun(m_fun(m_fun(m_fun(std::forward<State>(state),
                                                        GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tup>(tup))),
                                                  GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tup>(tup))),
                                            GT_TARGET_NAMESPACE_NAME::get<I + 2>(std::forward<Tup>(tup))),
                                      GT_TARGET_NAMESPACE_NAME::get<I + 3>(std::forward<Tup>(tup))),
                                GT_TARGET_NAMESPACE_NAME::get<I + 4>(std::forward<Tup>(tup))),
                            std::forward<Tup>(tup));
                    }

                    template <class State,
                        class Tup,
                        class Accessors = get_accessors<Tup>,
                        class Res = meta::foldl<meta_fun, State, Accessors>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(State &&state, Tup &&tup) const {
                        return impl<0, size<std::decay_t<Tup>>::value>(
                            std::forward<State>(state), std::forward<Tup>(tup));
                    }

                    template <class Tup,
                        class AllAccessors = get_accessors<Tup>,
                        class StateAccessor = meta::first<AllAccessors>,
                        class Accessors = meta::drop_front_c<1, AllAccessors>,
                        class Res = meta::foldl<meta_fun, StateAccessor, Accessors>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        return impl<1, size<std::decay_t<Tup>>::value>(
                            GT_TARGET_NAMESPACE_NAME::get<0>(std::forward<Tup>(tup)), std::forward<Tup>(tup));
                    }
                };

                template <class Fun>
                struct all_of_f {
                    Fun m_fun;

                    template <size_t I, size_t N, class... Tups, std::enable_if_t<I == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr bool impl(Tups &&...) const {
                        return true;
                    }

                    template <size_t I, size_t N, class... Tups, std::enable_if_t<I + 1 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr bool impl(Tups &&...tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, std::enable_if_t<I + 2 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr bool impl(Tups &&...tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, std::enable_if_t<I + 3 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr bool impl(Tups &&...tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 2>(std::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, std::enable_if_t<I + 4 == N, int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr bool impl(Tups &&...tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 2>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 3>(std::forward<Tups>(tups))...);
                    }

                    template <size_t I, size_t N, class... Tups, std::enable_if_t<(I + 4 < N), int> = 0>
                    GT_TARGET GT_FORCE_INLINE constexpr bool impl(Tups &&...tups) const {
                        return m_fun(GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 1>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 2>(std::forward<Tups>(tups))...) &&
                               m_fun(GT_TARGET_NAMESPACE_NAME::get<I + 3>(std::forward<Tups>(tups))...) &&
                               impl<I + 4, N>(std::forward<Tups>(tups)...);
                    }

                    template <class Tup, class... Tups>
                    GT_TARGET GT_FORCE_INLINE constexpr bool operator()(Tup &&tup, Tups &&...tups) const {
                        return impl<0, size<std::decay_t<Tup>>::value>(
                            std::forward<Tup>(tup), std::forward<Tups>(tups)...);
                    }
                };

                template <class To, class Index>
                struct implicit_convert_to_f {
                    using type = implicit_convert_to_f;
                    template <class Tup>
                    GT_TARGET GT_FORCE_INLINE constexpr To operator()(Tup &&tup) const {
                        return GT_TARGET_NAMESPACE_NAME::get<Index::value>(std::forward<Tup>(tup));
                    }
                };

                template <class DstFromTypesMetaClass>
                struct convert_to_f {
                    template <class Tup,
                        class ToTypes = get_accessors<Tup>,
                        class Res = meta::rename<DstFromTypesMetaClass::template apply, ToTypes>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        using generators_t =
                            meta::transform<implicit_convert_to_f, to_types<Res>, meta::make_indices_for<ToTypes>>;
                        return generate_f<generators_t, Res>{}(std::forward<Tup>(tup));
                    }
                };

                struct transpose_f {
                    template <class Tup>
                    struct get_inner_tuple_f {
                        template <class Types>
                        using apply = from_types<Tup, Types>;
                    };

                    template <class I>
                    using get_generator = transform_f<get_nth_f<I::value>>;

                    template <class Tup>
                    GT_TARGET
                        GT_FORCE_INLINE constexpr std::enable_if_t<tuple_util::size<std::decay_t<Tup>>::value == 0>
                        operator()(Tup &&) const {
                        static_assert(tuple_util::size<std::decay_t<Tup>>::value,
                            "tuple_util::transpose input should not be empty");
                    }

                    template <class Tup,
                        class First = meta::first<to_types<Tup>>,
                        class Accessors = meta::transform<get_accessors, get_accessors<Tup>>,
                        class Types = meta::transpose<Accessors>,
                        class InnerTuples = meta::transform<get_inner_tuple_f<Tup>::template apply, Types>,
                        class Res = from_types<First, InnerTuples>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        using inner_indices_t = meta::make_indices_for<to_types<First>>;
                        using generators_t = meta::transform<get_generator, inner_indices_t>;
                        return generate_f<generators_t, Res>{}(std::forward<Tup>(tup));
                    }
                };

                struct reverse_f {
                    template <class N>
                    struct generator_f {
                        template <class I>
                        using apply = get_nth_f<N::value - 1 - I::value>;
                    };

                    template <class Tup,
                        class Accessors = get_accessors<Tup>,
                        class Res = from_types<Tup, meta::reverse<Accessors>>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        using n_t = size<std::decay_t<Tup>>;
                        using generators_t = meta::transform<generator_f<n_t>::template apply, meta::make_indices<n_t>>;
                        return generate_f<generators_t, Res>{}(std::forward<Tup>(tup));
                    }
                };

                template <size_t I>
                struct insert_tup_generator_f {
                    using type = insert_tup_generator_f;

                    template <class Tup, class Val>
                    GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) operator()(Tup &&tup, Val &&) const {
                        return GT_TARGET_NAMESPACE_NAME::get<I>(std::forward<Tup>(tup));
                    }
                };

                struct insert_val_generator_f {
                    using type = insert_val_generator_f;
                    template <class Tup, class Val>
                    GT_TARGET GT_FORCE_INLINE constexpr Val operator()(Tup &&, Val &&val) const {
                        return std::forward<Val>(val);
                    }
                };

                template <size_t N, class Val>
                struct insert_f {
                    Val m_val;

                    template <class I>
                    using get_generator =
                        meta::if_c <
                        I::value<N,
                            insert_tup_generator_f<I::value>,
                            meta::if_c<I::value == N, insert_val_generator_f, insert_tup_generator_f<I::value - 1>>>;

                    template <class Tup,
                        class Accessors = get_accessors<Tup>,
                        class Types = meta::insert_c<N, Accessors, Val>,
                        class Res = from_types<Tup, Types>>
                    GT_TARGET GT_FORCE_INLINE constexpr Res operator()(Tup &&tup) const {
                        using generators_t = meta::transform<get_generator, meta::make_indices_for<Types>>;
                        return generate_f<generators_t, Res>{}(std::forward<Tup>(tup), m_val);
                    }
                };
            } // namespace detail

            /**
             * @brief Transforms each tuple element by a function.
             *
             * Transformations with functions with more than one argument are supported by passing multiple tuples of
             * the same size.
             *
             * @tparam Fun Functor type.
             * @tparam Tup Optional tuple-like type.
             * @tparam Tups Optional Tuple-like types.
             *
             * @param fun Function that should be applied to all elements of the given tuple(s).
             * @param tup First tuple-like object, serves as first arguments to `fun` if given.
             * @param tups Further tuple-like objects, serve as additional arguments to `fun` if given.
             *
             * Example code:
             * @code
             * #include <functional>
             * using namespace std::placeholders;
             *
             * struct add {
             *     template < class A, class B >
             *     auto operator()(A a, B b) -> decltype(a + b) {
             *         return a + b;
             *     }
             * };
             *
             * // Unary function, like boost::fusion::transform
             * auto tup = std::tuple(1, 2, 3.5);
             * auto fun = std::bind(add{}, 2, _1);
             * auto res = transform(fun, tup);
             * // res == {3, 4, 5.5}
             *
             * // Binary function
             * auto tup2 = std::tuple(1.5, 3, 4.1);
             * auto res2 = transform(add{}, tup, tup2);
             * // res2 == {2.5, 5, 7.6}
             * @endcode
             */
            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE constexpr auto transform(Fun && fun, Tup && tup, Tups && ...tups) {
                return detail::transform_f<Fun>{{std::forward<Fun>(fun)}}(
                    std::forward<Tup>(tup), std::forward<Tups>(tups)...);
            }

            /**
             * @brief Returns a functor that transforms each tuple element by a function.
             *
             * Composable version of `transform` that returns a functor which can be invoked with (one or multiple)
             * tuples.
             *
             * @tparam Fun Functor type.
             *
             * @param fun Function that should be applied to all elements of the given tuple(s).
             *
             * Example code:
             * @code
             * struct add {
             *     template < class A, class B >
             *     auto operator()(A a, B b) -> decltype(a + b) {
             *         return a + b;
             *     }
             * };
             *
             * // Composable usage with only a function argument
             * auto addtuples = transform(add{});
             * // addtuples takes now two tuples as arguments
             * auto tup1 = std::tuple(1, 2, 3.5);
             * auto tup2 = std::tuple(1.5, 3, 4.1);
             * auto res = addtuples(tup1, tup2)
             * // res == {2.5, 5, 7.6}
             * @endcode
             */
            template <class Fun>
            GT_TARGET GT_FORCE_INLINE constexpr detail::transform_f<Fun> transform(Fun fun) {
                return {std::move(fun)};
            }

            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE constexpr auto transform_index(Fun && fun, Tup && tup, Tups && ...tups) {
                return detail::transform_index_f<Fun>{std::forward<Fun>(fun)}(
                    std::forward<Tup>(tup), std::forward<Tups>(tups)...);
            }

            template <class Fun>
            GT_TARGET GT_FORCE_INLINE constexpr detail::transform_index_f<Fun> transform_index(Fun fun) {
                return {std::move(fun)};
            }

            /**
             * @brief Calls a function for each element in a tuple.
             *
             * Functions with more than one argument are supported by passing multiple tuples of the same size. If only
             * a function but no tuples are passed, a composable functor is returned.
             *
             * @tparam Fun Functor type.
             * @tparam Tup Optional tuple-like type.
             * @tparam Tups Optional Tuple-like types.
             *
             * @param fun Function that should be called for each element of the given tuple(s).
             * @param tup First tuple-like object, serves as first arguments to `fun` if given.
             * @param tups Further tuple-like objects, serve as additional arguments to `fun` if given.
             *
             * Example code:
             * @code
             * struct sum {
             *     double& value;
             *     template < class A >
             *     void operator()(A a, bool mask = true) const {
             *         if (mask)
             *             value += a;
             *     }
             * };
             *
             * // Unary function, like boost::fusion::for_each
             * auto tup = std::tuple(1, 2, 3.5);
             * double sum_value = 0.0;
             * for_each(sum{sum_value}, tup);
             * // sum_value == 6.5
             *
             * // Binary function
             * auto tup2 = std::tuple(false, true, true);
             * sum_value = 0.0;
             * for_each(sum{sum_value}, tup, tup2);
             * // sum_value == 5.5
             * @endcode
             */
            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE constexpr void for_each(Fun && fun, Tup && tup, Tups && ...tups) {
                transform_index(detail::for_each_adaptor_f<Fun>{std::forward<Fun>(fun)},
                    std::forward<Tup>(tup),
                    std::forward<Tups>(tups)...);
            }

            /**
             * @brief Returns a functor that calls a function for each element in a tuple.
             *
             * Composable version of `for_each` that returns a functor which can be invoked with (one or multiple)
             * tuples.
             *
             * @tparam Fun Functor type.
             *
             * @param fun Function that should be called for each element of the given tuple(s).
             *
             * Example code:
             * @code
             * struct sum {
             *     double& value;
             *     template < class A >
             *     void operator()(A a, bool mask = true) const {
             *         if (mask)
             *             value += a;
             *     }
             * };
             *
             * // Composable usage with only a function argument
             * sum_value = 0.0;
             * auto sumtuples = for_each(sum{sum_value});
             * // sumtuples takes now two tuples as arguments
             * auto tup1 = std::tuple(1, 2, 3.5);
             * auto tup2 = std::tuple(false, true, true);
             * auto res = sumtuples(tup1, tup2)
             * // sum_value == 5.5
             * @endcode
             */
            template <class Fun>
            GT_TARGET GT_FORCE_INLINE constexpr detail::transform_index_f<detail::for_each_adaptor_f<Fun>> for_each(
                Fun fun) {
                return {{std::move(fun)}};
            }

            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE constexpr void for_each_index(Fun && fun, Tup && tup, Tups && ...tups) {
                transform_index(detail::for_each_index_adaptor_f<Fun>{std::forward<Fun>(fun)},
                    std::forward<Tup>(tup),
                    std::forward<Tups>(tups)...);
            }

            template <class Fun>
            GT_TARGET GT_FORCE_INLINE constexpr detail::transform_index_f<detail::for_each_index_adaptor_f<Fun>>
            for_each_index(Fun fun) {
                return {{std::move(fun)}};
            }

            /**
             * @brief Calls a function for each element in a cartesian product of the given tuples.
             *
             * If only a function but no tuples are passed, a composable functor is returned.
             *
             * @tparam Fun Functor type.
             * @tparam Tup Optional tuple-like type.
             * @tparam Tups Optional Tuple-like types.
             *
             * @param fun Function that should be called for each element in a cartesian product of the given tuples.
             * @param tup First tuple-like object, serves as first arguments to `fun` if given.
             * @param tups Further tuple-like objects, serve as additional arguments to `fun` if given.
             *
             * Example code:
             * @code
             * struct sum {
             *     double& value;
             *     template < class A, class B >
             *     void operator()(A a, B b) const {
             *         value += a * b;
             *     }
             * };
             *
             * // Binary function
             * sum_value = 0.;
             * for_each_in_cartesian_product(sum{sum_value}, std::tuple(1, 2, 3), std::tuple(1, 10));
             * // sum_value == 66.
             * @endcode
             */
            template <class Fun, class Tup, class... Tups>
            GT_TARGET GT_FORCE_INLINE constexpr void for_each_in_cartesian_product(
                Fun && fun, Tup && tup, Tups && ...tups) {
                detail::for_each_in_cartesian_product_f<Fun>{std::forward<Fun>(fun)}(
                    std::forward<Tup>(tup), std::forward<Tups>(tups)...);
            }

            template <class Fun>
            GT_TARGET GT_FORCE_INLINE constexpr detail::for_each_in_cartesian_product_f<Fun>
            for_each_in_cartesian_product(Fun fun) {
                return {std::move(fun)};
            }

            /*
             *  Concatenate several tuple-likes into one.
             *  The type of result is deduced from the first argument
             */
            DEFINE_FUNCTOR_INSTANCE(concat, detail::concat_f<_impl::default_concat_result_maker_f>);

            /*
             * Variation of concat that allows to specify the resulting type.
             *
             * ResultMaker is a meta class with the following signature:
             *   struct ResultMaker {
             *     template <class FlattenTypesList,  ArgumentsOfConcatList>
             *     using apply = ...;
             *   }
             */
            template <class ResultMaker>
            DEFINE_TEMPLATED_FUNCTOR_INSTANCE(concat_ex, detail::concat_f<ResultMaker>);

            /**
             * @brief Non-recursively flattens a tuple of tuples into a single tuple.
             *
             * Flattens only the first two levels of nested tuples into a single level. Does not flatten further levels
             * of nesting.
             *
             * @tparam Tup Tuple-like type.
             * @param tup Tuple-like object.
             *
             * Example:
             * @code
             * auto tup1 = std::tuple(1, 2);
             * auto tup2 = std::tuple(3, 4, 5);
             * auto flat = flatten(tup1, tup2);
             * // flat == {1, 2, 3, 4, 5}
             * @endcode
             */
            DEFINE_FUNCTOR_INSTANCE(flatten, detail::flatten_f<_impl::default_concat_result_maker_f>);

            /*
             * Extended variation of flatten. See concat_ex comments
             */
            template <class ResultMaker>
            DEFINE_TEMPLATED_FUNCTOR_INSTANCE(flatten_ex, detail::flatten_f<ResultMaker>);

            /**
             * @brief Constructs an object from generator functors.
             *
             * `Generators` is a typelist of generator functors. Instances of those types are first default constructed,
             * then invoked with `args` as arguments. The results of those calls are then passed to the constructor of
             * `Res`.
             *
             * @tparam Generators A typelist of functors. All functor types must be default-constructible and callable
             * with arguments of type `Args`.
             * @tparam Res The type that should be constructed.
             * @tparam Args Argument types for the generator functors.
             *
             * @param args Arguments that will be passed to the generator functors.
             *
             * Example:
             * @code
             * // Generators that extract some information from the given arguments (a single std::string in this
             * example) struct ptr_extractor { const char* operator()(std::string const& s) const { return s.data();
             *     }
             * };
             *
             * struct size_extractor {
             *     std::size_t operator()(std::string const& s) const {
             *         return s.size();
             *     }
             * };
             *
             * // We want to generate a pair of a pointer and size,
             * // that represents this string in a simple C-style manner
             * std::string s = "Hello World!";
             * // Target-type to construct
             * using ptr_size_pair = std::pair< const char*, std::size_t >;
             *
             * // Typelist of generators
             * using generators = std::tuple< ptr_extractor, size_extractor>;
             *
             * // Generate pair
             * auto p = generate< generators, ptr_size_pair >(s);
             * // p.first is now a pointer to the first character of s, p.second is the size of s
             * @endcode
             */
            template <class Generators, class Res, class... Args>
            GT_TARGET GT_FORCE_INLINE constexpr Res generate(Args && ...args) {
                return detail::generate_f<Generators, Res>{}(std::forward<Args>(args)...);
            }

            /**
             * @brief Removes the first `N` elements from a tuple.
             *
             * @tparam N Number of elements to remove.
             * @tparam Tup Tuple-like type.
             *
             * @param tup Tuple to remove first `N` elements from.
             *
             * Example:
             * @code
             * auto tup = std::tuple(1, 2, 3, 4);
             * auto res = drop_front<2>(tup);
             * // res == {3, 4}
             * @endcode
             */
            template <size_t N>
            DEFINE_TEMPLATED_FUNCTOR_INSTANCE(drop_front, detail::drop_front_f<N>);

            /**
             * @brief Appends elements to a tuple.
             *
             * @tparam Tup Tuple-like type.
             * @tparam Args Argument types to append.
             *
             * @param tup Tuple-like object.
             * @param args Arguments to append.
             *
             * Example:
             * @code
             * auto tup = std::tuple(1, 2);
             * auto res = push_back(tup, 3, 4);
             * // res = {1, 2, 3, 4}
             * @endcode
             */
            DEFINE_FUNCTOR_INSTANCE(push_back, detail::push_back_f);

            /**
             * @brief Appends elements to a tuple from the front.
             */
            DEFINE_FUNCTOR_INSTANCE(push_front, detail::push_front_f);

            /**
             * @brief Removes elements to a tuple from the front.
             */
            DEFINE_FUNCTOR_INSTANCE(pop_front, detail::pop_front_f);

            /**
             * @brief Removes elements to a tuple from the back.
             */
            DEFINE_FUNCTOR_INSTANCE(pop_back, detail::pop_back_f);

            /**
             * @brief Left fold on tuple-like objects.
             *
             * This function accepts either two or three arguments. If three arguments are given, the second is the
             * initial state and the third a tuple-like object to fold. If only two arguments are given, the second is a
             * tuple-like object where the first element acts as the initial state.
             *
             * @tparam Fun Binary function type.
             * @tparam Arg Either the initial state if three arguments are given or the tuple to fold if two arguments
             * are given.
             * @tparam Args The tuple type to fold (if three arguments are given).
             *
             * @param fun Binary function object.
             * @param arg Either the initial state if three arguments are given or the tuple to fold if two arguments
             * are given.
             * @param args The tuple to fold (if three arguments are given).
             *
             * Example:
             * @code
             * auto tup = std::tuple(1, 2, 3);
             *
             * // Three arguments
             * auto res = fold(std::plus<int>{}, 0, tup);
             * // res == 6
             *
             * // Two arguments
             * auto res2 = fold(std::plus<int>{}, tup);
             * // res2 == 6
             * @endcode
             */
            template <class Fun, class Arg, class... Args>
            GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) fold(Fun && fun, Arg && arg, Args && ...args) {
                return detail::fold_f<Fun>{std::forward<Fun>(fun)}(std::forward<Arg>(arg), std::forward<Args>(args)...);
            }

            /**
             * @brief Returns a functor that performs a left fold on tuple-like objects.
             *
             * The returned functor accepts either one or two arguments. If two arguments are given, the first is the
             * initial state and the second a tuple-like object to fold. If only one argument is given, the argument
             * must be a tuple-like object where the first element acts as the initial state.
             *
             * @tparam Fun Binary function type.
             * @param fun Binary function object.
             *
             * Example:
             * @code
             * auto tup = std::tuple(1, 2, 3);
             * auto folder = fold(std::plus<int>{});
             *
             * // Three arguments
             * auto res = folder(0, tup);
             * // res == 6
             *
             * // Two arguments
             * auto res2 = folder(tup);
             * // res2 == 6
             * @endcode
             */
            template <class Fun>
            GT_TARGET GT_FORCE_INLINE constexpr detail::fold_f<Fun> fold(Fun fun) {
                return {std::move(fun)};
            }

            template <class Pred, class... Tups>
            GT_TARGET GT_FORCE_INLINE constexpr bool all_of(Pred && pred, Tups && ...tups) {
                return detail::all_of_f<Pred>{std::forward<Pred>(pred)}(std::forward<Tups>(tups)...);
            }

            template <class Pred>
            GT_TARGET GT_FORCE_INLINE constexpr detail::all_of_f<Pred> all_of(Pred pred) {
                return {std::move(pred)};
            }

            /**
             * transposes a `tuple like` of `tuple like`.
             *
             * Ex.
             *   transpose(array{array{1, 2, 3}, array{10, 20, 30}}) returns the same as
             *   array{array{1, 10}, array{2, 20}, array{3, 30}};
             */
            DEFINE_FUNCTOR_INSTANCE(transpose, detail::transpose_f);

            /**
             * @brief Replaces reference types by value types in a tuple.
             *
             * @tparam Tup Tuple-like type.
             * @param tup Tuple-like object, possibly containing references.
             *
             * Example:
             * @code
             * int foo = 3;
             * std::tuple<int&> tup(foo);
             * auto tupcopy = deep_copy(tup);
             * ++foo;
             * // tup == {4}, tupcopy == {3}
             * @endcode
             */
            DEFINE_FUNCTOR_INSTANCE(deep_copy, detail::transform_f<gridtools::GT_TARGET_NAMESPACE_NAME::clone>);

            namespace detail {
                // in impl as it is not as powerful as std::invoke (does not support invoking member functions)
                template <class Fun, class... Args>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) invoke_impl(Fun &&f, Args &&...args) {
                    return std::forward<Fun>(f)(std::forward<Args>(args)...);
                }

                template <class Fun, class Tup, std::size_t... Is>
                GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) apply_impl(
                    Fun &&f, Tup &&tup, std::index_sequence<Is...>) {
                    return invoke_impl(std::forward<Fun>(f), get<Is>(std::forward<Tup>(tup))...);
                }
            } // namespace detail

            /**
             * @brief Invoke callable f with tuple of arguments.
             *
             * @tparam Fun Functor type.
             * @tparam Tup Tuple-like type.
             * @param tup Tuple-like object containing arguments
             * @param fun Function that should be called with the arguments in tup
             *
             * See std::apply (c++17), with the limitation that it only works for FunctionObjects (not for any Callable)
             */
            template <class Fun, class Tup>
            GT_TARGET GT_FORCE_INLINE constexpr decltype(auto) apply(Fun && fun, Tup && tup) {
                return detail::apply_impl(std::forward<Fun>(fun),
                    std::forward<Tup>(tup),
                    std::make_index_sequence<size<std::decay_t<Tup>>::value>{});
            }

            /// Generalization of `std::tie`
            //
            template <template <class...> class L, class... Ts>
            GT_TARGET GT_FORCE_INLINE constexpr L<Ts &...> tie(Ts & ...elems) {
                return L<Ts &...>{elems...};
            }

            /**
             *   The family of `convert_to` functions.
             *
             *   First template parameter could be either some tuple [`std::tuple`, `std::pair` or gridtools clones]
             *   or some array [`std::array` or gridtools clone]
             *   Array vaiants can take additional parameter -- the desired type of array. If it is not provided,
             *   the type is deduced
             *   Runtime parameter is any "tuple like" or none. Variants without runtime parameter return functors as
             *   usual.
             *
             *   Examples of valid appllcations:
             *
             *   convert_to<std::tuple>(some_tuple_like);
             *   convert_to<std::pair>()(some_tuple_like_with_two_elements);
             *   convert_to<std::array>(some_tuple_like);
             *   convert_to<gridtools::array, int>(some_tuple_like);
             */
            template <template <class...> class L
#if defined(__NVCC__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 11 || __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 8)
                // workaround against nvcc bug: https://godbolt.org/z/orrev1xnM
                ,
                int = 0,
                class = std::void_t<L<int, int>>
#endif
                >
            GT_TARGET GT_FORCE_INLINE constexpr detail::convert_to_f<_impl::to_tuple_converter_helper<L>> convert_to() {
                return {};
            }

            template <template <class...> class L
#if defined(__NVCC__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 11 || __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 8)

                // workaround against nvcc bug: https://godbolt.org/z/orrev1xnM
                ,
                int = 0,
                class = std::void_t<L<int, int>>
#endif
                ,
                class Tup>
            GT_TARGET GT_FORCE_INLINE constexpr auto convert_to(Tup && tup) {
                return detail::convert_to_f<_impl::to_tuple_converter_helper<L>>{}(std::forward<Tup>(tup));
            }

            template <template <class, size_t> class Arr, class D = void>
            GT_TARGET GT_FORCE_INLINE constexpr detail::convert_to_f<_impl::to_array_converter_helper<Arr, D>>
            convert_to() {
                return {};
            }

            template <template <class, size_t> class Arr, class D = void, class Tup>
            GT_TARGET GT_FORCE_INLINE constexpr auto convert_to(Tup && tup) {
                return (detail::convert_to_f<_impl::to_array_converter_helper<Arr, D>>{}(std::forward<Tup>(tup)));
            }

            DEFINE_FUNCTOR_INSTANCE(reverse, detail::reverse_f);

            template <size_t I, class Val>
            GT_TARGET GT_FORCE_INLINE constexpr detail::insert_f<I, Val> insert(Val && val) {
                return {std::forward<Val>(val)};
            }

            template <size_t I, class Val, class Tup>
            GT_TARGET GT_FORCE_INLINE constexpr auto insert(Val && val, Tup && tup) {
                return insert<I>(std::forward<Val>(val))(std::forward<Tup>(tup));
            }
        }
    } // namespace tuple_util
} // namespace gridtools

#undef DEFINE_TEMPLATED_FUNCTOR_INSTANCE
#undef DEFINE_FUNCTOR_INSTANCE

#endif // GT_TARGET_ITERATING

/** @} */
/** @} */
