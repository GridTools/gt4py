/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/meta.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

namespace gridtools {
    namespace meta {
        template <class...>
        struct f;
        template <class...>
        struct g;
        template <class...>
        struct h;

        // is_list
        static_assert(!is_list<int>{});
        static_assert(is_list<list<int, void>>{});
        static_assert(is_list<f<void, int>>{});
        static_assert(is_list<std::pair<int, double>>{});
        static_assert(is_list<std::tuple<int, double>>{});

        // has_type
        static_assert(!has_type<int>{});
        static_assert(!has_type<g<>>{});
        static_assert(has_type<lazy::id<void>>{});
        static_assert(has_type<std::is_void<int>>{});

        // length
        static_assert(length<list<>>::value == 0);
        static_assert(length<std::tuple<int>>::value == 1);
        static_assert(length<std::pair<int, double>>::value == 2);
        static_assert(length<f<int, int, double>>::value == 3);

        // ctor
        static_assert(std::is_same_v<ctor<f<double>>::apply<int, void>, f<int, void>>);

        // rename
        static_assert(std::is_same_v<rename<f, g<int, double>>, f<int, double>>);

        // transform
        static_assert(std::is_same_v<transform<f, g<>>, g<>>);
        static_assert(std::is_same_v<transform<f, g<int, void>>, g<f<int>, f<void>>>);
        static_assert(std::is_same_v<transform<f, g<int, void>, g<int *, void *>, g<int **, void **>>,
            g<f<int, int *, int **>, f<void, void *, void **>>>);

        // st_contains
        static_assert(st_contains<g<int, bool>, int>{});
        static_assert(!st_contains<g<int, bool>, void>{});

        // mp_find
        using map = f<g<int, void *>, g<void, double *>, g<float, double *>>;
        static_assert(std::is_same_v<mp_find<map, int>, g<int, void *>>);
        static_assert(std::is_same_v<mp_find<map, double>, void>);

        // repeat
        static_assert(std::is_same_v<repeat_c<0, f<int>>, f<>>);
        static_assert(std::is_same_v<repeat_c<3, f<int>>, f<int, int, int>>);

        // drop_front
        static_assert(std::is_same_v<drop_front_c<0, f<int, double>>, f<int, double>>);
        static_assert(std::is_same_v<drop_front_c<1, f<int, double>>, f<double>>);
        static_assert(std::is_same_v<drop_front_c<2, f<int, double>>, f<>>);

        // at
        static_assert(std::is_same_v<at_c<f<int, double>, 0>, int>);
        static_assert(std::is_same_v<at_c<f<int, double>, 1>, double>);
        static_assert(std::is_same_v<last<f<int, double>>, double>);

        // conjunction
        static_assert(conjunction_fast<>{});
        static_assert(conjunction_fast<std::true_type, std::true_type>{});
        static_assert(!conjunction_fast<std::true_type, std::false_type>{});
        static_assert(!conjunction_fast<std::false_type, std::true_type>{});
        static_assert(!conjunction_fast<std::false_type, std::false_type>{});

        // disjunction
        static_assert(!disjunction_fast<>{});
        static_assert(disjunction_fast<std::true_type, std::true_type>{});
        static_assert(disjunction_fast<std::true_type, std::false_type>{});
        static_assert(disjunction_fast<std::false_type, std::true_type>{});
        static_assert(!disjunction_fast<std::false_type, std::false_type>{});

        // st_position
        static_assert(st_position<f<int, double>, int>{} == 0);
        static_assert(st_position<f<double, int>, int>{} == 1);
        static_assert(st_position<f<double, int>, void>{} == 2);

        // combine
        static_assert(std::is_same_v<combine<f, g<int>>, int>);
        static_assert(std::is_same_v<combine<f, repeat_c<8, g<int>>>,
            f<f<f<int, int>, f<int, int>>, f<f<int, int>, f<int, int>>>>);
        static_assert(std::is_same_v<combine<f, g<int, int, int>>, f<int, f<int, int>>>);
        static_assert(std::is_same_v<combine<f, repeat_c<4, g<int>>>, f<f<int, int>, f<int, int>>>);

        // concat
        static_assert(std::is_same_v<concat<g<int>>, g<int>>);
        static_assert(std::is_same_v<concat<g<int>, f<void>>, g<int, void>>);
        static_assert(std::is_same_v<concat<g<int>, g<void, double>, g<void, int>>, g<int, void, double, void, int>>);

        // flatten
        static_assert(std::is_same_v<flatten<f<g<int>>>, g<int>>);
        static_assert(std::is_same_v<flatten<f<g<int>, f<bool>>>, g<int, bool>>);

        // filter
        static_assert(std::is_same_v<filter<std::is_pointer, f<>>, f<>>);
        static_assert(std::is_same_v<filter<std::is_pointer, f<void, int *, double, double **>>, f<int *, double **>>);

        // all_of
        static_assert(all_of<is_list, f<f<>, f<int>>>{});

        // dedup
        static_assert(std::is_same_v<dedup<f<>>, f<>>);
        static_assert(std::is_same_v<dedup<f<int>>, f<int>>);
        static_assert(std::is_same_v<dedup<f<int, void>>, f<int, void>>);
        static_assert(std::is_same_v<dedup<f<int, void, void, void, int, void>>, f<int, void>>);

        // zip
        static_assert(std::is_same_v<zip<f<int>, f<void>>, f<list<int, void>>>);
        static_assert(std::is_same_v<zip<f<int, int *, int **>, f<void, void *, void **>, f<char, char *, char **>>,
            f<list<int, void, char>, list<int *, void *, char *>, list<int **, void **, char **>>>);

        // bind
        static_assert(std::is_same_v<bind<f, _2, void, _1>::apply<int, double>, f<double, void, int>>);

        // is_instantiation_of
        static_assert(is_instantiation_of<f, f<>>{});
        static_assert(is_instantiation_of<f, f<int, void>>{});
        static_assert(!is_instantiation_of<f, g<>>{});
        static_assert(!is_instantiation_of<f, int>{});
        static_assert(is_instantiation_of<f>::apply<f<int, void>>{});

        static_assert(std::is_same_v<replace<f<int, double, int, double>, double, void>, f<int, void, int, void>>);

        static_assert(std::is_same_v<mp_replace<f<g<int, int *>, g<double, double *>>, int, void>,
            f<g<int, void>, g<double, double *>>>);

        static_assert(std::is_same_v<replace_at_c<f<int, double, int, double>, 1, void>, f<int, void, int, double>>);

        namespace nvcc_sizeof_workaround {
            template <class...>
            struct a;

            template <int I>
            struct b {
                using c = void;
            };

            template <class... Ts>
            using d = b<GT_SIZEOF_3_DOTS(Ts)>;

            template <class... Ts>
            using e = typename d<a<Ts>...>::c;
        } // namespace nvcc_sizeof_workaround

        static_assert(is_set<f<>>{});
        static_assert(is_set<f<int>>{});
        static_assert(is_set<f<void>>{});
        static_assert(is_set<f<int, void>>{});
        static_assert(!is_set<int>{});
        static_assert(!is_set<f<int, void, int>>{});

        static_assert(is_set_fast<f<>>{});
        static_assert(is_set_fast<f<int>>{});
        static_assert(is_set_fast<f<void>>{});
        static_assert(is_set_fast<f<int, void>>{});
        static_assert(!is_set_fast<int>{});
        //        static_assert(!is_set_fast< f< int, void, int > >{});

        // rfold
        static_assert(std::is_same_v<foldl<f, int, g<>>, int>);
        static_assert(std::is_same_v<foldl<f, int, g<int>>, f<int, int>>);
        static_assert(std::is_same_v<foldl<f, int, g<int, int>>, f<f<int, int>, int>>);
        static_assert(std::is_same_v<foldl<f, int, g<int, int>>, f<f<int, int>, int>>);
        static_assert(std::is_same_v<foldl<f, int, g<int, int, int>>, f<f<f<int, int>, int>, int>>);
        static_assert(std::is_same_v<foldl<f, int, g<int, int, int, int>>, f<f<f<f<int, int>, int>, int>, int>>);
        static_assert(
            std::is_same_v<foldl<f, int, g<int, int, int, int, int>>, f<f<f<f<f<int, int>, int>, int>, int>, int>>);
        static_assert(std::is_same_v<foldl<f, int, g<int, int, int, int, int, int>>,
            f<f<f<f<f<f<int, int>, int>, int>, int>, int>, int>>);

        // foldr
        static_assert(std::is_same_v<foldr<f, int, g<>>, int>);
        static_assert(std::is_same_v<foldr<f, int, g<int>>, f<int, int>>);
        static_assert(std::is_same_v<foldr<f, int, g<int, int>>, f<f<int, int>, int>>);
        static_assert(std::is_same_v<foldr<f, int, g<int, int, int>>, f<f<f<int, int>, int>, int>>);
        static_assert(std::is_same_v<foldr<f, int, g<int, int, int, int>>, f<f<f<f<int, int>, int>, int>, int>>);
        static_assert(
            std::is_same_v<foldr<f, int, g<int, int, int, int, int>>, f<f<f<f<f<int, int>, int>, int>, int>, int>>);
        static_assert(std::is_same_v<foldr<f, int, g<int, int, int, int, int, int>>,
            f<f<f<f<f<f<int, int>, int>, int>, int>, int>, int>>);

        static_assert(std::is_same_v<cartesian_product<>, list<list<>>>);
        static_assert(std::is_same_v<cartesian_product<f<>>, list<>>);
        static_assert(std::is_same_v<cartesian_product<f<int>>, list<list<int>>>);
        static_assert(std::is_same_v<cartesian_product<f<int, double>>, list<list<int>, list<double>>>);
        static_assert(
            std::is_same_v<cartesian_product<f<int, double>, g<void>>, list<list<int, void>, list<double, void>>>);
        static_assert(std::is_same_v<cartesian_product<f<int, double>, g<int *, double *>>,
            list<list<int, int *>, list<int, double *>, list<double, int *>, list<double, double *>>>);
        static_assert(std::is_same_v<cartesian_product<f<int, double>, g<>, f<void>>, list<>>);
        static_assert(std::is_same_v<cartesian_product<f<>, g<int, double>>, list<>>);
        static_assert(std::is_same_v<cartesian_product<f<int>, g<double>, list<void>>, list<list<int, double, void>>>);

        static_assert(std::is_same_v<reverse<f<int, int *, int **, int ***, int ****, int *****, int ******>>,
            f<int ******, int *****, int ****, int ***, int **, int *, int>>);

        static_assert(find<f<>, int>::type::value == 0);
        static_assert(find<f<void>, int>::type::value == 1);
        static_assert(find<f<double, int, int, double, int>, int>::type::value == 1);
        static_assert(find<f<double, int, int, double, int>, void>::type::value == 5);

        static_assert(std::is_same_v<mp_insert<f<>, g<int, int *>>, f<g<int, int *>>>);
        static_assert(std::is_same_v<mp_insert<f<g<void, void *>>, g<int, int *>>, f<g<void, void *>, g<int, int *>>>);
        static_assert(std::is_same_v<mp_insert<f<g<int, int *>>, g<int, int **>>, f<g<int, int *, int **>>>);

        static_assert(std::is_same_v<mp_remove<f<g<int, int *>>, void>, f<g<int, int *>>>);
        static_assert(std::is_same_v<mp_remove<f<g<int, int *>>, int>, f<>>);
        static_assert(std::is_same_v<mp_remove<f<g<int, int *>, g<void, void *>>, int>, f<g<void, void *>>>);

        static_assert(std::is_same_v<mp_inverse<f<>>, f<>>);
        static_assert(std::is_same_v<mp_inverse<f<g<int, int *>, g<void, void *>>>, f<g<int *, int>, g<void *, void>>>);
        static_assert(std::is_same_v<mp_inverse<f<g<int, int *, int **>, g<void, void *>>>,
            f<g<int *, int>, g<int **, int>, g<void *, void>>>);
        static_assert(std::is_same_v<mp_inverse<f<g<int *, int>, g<int **, int>, g<void *, void>>>,
            f<g<int, int *, int **>, g<void, void *>>>);

        // take
        static_assert(std::is_same_v<take_c<2, f<int, double, void, void>>, f<int, double>>);
        static_assert(std::is_same_v<take_c<20, repeat_c<100, g<int>>>, repeat_c<20, g<int>>>);

        // insert
        static_assert(std::is_same_v<insert_c<3, f<void, void, void, void, void>, int, double>,
            f<void, void, void, int, double, void, void>>);

        // group
        static_assert(std::is_same_v<group<are_same, g, f<>>, f<>>);
        static_assert(std::is_same_v<group<are_same, g, f<int>>, f<g<int>>>);
        static_assert(std::is_same_v<group<are_same, g, f<int, int, int, double, void, void, int, int>>,
            f<g<int, int, int>, g<double>, g<void, void>, g<int, int>>>);

        // trim
        static_assert(std::is_same_v<trim<std::is_void, f<int, void, int>>, f<int, void, int>>);
        static_assert(std::is_same_v<trim<std::is_void, f<>>, f<>>);
        static_assert(std::is_same_v<trim<std::is_void, f<void, void>>, f<>>);
        static_assert(
            std::is_same_v<trim<std::is_void, f<void, void, int, int, void, int, void>>, f<int, int, void, int>>);

        // mp_make
        static_assert(std::is_same_v<mp_make<h, f<>>, f<>>);
        static_assert(std::is_same_v<mp_make<h, f<g<int, int *>>>, f<h<g<int, int *>>>>);
        static_assert(
            std::is_same_v<mp_make<h, f<g<int, int *>, g<void, void *>>>, f<h<g<int, int *>>, h<g<void, void *>>>>);
        static_assert(
            std::is_same_v<mp_make<h, f<g<int, int *>, g<int, int **>>>, f<h<g<int, int *>, g<int, int **>>>>);
        static_assert(
            std::is_same_v<mp_make<h, f<g<void, void *>, g<int, int *>, g<int, int **>, g<double, double **>>>,
                f<h<g<void, void *>>, h<g<int, int *>, g<int, int **>>, h<g<double, double **>>>>);
    } // namespace meta
} // namespace gridtools
