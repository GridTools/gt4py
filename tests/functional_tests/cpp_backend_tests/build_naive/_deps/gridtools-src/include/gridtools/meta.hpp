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

/**
 *  C++17 metaprogramming library.
 *
 *  Basic Concepts
 *  ==============
 *
 *  List
 *  ----
 *  An instantiation of the template class with class template parameters.
 *
 *  Examples of lists:
 *    meta::list<void, int> : elements are void and int
 *    std::tuple<double, double> : elements are double and double
 *    std::vector<std::tuple<>, some_allocator>: elements are std::tuple<> and some_allocator
 *
 *  Examples of non lists:
 *    std::array<N, double> : first template argument is not a class
 *    int : is not the instantiation of template
 *    struct foo; is not an instantiation of template
 *
 *  Function
 *  --------
 *  A template class or an alias with class template parameters.
 *  Note the difference with MPL approach: function is not required to have `type` inner alias.
 *  Functions that have `type` inside are called lazy functions in the context of this library.
 *  The function arguments are the actual parameters of the instantiation: Arg1, Arg2 etc. in F<Arg1, Arg2 etc.>
 *  The function invocation result is just F<Arg1, Arg2 etc.> not F<Arg1, Arg2 etc.>::type.
 *  This simplification of the function concepts (comparing with MPL) is possible because of C++ aliases.
 *  And it is significant for compile time performance.
 *
 *  Examples of functions:
 *    - std::is_same
 *    - std::pair
 *    - std::tuple
 *    - meta::list
 *    - meta::is_list
 *
 *  Examples of non functions:
 *    - std::array : first parameter is not a class
 *    - meta::list<int> : is not a template
 *
 *  In the library some functions have integers as arguments. Usually they have `_c` suffix and have the sibling
 *  without prefix. Disadvantage of having such a hybrid signature, that those functions can not be passed as
 *  arguments to high order functions.
 *
 *  Meta Class
 *  ----------
 *  A class that have `apply` inner template class or alias, which is a function [here and below the term `function`
 *  used in the context of this library]. Meta classes are used to return functions from functions.
 *
 *  Examples:
 *    - meta::always<void>
 *    - meta::rename<std::tuple>
 *
 *  High Order Function
 *  -------------------
 *  A template class or alias which first parameters are template of class class templates and the rest are classes
 *  Examples of metafuction signatures:
 *  template <template <class...> class, class...> struct foo;
 *  template <template <class...> class, template <class...> class> struct bar;
 *  template <template <class...> class...> struct baz;
 *
 *  Examples:
 *    - meta::rename
 *    - meta::foldl
 *    - meta::is_instantiation_of
 *
 *  Library Structure
 *  =================
 *
 *  It consists of the set of functions, `_c` functions and high order functions.
 *
 *  Regularly, a function has also its lazy version, which is defined in the `lazy` nested namespace under the same
 *  name. Exceptions are functions that return:
 *   - a struct with a nested `type` alias, which points to the struct itself;
 *       ex: `list`
 *   - a struct derived from `std::intergral_constant`
 *       ex: `length`, `is_list`
 *   - meta class
 *
 *  Syntax sugar: All high order functions being called with only functional arguments return partially applied
 *  versions of themselves [which became plane functions].
 *  Example, where it could be useful is:
 *  transform a list of lists:  <tt>using out = meta::transform<meta::transform<fun>::apply, in>;</tt>
 *
 *  TODO List
 *  =========
 *   - add numeric stuff like `plus`, `less` etc.
 */

#include "meta/always.hpp"
#include "meta/at.hpp"
#include "meta/bind.hpp"
#include "meta/cartesian_product.hpp"
#include "meta/clear.hpp"
#include "meta/combine.hpp"
#include "meta/concat.hpp"
#include "meta/ctor.hpp"
#include "meta/curry.hpp"
#include "meta/curry_fun.hpp"
#include "meta/debug.hpp"
#include "meta/dedup.hpp"
#include "meta/defer.hpp"
#include "meta/drop_back.hpp"
#include "meta/drop_front.hpp"
#include "meta/filter.hpp"
#include "meta/find.hpp"
#include "meta/first.hpp"
#include "meta/flatten.hpp"
#include "meta/fold.hpp"
#include "meta/force.hpp"
#include "meta/group.hpp"
#include "meta/has_type.hpp"
#include "meta/id.hpp"
#include "meta/if.hpp"
#include "meta/insert.hpp"
#include "meta/is_empty.hpp"
#include "meta/is_instantiation_of.hpp"
#include "meta/is_list.hpp"
#include "meta/is_set.hpp"
#include "meta/iseq_to_list.hpp"
#include "meta/last.hpp"
#include "meta/length.hpp"
#include "meta/list.hpp"
#include "meta/list_to_iseq.hpp"
#include "meta/logical.hpp"
#include "meta/macros.hpp"
#include "meta/make_indices.hpp"
#include "meta/mp_find.hpp"
#include "meta/mp_insert.hpp"
#include "meta/mp_inverse.hpp"
#include "meta/mp_make.hpp"
#include "meta/mp_remove.hpp"
#include "meta/not.hpp"
#include "meta/pop_back.hpp"
#include "meta/pop_front.hpp"
#include "meta/push_back.hpp"
#include "meta/push_front.hpp"
#include "meta/rename.hpp"
#include "meta/repeat.hpp"
#include "meta/replace.hpp"
#include "meta/reverse.hpp"
#include "meta/second.hpp"
#include "meta/st_contains.hpp"
#include "meta/st_position.hpp"
#include "meta/take.hpp"
#include "meta/third.hpp"
#include "meta/transform.hpp"
#include "meta/trim.hpp"
#include "meta/val.hpp"
#include "meta/zip.hpp"
