#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include <gridtools/common/hymap.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

#include "dim.hpp"
#include "helpers.hpp"

namespace gridtools::usid::naive {
using traits_t = storage::cpu_ifirst;

inline auto make_allocator() {
  return sid::make_cached_allocator(&std::make_unique<char[]>);
}

template <class Kernel, class Size, class Sid, class... Sids>
void call_kernel(Size size, Sid &&fields, Sids &&...neighbor_fields) {
  sid::make_loop<dim::h>(size)(
      [params = std::make_tuple(std::make_pair(
           sid::get_origin(neighbor_fields)(),
           sid::get_stride<dim::h>(sid::get_strides(neighbor_fields)))...)](
          auto &ptr, auto const &strides) {
        std::apply(Kernel()(),
                   std::tuple_cat(std::forward_as_tuple(ptr, strides), params));
      })(sid::get_origin(fields)(), sid::get_strides(fields));
}

template <class Tag, class Ptr> decltype(auto) field(Ptr const &ptr) {
  return *at_key<Tag>(ptr);
}
} // namespace gridtools::usid::naive
