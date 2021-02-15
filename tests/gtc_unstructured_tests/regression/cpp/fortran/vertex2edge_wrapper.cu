#include "../generated_vertex2edge_gtir_ugpu.hpp"
#include <cpp_bindgen/export.hpp>
#include <gridtools/storage/adapter/fortran_array_view.hpp>
#include <gridtools/usid/test_helper/field_builder.hpp>
#include <gridtools/usid/test_helper/simple_mesh.hpp>

// #include <gtest/gtest.h>
#include <tuple>

namespace gridtools::usid {

using namespace gridtools::usid::test_helper;

// template <class Ptr> __global__ void print_kernel(Ptr ptr_holder) {
//  auto *ptr = ptr_holder();
//  printf("%f\n", ptr[0]);
//}

template <class T>
void run_stencil_impl(fortran_array_view<T, 1> in,
                      fortran_array_view<T, 1> out) {
  // auto *orig = sid::get_origin(in)();
  // std::cout << "data: " << *orig << std::endl;
  // print_kernel<<<1, 1>>>(sid::get_origin(in));
  cudaDeviceSynchronize();
  sten({-1, simple_mesh::edges, -1, 1}, simple_mesh{}.e2v())(in, out);
}

BINDGEN_EXPORT_GENERIC_BINDING_WRAPPED(2, run_stencil, run_stencil_impl,
                                       (double)(float));

} // namespace gridtools::usid
