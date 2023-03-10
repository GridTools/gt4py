
#include <cmath>
#include <cstdint>
#include <gridtools/fn/cartesian.hpp>

namespace generated {

namespace gtfn = ::gridtools::fn;

namespace {
using namespace ::gridtools::literals;

struct IDim_t {};
constexpr inline IDim_t IDim{};

struct JDim_t {};
constexpr inline JDim_t JDim{};

struct KDim_t {};
constexpr inline KDim_t KDim{};

using i_t = IDim_t;
constexpr inline i_t i{};

using j_t = JDim_t;
constexpr inline j_t j{};

struct _fun_1 {
  constexpr auto operator()() const {
    return [](auto const &inp) {
      return (([=](auto inp_) {
                return (gtfn::deref(gtfn::shift(inp_, i, -1_c)) -
                        gtfn::deref(inp_));
              }(gtfn::shift(inp, i, -1_c, i, 1_c)) -
               [=](auto inp_) {
                 return (gtfn::deref(gtfn::shift(inp_, i, -1_c)) -
                         gtfn::deref(inp_));
               }(gtfn::shift(inp, i, 1_c))) +
              ([=](auto inp_) {
                return (gtfn::deref(gtfn::shift(inp_, j, -1_c)) -
                        gtfn::deref(inp_));
              }(gtfn::shift(inp, j, -1_c, j, 1_c)) -
               [=](auto inp_) {
                 return (gtfn::deref(gtfn::shift(inp_, j, -1_c)) -
                         gtfn::deref(inp_));
               }(gtfn::shift(inp, j, 1_c))));
    };
  }
};

inline auto lap_fencil = [](auto... connectivities__) {
  return [connectivities__...](auto backend, auto &&i_size, auto &&j_size,
                               auto &&k_size, auto &&i_off, auto &&j_off,
                               auto &&k_off, auto &&out, auto &&inp) {
    auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);

    make_backend(
        backend,
        gtfn::cartesian_domain(
            ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                ((i_size + i_off) - i_off), ((j_size + j_off) - j_off),
                ((k_size + k_off) - k_off)),
            ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                i_off, j_off, k_off)))
        .stencil_executor()()
        .arg(out)
        .arg(inp)
        .assign(0_c, _fun_1(), 1_c)
        .execute();
  };
};
} // namespace
} // namespace generated
