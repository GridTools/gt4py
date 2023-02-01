
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

struct _fun_1 {
  constexpr auto operator()() const {
    return [](auto const &x) { return gtfn::deref(x); };
  }
};

inline auto copy_fencil = [](auto... connectivities__) {
  return [connectivities__...](auto backend, auto &&isize, auto &&jsize,
                               auto &&ksize, auto &&inp, auto &&out) {
    auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);

    make_backend(
        backend,
        gtfn::cartesian_domain(
            ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                (isize - 0_c), (jsize - 0_c), (ksize - 0_c)),
            ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                0_c, 0_c, 0_c)))
        .stencil_executor()()
        .arg(out)
        .arg(inp)
        .assign(0_c, _fun_1(), 1_c)
        .execute();
  };
};
} // namespace
} // namespace generated
