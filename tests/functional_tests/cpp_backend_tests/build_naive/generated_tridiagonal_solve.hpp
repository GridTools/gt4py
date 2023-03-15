
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

struct _scan_1 : gtfn::fwd {
  static constexpr GT_FUNCTION auto body() {
    return gtfn::scan_pass(
        [](auto const &state, auto const &a, auto const &b, auto const &c,
           auto const &d) {
          return [=](auto _cs_1) {
            return gtfn::make_tuple(
                (gtfn::deref(c) / _cs_1),
                ((gtfn::deref(d) -
                  (gtfn::deref(a) * gtfn::tuple_get(1_c, state))) /
                 _cs_1));
          }((gtfn::deref(b) - (gtfn::deref(a) * gtfn::tuple_get(0_c, state))));
        },
        ::gridtools::host_device::identity());
  }
};

struct _scan_2 : gtfn::bwd {
  static constexpr GT_FUNCTION auto body() {
    return gtfn::scan_pass(
        [](auto const &x_kp1, auto const &cpdp) {
          return [=](auto _cs_2) {
            return (gtfn::tuple_get(1_c, _cs_2) -
                    (gtfn::tuple_get(0_c, _cs_2) * x_kp1));
          }(gtfn::deref(cpdp));
        },
        ::gridtools::host_device::identity());
  }
};

inline auto tridiagonal_solve_fencil = [](auto... connectivities__) {
  return [connectivities__...](auto backend, auto &&isize, auto &&jsize,
                               auto &&ksize, auto &&a, auto &&b, auto &&c,
                               auto &&d, auto &&x) {
    auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);
    auto _gtmp_0 =
        gtfn::allocate_global_tmp<::gridtools::tuple<double, double>>(
            tmp_alloc__,
            gtfn::cartesian_domain(
                ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                    (isize - 0_c), (jsize - 0_c), (ksize - 0_c)),
                ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                    0_c, 0_c, 0_c))
                .sizes());
    make_backend(
        backend,
        gtfn::cartesian_domain(
            ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                (isize - 0_c), (jsize - 0_c), (ksize - 0_c)),
            ::gridtools::hymap::keys<IDim_t, JDim_t, KDim_t>::make_values(
                0_c, 0_c, 0_c)))
        .vertical_executor(KDim)()
        .arg(_gtmp_0)
        .arg(a)
        .arg(b)
        .arg(c)
        .arg(d)
        .arg(x)
        .assign(0_c, _scan_1(), gtfn::make_tuple(0.0, 0.0), 1_c, 2_c, 3_c, 4_c)
        .assign(5_c, _scan_2(), 0.0, 0_c)
        .execute();
  };
};
} // namespace
} // namespace generated
