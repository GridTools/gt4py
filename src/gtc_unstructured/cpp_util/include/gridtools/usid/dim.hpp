#pragma once

#include <gridtools/common/defs.hpp>
#include <gridtools/common/integral_constant.hpp>

namespace gridtools::usid::dim {
    using horizontal = integral_constant<int_t, 0>;
    using vertical = integral_constant<int_t, 1>;
    using neighbor = integral_constant<int_t, 1>;
    using sparse = integral_constant<int_t, 2>;

    using h = horizontal;
    using k = vertical;
    using n = neighbor;
    using s = sparse;
} // namespace gridtools::usid::dim
