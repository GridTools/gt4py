/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/boundaries/boundary.hpp>
#include <gridtools/boundaries/copy.hpp>
#include <gridtools/boundaries/value.hpp>
#include <gridtools/boundaries/zero.hpp>
#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/storage/builder.hpp>

#include <gcl_select.hpp>

using namespace gridtools;
using namespace boundaries;

struct bc_basic {
    template <typename Direction, typename DataField0>
    GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = i + j + k;
    }
};

struct bc_two {
    template <typename Direction, typename DataField0>
    GT_FUNCTION void operator()(Direction, DataField0 &data_field0, uint_t i, uint_t j, uint_t k) const {
        data_field0(i, j, k) = 0;
    }

    template <sign I, sign J, sign K, typename DataField0>
    GT_FUNCTION void operator()(direction<I, J, K>,
        DataField0 &data_field0,
        uint_t i,
        uint_t j,
        uint_t k,
        std::enable_if_t<J == minus_ || K == minus_> * = nullptr) const {
        data_field0(i, j, k) = (i + j + k + 1);
    }
};

struct minus_predicate {
    template <sign I, sign J, sign K>
    bool operator()(direction<I, J, K>) const {
        return !(I == minus_ || J == minus_ || K == minus_);
    }
};

auto make_storage(uint_t d1, uint_t d2, uint_t d3, int_t value = 0) {
    return storage::builder<storage_traits_t>.type<int_t>().dimensions(d1, d2, d3).value(value)();
}

bool basic() {

    uint_t d1 = 4;
    uint_t d2 = 3;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

    boundary<bc_basic, gcl_arch_t>(halos, bc_basic()).apply(in);

    auto inv = in->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool predicate() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(in->target_view());
#else
    boundary_apply<bc_basic, minus_predicate>(halos, bc_basic(), minus_predicate()).apply(in->target_view());
#endif

    auto inv = in->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool twosurfaces() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3, 1);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<bc_two>(halos, bc_two()).apply(in->target_view());
#else
    boundary_apply<bc_two>(halos, bc_two()).apply(in->target_view());
#endif

    auto inv = in->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != i + j + k + 1) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != i + j + k + 1) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 1; j < d2; ++j) {
            for (uint_t k = 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != 1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingzero_1() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3, -1);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<zero_boundary>(halos).apply(in->target_view());
#else
    boundary_apply<zero_boundary>(halos).apply(in->target_view());
#endif

    auto inv = in->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingzero_2() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3, -1);
    auto out = make_storage(d1, d2, d3, -1);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<zero_boundary>(halos).apply(in->target_view(), out->target_view());
#else
    boundary_apply<zero_boundary>(halos).apply(in->target_view(), out->target_view());
#endif

    auto inv = in->host_view();
    auto outv = in->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
                if (outv(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingzero_3_empty_halos() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3, -1);
    auto out = make_storage(d1, d2, d3, -1);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(0, 0, 0, d2 - 1, d2);
    halos[2] = halo_descriptor(0, 0, 0, d3 - 1, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<zero_boundary>(halos).apply(in->target_view(), out->target_view());
#else
    boundary_apply<zero_boundary>(halos).apply(in->target_view(), out->target_view());
#endif

    auto inv = in->host_view();
    auto outv = out->host_view();

    bool result = true;

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 0) {
                    result = false;
                }
                if (outv(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
                if (outv(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingvalue_2() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto in = make_storage(d1, d2, d3, -1);
    auto out = make_storage(d1, d2, d3, -1);

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<value_boundary<int_t>>(halos, value_boundary<int_t>(101))
        .apply(in->target_view(), out->target_view());
#else
    boundary_apply<value_boundary<int_t>>(halos, value_boundary<int_t>(101))
        .apply(in->target_view(), out->target_view());
#endif

    auto inv = in->host_view();
    auto outv = out->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (inv(i, j, k) != 101) {
                    result = false;
                }
                if (outv(i, j, k) != 101) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (inv(i, j, k) != -1) {
                    result = false;
                }
                if (outv(i, j, k) != -1) {
                    result = false;
                }
            }
        }
    }

    return result;
}

bool usingcopy_3() {

    uint_t d1 = 5;
    uint_t d2 = 5;
    uint_t d3 = 5;

    auto src = make_storage(d1, d2, d3, -1);
    auto one = make_storage(d1, d2, d3, -1);
    auto two = make_storage(d1, d2, d3, 0);

    auto srcv = src->host_view();

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                srcv(i, j, k) = i + k + j;
            }
        }
    }

    array<halo_descriptor, 3> halos;
    halos[0] = halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = halo_descriptor(1, 1, 1, d3 - 2, d3);

#ifdef GT_STORAGE_GPU
    boundary_apply_gpu<copy_boundary>(halos).apply(one->target_view(), two->target_view(), src->target_view());
#else
    boundary_apply<copy_boundary>(halos).apply(one->target_view(), two->target_view(), src->target_view());
#endif

    auto onev = one->host_view();
    auto twov = two->host_view();

    bool result = true;

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < 1; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = d3 - 1; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < 1; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = d2 - 1; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 0; i < 1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = d1 - 1; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                if (onev(i, j, k) != i + j + k) {
                    result = false;
                }
                if (twov(i, j, k) != i + j + k) {
                    result = false;
                }
            }
        }
    }

    for (uint_t i = 1; i < d1 - 1; ++i) {
        for (uint_t j = 1; j < d2 - 1; ++j) {
            for (uint_t k = 1; k < d3 - 1; ++k) {
                if (onev(i, j, k) != -1) {
                    result = false;
                }
                if (twov(i, j, k) != 0) {
                    result = false;
                }
            }
        }
    }

    return result;
}

TEST(boundaryconditions, predicate) { EXPECT_EQ(predicate(), true); }

TEST(boundaryconditions, twosurfaces) { EXPECT_EQ(twosurfaces(), true); }

TEST(boundaryconditions, usingzero_1) { EXPECT_EQ(usingzero_1(), true); }

TEST(boundaryconditions, usingzero_2) { EXPECT_EQ(usingzero_2(), true); }

TEST(boundaryconditions, usingzero_3_empty_halos) { EXPECT_EQ(usingzero_3_empty_halos(), true); }

TEST(boundaryconditions, basic) { EXPECT_EQ(basic(), true); }

TEST(boundaryconditions, usingvalue2) { EXPECT_EQ(usingvalue_2(), true); }

TEST(boundaryconditions, usingcopy3) { EXPECT_EQ(usingcopy_3(), true); }
