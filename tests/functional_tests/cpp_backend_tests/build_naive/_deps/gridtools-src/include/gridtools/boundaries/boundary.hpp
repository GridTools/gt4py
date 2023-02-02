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

#include "../common/defs.hpp"
#include "../gcl/low_level/arch.hpp"
#include "apply.hpp"
#include "predicate.hpp"

#ifdef GT_CUDACC
#include "apply_gpu.hpp"
#endif

/** \defgroup Boundary-Conditions Boundary Conditions
 */

namespace gridtools {
    namespace boundaries {
        namespace _impl {
            /** \ingroup Boundary-Conditions
             * @{
             */

            template <typename /*GclArch*/, typename BoundaryFunction, typename Predicate>
            struct select_apply {
                using type = boundary_apply<BoundaryFunction, Predicate>;
            };

#ifdef GT_CUDACC
            template <typename BoundaryFunction, typename Predicate>
            struct select_apply<gcl::gpu, BoundaryFunction, Predicate> {
                using type = boundary_apply_gpu<BoundaryFunction, Predicate>;
            };
#endif
            /** @} */
        } // namespace _impl

        /** \ingroup Boundary-Conditions
         * @{
         */

        /**
           @brief Main interface for boundary condition application.

           \tparam BoundaryFunction The boundary condition functor
           \tparam Arch The target where the data is (e.g., Host or Cuda)
           \tparam Predicate Runtime predicate for deciding if to apply boundary conditions or not on certain regions
           based on runtime values (useful to deal with non-periodic distributed examples
         */
        template <typename BoundaryFunction, class Arch, typename Predicate = default_predicate>
        struct boundary {
            using bc_apply_t = typename _impl::select_apply<Arch, BoundaryFunction, Predicate>::type;

            bc_apply_t bc_apply;

            boundary(array<halo_descriptor, 3> const &hd,
                BoundaryFunction const &boundary_f,
                Predicate predicate = Predicate())
                : bc_apply(hd, boundary_f, predicate) {}

            template <typename... DataFields>
            void apply(DataFields &...data_fields) const {
                bc_apply.apply(data_fields->target_view()...);
            }
        };

        template <class Arch, class BoundaryFunction, class Predicate = default_predicate>
        auto make_boundary(
            array<halo_descriptor, 3> const &hd, BoundaryFunction &&boundary_f, Predicate &&predicate = Predicate()) {
            return boundary<BoundaryFunction, Arch, Predicate>(
                hd, std::forward<BoundaryFunction>(boundary_f), std::forward<Predicate>(predicate));
        }
        /** @} */
    } // namespace boundaries
} // namespace gridtools
