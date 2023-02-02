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

#include <mpi.h>

namespace gridtools {
    namespace gcl {
        inline auto world() { return MPI_COMM_WORLD; }

        namespace impl_ {
            inline int &pid_holder() {
                static int res;
                return res;
            }

            inline int &procs_holder() {
                static int res;
                return res;
            }

            inline void init(int *argc, char ***argv) {
                int ready;
                MPI_Initialized(&ready);
                if (!ready)
                    MPI_Init(argc, argv);
                MPI_Comm_rank(world(), &pid_holder());
                MPI_Comm_size(world(), &procs_holder());
            }
        } // namespace impl_

        inline int pid() { return impl_::pid_holder(); }

        inline int procs() { return impl_::procs_holder(); }

        inline void init(int argc, char **argv) { impl_::init(&argc, &argv); }

        inline void init() { impl_::init(nullptr, nullptr); }

        inline void finalize() { MPI_Finalize(); }
    } // namespace gcl
} // namespace gridtools
