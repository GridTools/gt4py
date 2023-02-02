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

/** \defgroup Distributed-Boundaries Distributed Boundary Conditions
 */

#include <memory>
#include <type_traits>
#include <utility>

#include "../common/halo_descriptor.hpp"
#include "../common/timer/timer.hpp"
#include "../gcl/halo_exchange.hpp"
#include "bound_bc.hpp"
#include "grid_predicate.hpp"
#include "predicate.hpp"

namespace gridtools {
    namespace boundaries {
        /** \ingroup Distributed-Boundaries
         * @{ */

        /**
            @brief This class takes a communication traits class and provide a facility to
            perform boundary conditions and communications in a single call.

            After construction a call to gridtools::distributed_boundaries::exchange takes
            a list of gridtools::data_store or girdtools::bound_bc. The data stores will be
            directly used in communication primitives for performing halo_update operation,
            while bound_bc elements will be priocessed by exracting the data stores that need
            communication and others that will go through boundary condition application as
            specified in the bound_bc class.

            Example of use (where `a`, `b`, `c`, and `d` are of data_store type:
            \verbatim
                using storage_info_t = storage_tr::storage_info_t< 0, 3, halo< 2, 2, 0 > >;
                using storage_type = storage_tr::data_store_t< triplet, storage_info_t >;

                halo_descriptor di{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1};
                halo_descriptor dj{halo_size, halo_size, halo_size, d2 - halo_size - 1, d2};
                halo_descriptor dk{0, 0, 0, d3 - 1, d3};
                array< halo_descriptor, 3 > halos{di, dj, dk};

                using cabc_t = distributed_boundaries< comm_traits< storage_type, gcl::cpu > >;

                cabc_t cabc{halos, // halos for communication
                            {false, false, false}, // Periodicity in first, second and third dimension
                            4, // Maximum number of data_stores to be handled by this communication object
                            GCL_WORLD}; // Communicator to be used

                cabc.exchange(bind_bc(value_boundary< triplet >{triplet{42, 42, 42}}, a),
                              bind_bc(copy_boundary{}, b, _1).associate(c),
                              d);
            \endverbatim

            \tparam CTraits Communication traits. To see an example see gridtools::comm_traits
        */
        template <typename CTraits>
        struct distributed_boundaries {

            using pattern_type = gcl::halo_exchange_dynamic_ut<typename CTraits::data_layout,
                typename CTraits::proc_layout,
                typename CTraits::value_type,
                typename CTraits::comm_arch_type>;

          private:
            using performance_meter_t = timer<typename CTraits::timer_impl_t>;

            array<halo_descriptor, 3> m_halos;
            array<int_t, 3> m_sizes;
            uint_t m_max_stores;
            std::unique_ptr<pattern_type> m_he;

            performance_meter_t m_meter_pack;
            performance_meter_t m_meter_exchange;
            performance_meter_t m_meter_bc;

          public:
            /**
                @brief Constructor of distributed_boundaries.

                \param halos array of 3 gridtools::halo_desctiptor containing the halo information to be used for
               communication
                \param period Periodicity specification, a gridtools::boollist with three elements, one for each
               dimension. true mean the dimension is periodic \param max_stores Maximum number of data_stores to be used
               in communication. PAssing more will couse a runtime error (probably segmentation fault), passing less
               will underutilize the memory \param CartComm MPI communicator to use in the halo update operation [must
               be a cartesian communicator]
            */
            distributed_boundaries(array<halo_descriptor, 3> halos,
                typename pattern_type::grid_type::period_type period,
                uint_t max_stores,
                MPI_Comm CartComm)
                : m_halos{halos}, m_sizes{0, 0, 0}, m_max_stores{max_stores},
                  m_he(std::make_unique<pattern_type>(period, CartComm)), m_meter_pack("pack/unpack       "),
                  m_meter_exchange("exchange          "), m_meter_bc("boundary condition") {
                m_he->pattern().proc_grid().fill_dims(m_sizes);

                m_he->template add_halo<0>(m_halos[0].minus(),
                    m_halos[0].plus(),
                    m_halos[0].begin(),
                    m_halos[0].end(),
                    m_halos[0].total_length());

                m_he->template add_halo<1>(m_halos[1].minus(),
                    m_halos[1].plus(),
                    m_halos[1].begin(),
                    m_halos[1].end(),
                    m_halos[1].total_length());

                m_he->template add_halo<2>(m_halos[2].minus(),
                    m_halos[2].plus(),
                    m_halos[2].begin(),
                    m_halos[2].end(),
                    m_halos[2].total_length());

                m_he->setup(m_max_stores);
            }

            /**
                @brief Member function to perform boundary condition only
                on a list of jobs.  A job is either a
                gridtools::data_store to be used during communication (so
                it is skipped by this function) or a
                gridtools::bound_bc to apply boundary conditions. The
                synthax is the same as the
                distributed_boundaries::exchange, but the communication is
                not performed.

                \param jobs Variadic list of jobs
            */
            template <typename... Jobs>
            void boundary_only(Jobs const &...jobs) {
                using execute_in_order = int[];
                m_meter_bc.start();
                (void)execute_in_order{(apply_boundary(jobs), 0)...};
                m_meter_bc.pause();
            }

            /**
                @brief Member function to perform boundary condition and communication on a list of jobs.
                A job is either a gridtools::data_store to be used during communication or a gridtools::bound_bc
                to apply boundary conditions and halo_update operations for the data_stores that are not input-only
                (that will be indicated with the gridtools::bound_bc::associate member function.)

                The function first perform communication then applies the boundary condition. This allows a
               copy-boundary from the inner region to the halo region to run as expected.

                \param jobs Variadic list of jobs
            */
            template <typename... Jobs>
            void exchange(Jobs const &...jobs) {
                auto all_stores_for_exc = std::tuple_cat(collect_stores(jobs)...);
                if (m_max_stores < sizeof...(jobs)) {
                    std::string err{"Too many data stores to be exchanged" + std::to_string(sizeof...(jobs)) +
                                    " instead of the maximum allowed, which is " + std::to_string(m_max_stores)};
                    throw std::runtime_error(err);
                }

                m_meter_pack.start();
                call_pack(all_stores_for_exc, std::make_integer_sequence<uint_t, sizeof...(jobs)>{});
                m_meter_pack.pause();
                m_meter_exchange.start();
                m_he->exchange();
                m_meter_exchange.pause();
                m_meter_pack.start();
                call_unpack(all_stores_for_exc, std::make_integer_sequence<uint_t, sizeof...(jobs)>{});
                m_meter_pack.pause();

                boundary_only(jobs...);
            }

            auto const &proc_grid() const { return m_he->comm(); }

            std::string print_meters() const {
                return m_meter_pack.to_string() + "\n" + m_meter_exchange.to_string() + "\n" + m_meter_bc.to_string();
            }

            double get_time_pack() const { return m_meter_pack.get_time(); }
            double get_time_exchange() const { return m_meter_exchange.get_time(); }
            double get_time_boundary() const { return m_meter_bc.get_time(); }

            size_t get_count_exchange() const { return m_meter_exchange.get_count(); }
            // no get_count_pack() as it is equivalent to get_count_exchange()
            size_t get_count_boundary() const { return m_meter_bc.get_count(); }

            void reset_meters() {
                m_meter_pack.reset_meter();
                m_meter_exchange.reset_meter();
                m_meter_bc.reset_meter();
            }

          private:
            template <typename BoundaryApply, typename ArgsTuple, uint_t... Ids>
            static void call_apply(
                BoundaryApply boundary_apply, ArgsTuple const &args, std::integer_sequence<uint_t, Ids...>) {
                boundary_apply.apply(std::get<Ids>(args)...);
            }

            template <typename BCApply>
            std::enable_if_t<is_bound_bc<BCApply>::value, void> apply_boundary(BCApply bcapply) {
                call_apply(make_boundary<typename CTraits::comm_arch_type>(
                               m_halos, bcapply.boundary_to_apply(), make_proc_grid_predicate(m_he->comm())),
                    bcapply.stores(),
                    std::make_integer_sequence<uint_t, std::tuple_size_v<typename BCApply::stores_type>>{});
            }

            template <typename BCApply>
            std::enable_if_t<not is_bound_bc<BCApply>::value, void> apply_boundary(BCApply) {
                /* do nothing for a pure data_store*/
            }

            template <typename FirstJob>
            static auto collect_stores(
                FirstJob const &firstjob, std::enable_if_t<is_bound_bc<FirstJob>::value, void *> = nullptr) {
                return firstjob.exc_stores();
            }

            template <typename FirstJob>
            static auto collect_stores(
                FirstJob const &first_job, std::enable_if_t<not is_bound_bc<FirstJob>::value, void *> = nullptr) {
                return std::make_tuple(first_job);
            }

            template <typename Stores, uint_t... Ids>
            void call_pack(Stores const &stores, std::integer_sequence<uint_t, Ids...>) {
                m_he->pack(std::get<Ids>(stores)->get_const_target_ptr()...);
            }

            template <typename Stores>
            void call_pack(Stores const &stores, std::integer_sequence<uint_t>) {}

            template <typename Stores, uint_t... Ids>
            void call_unpack(Stores const &stores, std::integer_sequence<uint_t, Ids...>) {
                m_he->unpack(std::get<Ids>(stores)->get_target_ptr()...);
            }

            template <typename Stores>
            static void call_unpack(Stores const &stores, std::integer_sequence<uint_t>) {}
        };
        /** @} */
    } // namespace boundaries
} // namespace gridtools
