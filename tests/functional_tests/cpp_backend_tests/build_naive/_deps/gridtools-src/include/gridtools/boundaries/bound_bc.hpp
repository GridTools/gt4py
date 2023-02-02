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

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../common/halo_descriptor.hpp"
#include "../storage/data_store.hpp"
#include "boundary.hpp"

namespace gridtools {
    namespace boundaries {
        namespace _impl {

            /** \ingroup Distributed-Boundaries
             * @{ */

            /** \internal
                @brief Tag type to indicate that a type is a placeholder
            */
            struct Plc {};
            /** \internal
                @brief Tag type to indicate that a type is a placeholder
            */
            struct NotPlc {};

            /** \internal
                @brief Small metafunction that return a type indicating if a typeis a placeholder or not.
                Since std::is_placeholder does not return a boolean but the index of the placeholder if
                the passed type is a placeholder or zero otherwise, this metafunction takes as input an
                index. The use of this metafunction is as this:

                PlcOtNot<std::is_placeholder<T> >::type
            */
            template <int V>
            struct PlcOrNot {
                using type = Plc;
            };

            template <>
            struct PlcOrNot<0> {
                using type = NotPlc;
            };

            /** \internal
                @brief This function is used by gridtools::_impl::substitute_placeholders to
                discrimintate between placeholders and other elements in a tuple. There is a
                specialization for Plc and one for NotPlc.
            */
            template <std::size_t I, typename ROTuple, typename AllTuple>
            decltype(auto) select_element(ROTuple const &ro_tuple, AllTuple const &, Plc) {
                return std::get<std::is_placeholder<std::tuple_element_t<I, AllTuple>>::value - 1>(ro_tuple);
            }

            template <std::size_t I, typename ROTuple, typename AllTuple>
            decltype(auto) select_element(ROTuple const &, AllTuple const &all, NotPlc) {
                return std::get<I>(all);
            }

            /** \internal
                @brief This functions takes a tuple that may contain placeholders and returns a tuple
                for which the placeholders have been substituted by the corresponding elements
                of the another tuple. The function takes a index_sequence of the size of the tuple
                with placeholders.

                This facility uses gridtools::_impl::select_element to discriminate between elements that
                are placeholders from elements that are not.

                \param ro_tuple Tuple of elements to replace the placeholders
                \param all      Tuple of elements that may include placeholders
            */
            template <typename ROTuple, typename AllTuple, std::size_t... IDs>
            auto substitute_placeholders(ROTuple const &ro_tuple, AllTuple const &all, std::index_sequence<IDs...>) {
                return std::make_tuple(select_element<IDs>(ro_tuple,
                    all,
                    typename PlcOrNot<std::is_placeholder<std::tuple_element_t<IDs, AllTuple>>::value>::type{})...);
            }

            inline std::tuple<> rest_tuple(std::tuple<>, std::index_sequence<>) { return {}; }

            /** \internal
                Small facility to obtain a tuple with the elements of am input  tuple execpt the first.
            */
            template <typename... Elems, std::size_t... IDs>
            auto rest_tuple(std::tuple<Elems...> const &x, std::index_sequence<IDs...>) {
                return std::make_tuple(std::get<IDs + 1u>(x)...);
            }

            /**
               @brief Metafunction to return an index_sequence indicating
               the elements in the Tuple that are not std::placeholders

               \tparam Tuple Tuple to be evaluated
            */
            template <typename InputTuple>
            struct comm_indices {
                template <std::size_t I, typename ISeq, typename Tuple, typename VOID = void>
                struct collect_indices;

                template <std::size_t I, typename ISeq>
                struct collect_indices<I, ISeq, std::tuple<>> {
                    using type = ISeq;
                };

                template <std::size_t I, size_t... Is, typename First, typename... Elems>
                struct collect_indices<I,
                    std::index_sequence<Is...>,
                    std::tuple<First, Elems...>,
                    std::enable_if_t<std::is_placeholder_v<First> == 0, void>> {
                    using type =
                        typename collect_indices<I + 1, std::index_sequence<Is..., I>, std::tuple<Elems...>>::type;
                };

                template <std::size_t I, typename ISeq, typename First, typename... Elems>
                struct collect_indices<I,
                    ISeq,
                    std::tuple<First, Elems...>,
                    std::enable_if_t<(std::is_placeholder_v<First> > 0), void>> {
                    using type = typename collect_indices<I + 1, ISeq, std::tuple<Elems...>>::type;
                };

                using type = typename collect_indices<0, std::index_sequence<>, InputTuple>::type;
            };

            template <typename T, typename VOID = void>
            struct contains_placeholders : std::false_type {};

            template <>
            struct contains_placeholders<std::tuple<>> : std::false_type {};

            template <typename T, typename... Ts>
            struct contains_placeholders<std::tuple<T, Ts...>, std::enable_if_t<std::is_placeholder_v<T> == 0, void>>
                : contains_placeholders<std::tuple<Ts...>>::type {};

            template <typename T, typename... Ts>
            struct contains_placeholders<std::tuple<T, Ts...>, std::enable_if_t<(std::is_placeholder_v<T> > 0), void>>
                : std::true_type {};

            /** @} */

        } // namespace _impl

        /** \ingroup Distributed-Boundaries
         * @{ */

        /**
         * @brief class to associate data store to gridtools::boundary class for
         * boundary condition class, and data_stores
         *
         * User is not supposed to instantiate this class explicitly but instead
         * gridtools::bind_bc function, which is a maker, will be used to indicate
         * the boundary conditions to be applied in a distributed boundary
         * conditions application.
         *
         * \tparam BCApply The class name with boudary condition functions applied by gridtools::boundary
         * \tparam DataStores Tuple type of data stores (or placeholders) to be passed for boundary condition
         * application \tparam ExcStoresIndicesSeq index_sequence with the indices of data_stores in DataStores that
         * will be undergo halo-update operations
         */
        template <typename BCApply, typename DataStores, typename ExcStoresIndicesSeq>
        struct bound_bc;

        template <typename BCApply, typename... DataStores, std::size_t... ExcStoresIndices>
        struct bound_bc<BCApply, std::tuple<DataStores...>, std::index_sequence<ExcStoresIndices...>> {
            using boundary_class = BCApply;
            using stores_type = std::tuple<DataStores...>;
            using exc_stores_type = std::tuple<std::tuple_element_t<ExcStoresIndices, stores_type> const &...>;

          private:
            boundary_class m_bcapply;
            stores_type m_stores;

          public:
            /**
             * @brief Constructor to associate the objects whose types are listed in the
             * template argument list to the corresponding data members
             */
            template <typename ST>
            bound_bc(BCApply bca, ST &&stores_list)
                : m_bcapply(bca), m_stores{std::forward<stores_type>(stores_list)} {}

            /**
             * @brief Function to retrieve the tuple of data stores to pass to the the boundary
             * condition class
             */
            stores_type const &stores() const {
                static_assert(not _impl::contains_placeholders<stores_type>::value,
                    "Some inputs to boundary conditions are placeholders. Remeber to use .associate(...) member "
                    "function "
                    "to substitute tham");
                return m_stores;
            }

            /**
             * @brief Function to retrieve the tuple of data stores to pass to the the halo-update
             * communication pattern
             */
            exc_stores_type exc_stores() const {
                return std::tuple<std::tuple_element_t<ExcStoresIndices, stores_type> const &...>(
                    std::get<ExcStoresIndices>(m_stores)...);
            }

            /**
             * @brief Function to retrieve the boundary condition application class
             */
            boundary_class boundary_to_apply() const { return m_bcapply; }

            /**
             * @brief In the case in which the DataStores passed as template to the bound_bc class
             * contains placeholders, this member function will return a bound_bc object in which
             * the placeholders have been substituted with the data stores in the corresponding
             * position. These data stores will not be passed to the halo-update operation, thus
             * implementing a separation between read-only data stores and the others.
             *
             * \tparam ReadOnly Variadic pack with the types of the data stores to associate to placeholfders
             * \param ro_stores Variadic pack with the data stores to associate to placeholders
             */
            template <typename... ReadOnly>
            auto associate(ReadOnly &&...ro_stores) const -> bound_bc<BCApply,
                decltype(_impl::substitute_placeholders(std::make_tuple(ro_stores...),
                    m_stores,
                    std::make_index_sequence<std::tuple_size_v<decltype(m_stores)>>{})),
                typename _impl::comm_indices<stores_type>::type> {
                auto ro_store_tuple = std::forward_as_tuple(ro_stores...);
                // we need to substitute the placeholders with the
                auto full_list = _impl::substitute_placeholders(
                    ro_store_tuple, m_stores, std::make_index_sequence<std::tuple_size_v<decltype(m_stores)>>{});

                return bound_bc<BCApply, decltype(full_list), typename _impl::comm_indices<stores_type>::type>{
                    m_bcapply, std::move(full_list)};
            }
        };

        /**
         * @brief Free-standing function used to construcs a gridtools::bound_bc object, which is
         * used to run boundary condition application and halo-update operations.
         *
         * If the DataStores provided are std::placeholders, a subsequent call to
         * gridtools::bound_bc::associate to substitute the placeholders with data stores
         * that will be then excluded by halo-update operations.
         *
         * \tparam BCApply Boundary condition class (usually deduced)
         * \tparam DataStores Parameter pack type with the data stores or placeholders (std::placeholders should be
         * used) (deduced)
         *
         * \param bc_apply The boundary condition class
         * \param stores Parameter pack with the data stores or placeholders (std::placeholders hosuld be used)
         */
        template <typename BCApply, typename... DataStores>
        bound_bc<BCApply, std::tuple<std::decay_t<DataStores>...>, std::index_sequence_for<DataStores...>> bind_bc(
            BCApply bc_apply, DataStores &&...stores) {

            // Concept checking on BCApply is not ready yet.
            // Check that the stores... are either data stores or placeholders
            static_assert(std::conjunction<
                              std::bool_constant<storage::is_data_store_ptr<typename std::decay_t<DataStores>>::value or
                                                 std::is_placeholder<std::decay_t<DataStores>>::value>...>::value,
                "The arguments of bind_bc, after the first, must be data_stores or std::placeholders");
            return {bc_apply, std::forward_as_tuple(stores...)};
        }

        /** @brief Metafunctions to query if a type is a bound_bc
         */
        template <typename T>
        struct is_bound_bc : std::false_type {};

        template <typename... T>
        struct is_bound_bc<bound_bc<T...>> : std::true_type {};
        /** @} */
    } // namespace boundaries
} // namespace gridtools
