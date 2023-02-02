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

#include <vector>

#include "../common/halo_descriptor.hpp"
#include "../common/layout_map.hpp"
#include "high_level/descriptor_generic_manual.hpp"
#include "high_level/descriptors.hpp"
#include "high_level/descriptors_manual_gpu.hpp"
#include "high_level/field_on_the_fly.hpp"
#include "low_level/Halo_Exchange_3D.hpp"
#include "low_level/arch.hpp"
#include "low_level/proc_grids_3D.hpp"

namespace gridtools {
    namespace gcl {
        /**
           \anchor descr_halo_exchange_dynamic_ut
           This is the main class for the halo exchange pattern in the case
           in which the data pointers are not known before hand and they are
           passed to the pattern when packing and unpacking is needed.

           The interface requires two layout maps ( \link gridtools::layout_map
           \endlink ) one to specify the data layout, the other to
           specify the relation between data layout and processor grid. This
           is an important asepct that will be explained here and also in
           the introduction.

           The First layout map to be passed to the pattern class is the
           data layout. The user defines a convention in which the
           dimensions of the data fields are ordered logically depending on
           the application and/or user preferences. For instance, we can
           call the dimensions in this application order i, j, and k. The
           layout map in this case specifies in what position each
           dimension is in the increasing stride order. For instance:

           \code
           gridtools::layout_map<1,0,2>
           \endcode

           Indicates that the first dimension in the data (i) is the second
           in the increasing stride order, while the second (j) is actually
           the first one (has stride 1). Finally the k dimension is the one
           with the highest stride and is in the third position (2). The
           layout in memory of the user data can be specified then as 'jik'.

           Similarly, the second template argument in the halo exchange
           pattern is the map between data coordinates and the processor
           grid coordinates. The following layout specification

           \code
           gridtools::layout_map<1,0,2>
           \endcode

           would mean: The first dimension in data matches with the second
           dimension of the computing grid, the second is the first, and the
           third one is the third one.

           Let's consider a 2D case at first, to show an additional
           example. Suppose user data is thought to be ij, meaning the user
           think to i as the first coordinate and j to the
           second. Alternatively the user would use (i,j) to indicate an
           element in the data array. The layout is C like, so that j is
           actuallly the first coordinate in the increasing stride ordering,
           and i the second.

           The first template argument to the pattern would then be
           \code
           gridtools::layout_map<1,0>
           \endcode

           The second template argument is still a \link gridtools::layout_map
           \endlink , but this time it indicates the mapping between data
           and processor grid. The data is still condidered in the user
           convention.

           Suppose the processor gris has size PIxPJ, and these may be the
           MPI communicator sizes (coords[0]=PI, coords[1]=PJ). Now, we want
           to say that the first dimension on data (first in the user
           convention, not int the increasing stride order) 'extends' to the
           computing gris, or that the first dimension in the data
           correspons to the first dimension in the computing grid. Let's
           consider a 2x1 proc grid, and the first dimension of the data
           being the rows (i) and the second the column (j). In this case we
           are thinking to a distribution like this:

           \code
           >j>>
           ------
           v |0123|
           i |1234|  Proc 0,0
           v |2345|
           v |3456|
           ------

           >j>>
           ------
           v |4567|
           i |5678|  Proc 1,0
           v |6789|
           v |7890|
           ------
           \endcode

           In this case the map between data and the processor grid is:
           \code
           gridtools::layout_map<0,1>
           \endcode

           On the other hand, having specified
           \code
           gridtools::layout_map<1,0>
           \endcode
           for this map, would imply a layout/distribution like the following:

           \code
           >j>>                 >j>>
           ------               ------
           v |0123|             v |4567|
           i |1234|  Proc 0,0;  i |5678|  Proc 1,0
           v |2345|             v |6789|
           v |3456|             v |7890|
           ------               ------
           \endcode

           Where the second dimension in the data correspond to the fist
           dimension in the processor grid. Again, the data coordinates
           ordering is the one the user choose to be the logical order in
           the application, not the increasing stride order.

           To find an example in 2D refer to
           test_halo_exchange_2D.cpp

           while a 3D example can be found in
           test_halo_exchange.cpp

           The other template arguments are the type of the elements
           contained in the data arrays and the number of dimensions of
           data.

           \tparam layout_map Layout_map \link gridtools::layout_map \endlink specifying the data layout as the position
           of each dimension of the user data in the increasing stride order \tparam layout2proc_map_abs Layout_map
           \link gridtools::layout_map \endlink specifying which dimension in the data corresponds to the which
           dimension in the processor grid \tparam DataType Value type the elements int the arrays \tparam DIMS Number
           of dimensions of data arrays (equal to the dimension of the processor grid) \tparam GCL_ARCH Specification of
           the "architecture", that is the place where the data to be exchanged is. Possible coiches are defined in
           low_level/gcl_arch.h .
        */
        template <typename T_layout_map,
            typename layout2proc_map_abs,
            typename DataType,
            typename Gcl_Arch = cpu,
            int version = 0>
        class halo_exchange_dynamic_ut {
            // This is necessary since the internals of gcl use "increasing stride order" instead of "decreasing stride
            // order"
            using layout_map = reverse_map<T_layout_map>;
            using layout2proc_map = layout_transform<layout_map, layout2proc_map_abs>;

          public:
            /**
               Type of the computin grid associated to the pattern
            */
            typedef MPI_3D_process_grid_t<3> grid_type;

            static constexpr int DIMS = 3;

            /**
               Type of the Level 3 pattern used.
            */
            typedef Halo_Exchange_3D<grid_type> pattern_type;

          private:
            typedef hndlr_dynamic_ut<DataType, grid_type, pattern_type, layout2proc_map, Gcl_Arch> hd_t;

            hd_t hd;

            halo_exchange_dynamic_ut(halo_exchange_dynamic_ut const &) {}

          public:
            /** constructor that takes the periodicity (mathich the \link
                boollist_concept \endlink concept, and the MPI CART
                communicator in 3 dimensions of the processing grid. the
                periodicity is specified in the order chosen by the
                programmer for the data, as in the rest of the
                application. It is up to the constructor implementation
                to translate it into the right order depending on the
                layout_map passed to the class.

                \param[in] c Periodicity specification as in \link boollist_concept \endlink
                \param[in] comm MPI CART communicator with dimension 3
            */
            explicit halo_exchange_dynamic_ut(typename grid_type::period_type const &c, MPI_Comm const &comm)
                : hd(c.template permute<layout2proc_map_abs>(), comm) {}

            /** Function to rerturn the L3 level pattern used inside the pattern itself.

                \return The pattern al level 3 used to exchange data
            */
            pattern_type const &pattern() const { return hd.pattern(); }

            /**
               Function to setup internal data structures for data exchange and preparing eventual underlying layers

               \param max_fields_n Maximum number of data fields that will be passed to the communication functions
            */
            void setup(int max_fields_n) { hd.setup(max_fields_n); }

            /**
               Function to register halos with the pattern. The registration
               happens specifing the ordiring of the dimensions as the user
               defines it. For instance in the example given at the
               introduction of the class, the ordering will be 0 for i and 1
               for j, no matter of the layout_map parameter passed during
               class instantiation.

               \tparam DI index of the dimension to be set relative to the logical ordering chosen in the application,
               not the increasing stride ordering. \param[in] minus Please see field_descriptor, halo_descriptor or
               \link MULTI_DIM_ACCESS \endlink for details \param[in] plus Please see field_descriptor, halo_descriptor
               or \link MULTI_DIM_ACCESS \endlink for details \param[in] begin Please see field_descriptor,
               halo_descriptor or \link MULTI_DIM_ACCESS \endlink for details \param[in] end Please see
               field_descriptor, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for details \param[in] t_len Please
               see field_descriptor, halo_descriptor or \link MULTI_DIM_ACCESS \endlink for details
            */
            template <int DI>
            void add_halo(int minus, int plus, int begin, int end, int t_len) {
                hd.halo.add_halo(layout_map::at(DI), minus, plus, begin, end, t_len);
            }

            template <int DI>
            void add_halo(halo_descriptor const &halo) {
                hd.halo.add_halo(layout_map::at(DI), halo);
            }

            /**
               Function to pack data to be sent

               \param[in] _fields data fields to be packed
            */
            template <typename... FIELDS>
            void pack(const FIELDS *..._fields) {
                hd.pack(_fields...);
            }

            /**
               Function to unpack received data

               \param[in] _fields data fields where to unpack data
            */
            template <typename... FIELDS>
            void unpack(FIELDS *..._fields) {
                hd.unpack(_fields...);
            }

            /**
               Function to unpack received data

               \param[in] fields vector with data fields pointers to be packed from
            */
            void pack(std::vector<DataType *> const &fields) { hd.pack(fields); }

            /**
               Function to unpack received data

               \param[in] fields vector with data fields pointers to be unpacked into
            */
            void unpack(std::vector<DataType *> const &fields) { hd.unpack(fields); }

            /**
               function to trigger data exchange

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void exchange() { hd.exchange(); }

            void post_receives() { hd.post_receives(); }

            void do_sends() { hd.do_sends(); }

            /**
               function to trigger data exchange initiation when using split-phase communication.

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void start_exchange() { hd.start_exchange(); }

            /**
               function to trigger data exchange

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void wait() { hd.wait(); }

            grid_type const &comm() const { return hd.comm(); }
        };

        /**
           This is the main class for the halo exchange pattern in the case
           in which the data pointers, data types, and shapes are not known
           before hand and they are passed to the pattern when packing and
           unpacking is needed. This is the most generic pattern available
           in GCL for halo exchange.

           The interface requires one layout map ( \link gridtools::layout_map
           \endlink ) to specify the relation between data layout and
           processor grid. as follows The following layout specification

           \code
           gridtools::layout_map<1,0,2>
           \endcode

           would mean: The first dimension in data matches with the second
           dimension of the computing grid, the second is the first, and the
           third one is the third one.

           \tparam layout2proc_map_abs Layout_map \link gridtools::layout_map \endlink specifying which dimension in the
           data corresponds to the which dimension in the processor grid
           \tparam DIMS Number of dimensions of data arrays (equal to the dimension of the processor grid)
           \tparam GCL_ARCH Specification of the "architecture", that is the place where the data to be exchanged is.
           Possible coiches are defined in low_level/gcl_arch.h .
        */
        template <typename layout2proc_map, typename Gcl_Arch>
        class halo_exchange_generic_base {

          public:
            static constexpr int DIMS = 3;

            /**
               Type of the computin grid associated to the pattern
            */
            typedef MPI_3D_process_grid_t<3> grid_type;

            /**
               Type of the Level 3 pattern used. This is available only if the pattern uses a Level 3 pattern.
               In the case the implementation is not using L3, the type is not available.
            */
            typedef Halo_Exchange_3D<grid_type> pattern_type;

          private:
            hndlr_generic<pattern_type, layout2proc_map, Gcl_Arch> hd;

          public:
            /** constructor that takes the periodicity (matching the \link
                boollist_concept \endlink concept, and the MPI CART
                communicator in DIMS (specified as template argument to the
                pattern) dimensions of the processing grid. the periodicity is
                specified in the order chosen by the programmer for the data,
                as in the rest of the application. It is up tp the
                construnctor implementation to translate it into the right
                order depending on the gridtools::layout_map passed to the class.

                \param[in] c Periodicity specification as in \link boollist_concept \endlink
                \param[in] comm MPI CART communicator with dimension DIMS (specified as template argument to the
               pattern).
            */
            explicit halo_exchange_generic_base(typename grid_type::period_type const &c, MPI_Comm comm)
                : hd(grid_type(c.template permute<layout2proc_map>(), comm)) {}

            explicit halo_exchange_generic_base(grid_type const &g) : hd(g) {}

            /** Function to rerturn the L3 level pattern used inside the pattern itself.

                \return The pattern al level 3 used to exchange data
            */
            pattern_type const &pattern() const { return hd.pattern(); }

            /**
               Function to setup internal data structures for data exchange and preparing eventual underlying layers.
               The sizes are computed mulplitplying the entities.

               \param max_fields_n Maximum, or sufficiently large (if compensated by other parameters), number of data
               fields that will be passed to the communication functions
               \param halo_example the maximum, or a suffciently large, halo from which the pattern can determine the
               needs of the data exchange \param typesize Maximum, or sufficiently large, size fo the types of values to
               be exchanged.
            */
            template <typename DataType, typename layomap, template <typename> class _traits>
            void setup(
                int max_fields_n, field_on_the_fly<DataType, layomap, _traits> const &halo_example, int typesize) {
                hd.setup(max_fields_n, halo_example, typesize);
            }

            /**
               Function to pack data to be sent

               \param[in] _fields data fields to be packed
            */
            template <typename... FIELDS>
            void pack(const FIELDS &..._fields) const {
                hd.pack(_fields...);
            }

            /**
               Function to unpack received data

               \param[in] _fields data fields where to unpack data
            */
            template <typename... FIELDS>
            void unpack(const FIELDS &..._fields) const {
                hd.unpack(_fields...);
            }

            /**
               Function to unpack received data

               \tparam array_of_fotf this should be an array of field_on_the_fly
               \param[in] fields vector with fields on the fly
            */
            template <typename T1, typename T2, template <typename> class T3>
            void pack(std::vector<field_on_the_fly<T1, T2, T3>> const &fields) {
                hd.pack(fields);
            }

            /**
               Function to unpack received data

               \tparam array_of_fotf this should be an array of field_on_the_fly
               \param[in] fields vector with fields on the fly
            */
            template <typename T1, typename T2, template <typename> class T3>
            void unpack(std::vector<field_on_the_fly<T1, T2, T3>> const &fields) {
                hd.unpack(fields);
            }

            /**
               function to trigger data exchange

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void exchange() { hd.exchange(); }

            void post_receives() { hd.post_receives(); }

            void do_sends() { hd.do_sends(); }

            /**
               function to trigger data exchange initiation when using split-phase communication.

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void start_exchange() { hd.start_exchange(); }

            /**
               function to trigger data exchange

               Note: when the start_exchange() + wait() combination is used, the exchange() method should not be used,
               and vice versa.
            */
            void wait() { hd.wait(); }
        };

        template <typename layout2proc_map, typename Gcl_Arch = cpu>
        class halo_exchange_generic;

        // different traits are needed
        template <typename layout2proc_map>
        class halo_exchange_generic<layout2proc_map, cpu> : public halo_exchange_generic_base<layout2proc_map, cpu> {

            typedef halo_exchange_generic_base<layout2proc_map, cpu> base_type;

          public:
            typedef typename base_type::grid_type grid_type;

            typedef typename base_type::pattern_type pattern_type;

            template <typename DT>
            struct traits {
                static const int I = 3;
                typedef empty_field_no_dt base_field;
            };

            explicit halo_exchange_generic(typename grid_type::period_type const &c, MPI_Comm comm)
                : base_type(c, comm) {}

            explicit halo_exchange_generic(grid_type const &g) : base_type(g) {}
        };

        // different traits are needed
        template <typename layout2proc_map>
        class halo_exchange_generic<layout2proc_map, gpu> : public halo_exchange_generic_base<layout2proc_map, gpu> {
            static const int DIMS = 3;

            typedef halo_exchange_generic_base<layout2proc_map, gpu> base_type;

          public:
            typedef typename base_type::grid_type grid_type;

            typedef typename base_type::pattern_type pattern_type;

            template <typename DT>
            struct traits {
                static const int I = DIMS;
                typedef empty_field_no_dt_gpu<I> base_field;
            };

            explicit halo_exchange_generic(typename grid_type::period_type const &c, MPI_Comm comm)
                : base_type(c, comm) {}

            explicit halo_exchange_generic(grid_type const &g) : base_type(g) {}
        };
    } // namespace gcl
} // namespace gridtools
