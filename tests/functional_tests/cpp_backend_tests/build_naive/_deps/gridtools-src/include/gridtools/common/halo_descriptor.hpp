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

#include <cassert>
#include <string>

#include "defs.hpp"
#include "host_device.hpp"

/**
@file
@brief defines the halo region
*/

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup halo_descriptor Halo Descriptor
        @{
    */

    /** \class halo_descriptor
        Given a dimension it is described like this:
        \code
        |-----|------|---------------|---------|----|
        | pad0|minus |    length     | plus    |pad1|
                      ^begin        ^end
        |               total_length                |
        \endcode

        In this way five numbers are needed to describe the whole
        structure, assuming the first index of pad0 is fixed to 0, and
        they are minus, begin, end, plus, and total_length.
    */
    class halo_descriptor {
        uint_t m_minus;        // halo on the minus direction
        uint_t m_plus;         // halo on the plus direction
        uint_t m_begin;        // index of the first element of the active region
        uint_t m_end;          // index of the last element of the active region
        uint_t m_total_length; // minus+plus+(end-begin+1)+pads

      public:
        /**
         * Default constructors: all parameters to zero. Used to be able to have vectors of halo descriptors with
         * initialized values that are subsequently copy constructed (since this is a POD structure it can be copied
         * without an explicit copy constructor.
         */
        GT_FUNCTION halo_descriptor() : halo_descriptor(0, 0, 0, 0, 1) {}

        /**
         * @brief simple ctor without halos and padding
         */
        GT_FUNCTION halo_descriptor(uint_t size) : halo_descriptor(0, 0, 0, size - 1, size) {}

        /** Main constructors where the five integers are passed.
          \code
          |-----|------|---------------|---------|----|
          | pad0|minus |    length     | plus    |pad1|
                        ^begin        ^end
          |               total_length                |
          \endcode

          \param[in] m The minus parameter
          \param[in] p The plus parameter
          \param[in] b The begin parameter
          \param[in] e The end parameter (inclusive)
          \param[in] l The total_length parameter
         */
        GT_FUNCTION halo_descriptor(uint_t m, uint_t p, uint_t b, uint_t e, uint_t l)
            : m_minus(m), m_plus(p), m_begin(b), m_end(e), m_total_length(l) {
            assert(m_minus + m_plus + (m_end - m_begin + 1) <= m_total_length);
            assert(m_begin <= m_end);
            assert(m_plus <= m_total_length - m_end - 1);
            assert(m_begin >= m_minus);
        }

        /**
           Begin index for the loop on the outside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
        */
        GT_FUNCTION int_t loop_low_bound_outside(
            int_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            switch (I) {
            case 0:
                return m_begin;
            case 1:
                return m_end + 1;
            case -1:
                return (m_begin - m_minus);
            default:
                assert(false);
                return 1;
            }
        }

        /**
           End index for the loop on the outside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        GT_FUNCTION int_t loop_high_bound_outside(
            int_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            switch (I) {
            case 0:
                return m_end;
            case 1:
                return m_end + m_plus;
            case -1:
                return m_begin - 1;
            default:
                assert(false);
                return 1;
            }
        }

        /**
           Begin index for the loop on the inside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        GT_FUNCTION int_t loop_low_bound_inside(
            int_t I) const { // inside is the fact that the halos are ones outside the begin-end region

            switch (I) {
            case 0:
                return m_begin;
            case 1:
                return m_end - m_minus + 1;
            case -1:
                return m_begin;
            default:
                assert(false);
                return 1;
            }
        }

        /**
           End index for the loop on the inside region.
           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS \endlink)
         */
        GT_FUNCTION int_t loop_high_bound_inside(
            int_t I) const { // inside is the fact that the halos are ones outside the begin-end region
            switch (I) {
            case 0:
                return m_end;
            case 1:
                return m_end;
            case -1:
                return m_begin + m_plus - 1;
            default:
                assert(false);
                return 1;
            }
        }

        /**
           receive_length. Return the parameter \f$S_i\f$ as described in \link MULTI_DIM_ACCESS \endlink.

           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS
           \endlink)
        */
        GT_FUNCTION uint_t r_length(int_t I) const {
            switch (I) {
            case 0:
                return (m_end - m_begin + 1);
            case 1:
                return m_plus;
            case -1:
                return m_minus;
            default:
                assert(false);
                return 1;
            }
        }

        /**
           send_length. Return the parameter \f$S_i\f$ as described in \link MULTI_DIM_ACCESS \endlink.

           \param[in] I relative coordinate of the neighbor (the \f$eta\f$ parameter in \link MULTI_DIM_ACCESS
           \endlink)
        */
        GT_FUNCTION uint_t s_length(int_t I) const {
            switch (I) {
            case 0:
                return (m_end - m_begin + 1);
            case -1:
                return m_plus;
            case 1:
                return m_minus;
            default:
                assert(false);
                return 1;
            }
        }

        GT_FUNCTION uint_t minus() const { return m_minus; }
        GT_FUNCTION uint_t plus() const { return m_plus; }
        GT_FUNCTION uint_t begin() const { return m_begin; }
        GT_FUNCTION uint_t end() const { return m_end; }
        GT_FUNCTION uint_t total_length() const { return m_total_length; }

        /**
         * @brief sets minus halo to zero.
         * This operation is needed in the communication module.
         * Unlike a general setter this operation will always result in a valid halo_descriptor.
         */
        GT_FUNCTION void reset_minus() { m_minus = 0; }
        /**
         * @brief sets plus halo to zero.
         * This operation, unlike a general setter, will always result in a valid halo_descriptor.
         */
        GT_FUNCTION void reset_plus() { m_plus = 0; }

        GT_FUNCTION bool operator==(const halo_descriptor &rhs) const {
            return (m_minus == rhs.m_minus) && (m_plus == rhs.m_plus) && (m_begin == rhs.m_begin) &&
                   (m_end == rhs.m_end) && (m_total_length == rhs.m_total_length);
        }

        GT_FUNCTION bool operator!=(const halo_descriptor &rhs) const { return !(*this == rhs); }
    };

    /** @} */
    /** @} */

} // namespace gridtools
