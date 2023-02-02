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

#include <type_traits>

#include <cpp_bindgen/fortran_array_view.hpp>

#include "../../layout_transformation.hpp"
#include "../data_store.hpp"

namespace gridtools {
    template <class DataStorePtr>
    class fortran_array_adapter {
        static_assert(storage::is_data_store_ptr<DataStorePtr>::value);
        using data_store_t = typename DataStorePtr::element_type;
        using lengths_t = std::decay_t<decltype(DataStorePtr()->lengths())>;
        using strides_t = std::decay_t<decltype(DataStorePtr()->strides())>;
        using data_ptr_t = decltype(DataStorePtr()->get_target_ptr());

        bindgen_fortran_array_descriptor const &m_descriptor;

        data_ptr_t fortran_ptr() const {
            assert(m_descriptor.data);
            return static_cast<data_ptr_t>(m_descriptor.data);
        }

        // verify dimensions of fortran array
        void check_fortran_lengths(DataStorePtr const &ds) const {
            auto &&lengths = ds->lengths();
            auto &&strides = ds->strides();
            for (size_t c_dim = 0, fortran_dim = 0; c_dim < data_store_t::layout_t::masked_length; ++c_dim)
                if (strides[c_dim] != 0) {
                    if (m_descriptor.dims[fortran_dim] != lengths[c_dim])
                        throw std::runtime_error("dimensions do not match (descriptor [" +
                                                 std::to_string(m_descriptor.dims[fortran_dim]) + "] != data_store [" +
                                                 std::to_string(lengths[c_dim]) + "])");
                    ++fortran_dim;
                }
        }

        strides_t fortran_strides(DataStorePtr const &ds) const {
            auto &&lengths = ds->lengths();
            auto &&strides = ds->strides();
            strides_t res = {};
            uint_t current_stride = 1;
            for (size_t i = 0; i < res.size(); ++i)
                if (strides[i] != 0) {
                    res[i] = current_stride;
                    current_stride *= lengths[i];
                }
            return res;
        }

      public:
        fortran_array_adapter(const bindgen_fortran_array_descriptor &descriptor) : m_descriptor(descriptor) {
            if (m_descriptor.rank != bindgen_view_rank::value)
                throw std::runtime_error("rank does not match (descriptor-rank [" + std::to_string(m_descriptor.rank) +
                                         "] != datastore-rank [" + std::to_string(bindgen_view_rank::value) + "]");
        }

        using bindgen_view_rank = std::integral_constant<size_t, data_store_t::layout_t::unmasked_length>;
        using bindgen_view_element_type = std::remove_pointer_t<data_ptr_t>;
        using bindgen_is_acc_present = std::true_type;

        void transform_to(DataStorePtr const &dst) const {
            check_fortran_lengths(dst);
            transform_layout(
                dst->get_target_ptr(), fortran_ptr(), dst->lengths(), dst->strides(), fortran_strides(dst));
        }

        void transform_from(DataStorePtr const &src) const {
            check_fortran_lengths(src);
            transform_layout(
                fortran_ptr(), src->get_target_ptr(), src->lengths(), fortran_strides(src), src->strides());
        }
    };
} // namespace gridtools
