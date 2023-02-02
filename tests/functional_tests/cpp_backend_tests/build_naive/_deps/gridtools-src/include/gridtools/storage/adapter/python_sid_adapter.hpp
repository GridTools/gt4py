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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <pybind11/pybind11.h>

#include "../../common/array.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/if.hpp"
#include "../../sid/simple_ptr_holder.hpp"
#include "../../sid/synthetic.hpp"
#include "../../sid/unknown_kind.hpp"

namespace gridtools {
    namespace python_sid_adapter_impl_ {
        template <size_t, class>
        struct kind {};

        template <size_t UnitStrideDim>
        struct transform_strides_f {
            bool m_unit_stride_can_be_used;
            template <size_t I, class T, std::enable_if_t<I == UnitStrideDim, int> = 0>
            integral_constant<pybind11::ssize_t, 1> operator()(T val) const {
                if (m_unit_stride_can_be_used && val != 1)
                    throw std::domain_error("incompatible strides, expected unit stride");
                return {};
            }

            template <size_t I, class T, std::enable_if_t<I != UnitStrideDim, int> = 0>
            T operator()(T val) const {
                return val;
            }
        };

        template <size_t UnitStrideDim,
            class Strides,
            class Shape,
            std::enable_if_t<(UnitStrideDim >= tuple_util::size<std::decay_t<Strides>>::value), int> = 0>
        Strides &&assign_unit_stride(Strides &&strides, Shape &&) {
            return std::forward<Strides>(strides);
        }

        template <size_t UnitStrideDim,
            class Strides,
            class Shape,
            std::enable_if_t<(UnitStrideDim < tuple_util::size<std::decay_t<Strides>>::value), int> = 0>
        decltype(auto) assign_unit_stride(Strides &&strides, Shape &&shape) {
            // Numpy may shuffle array layout if the array sizes are equal to one in some dimensions.
            // In this case the static unit stride calculation doesn't match the actual strides.
            // Luckily we can ignore that because those strides will not be used (corresponding shape == 1).
            // See: https://numpy.org/devdocs/release/1.8.0-notes.html#npy-relaxed-strides-checking
            bool unit_stride_can_be_used = tuple_util::get<UnitStrideDim>(std::forward<Shape>(shape)) > 1;
            return tuple_util::transform_index(transform_strides_f<UnitStrideDim>{unit_stride_can_be_used},
                tuple_util::convert_to<tuple>(std::forward<Strides>(strides)));
        }

        template <class T, size_t Dim, class Kind, size_t UnitStrideDim>
        struct wrapper {
            pybind11::buffer_info m_info;

            friend sid::simple_ptr_holder<T *> sid_get_origin(wrapper const &obj) {
                return {reinterpret_cast<T *>(obj.m_info.ptr)};
            }
            friend auto sid_get_strides(wrapper const &obj) {
                std::array<pybind11::ssize_t, Dim> res;
                assert(obj.m_info.strides.size() == Dim);
                for (std::size_t i = 0; i != Dim; ++i) {
                    assert(obj.m_info.strides[i] % obj.m_info.itemsize == 0);
                    res[i] = obj.m_info.strides[i] / obj.m_info.itemsize;
                }
                return assign_unit_stride<UnitStrideDim>(std::move(res), sid_get_upper_bounds(obj));
            }

            friend std::array<integral_constant<pybind11::ssize_t, 0>, Dim> sid_get_lower_bounds(wrapper const &) {
                return {};
            }
            friend std::array<pybind11::ssize_t, Dim> sid_get_upper_bounds(wrapper const &obj) {
                std::array<pybind11::ssize_t, Dim> res;
                assert(obj.m_info.shape.size() == Dim);
                for (std::size_t i = 0; i != Dim; ++i) {
                    assert(obj.m_info.shape[i] > 0);
                    res[i] = obj.m_info.shape[i];
                }
                return res;
            }
            friend meta::if_<std::is_same<Kind, sid::unknown_kind>, Kind, kind<Dim, Kind>> sid_get_strides_kind(
                wrapper const &) {
                return {};
            }
        };

        template <class T, std::size_t Dim, class Kind = void, size_t UnitStrideDim = size_t(-1)>
        wrapper<T, Dim, Kind, UnitStrideDim> as_sid(pybind11::buffer const &src) {
            static_assert(std::is_trivially_copy_constructible_v<T>,
                "as_sid should be instantiated with the trivially copyable type");
            constexpr bool writable = !std::is_const<T>();
            // pybind11::buffer::request accepts writable as an optional parameter (default is false).
            // if writable is true PyBUF_WRITABLE flag is added while delegating to the PyObject_GetBuffer.
            auto info = src.request(writable);
            assert(!(writable && info.readonly));
            if (info.ndim != Dim)
                throw std::domain_error("buffer has incorrect number of dimensions: " + std::to_string(info.ndim) +
                                        "; expected " + std::to_string(Dim));
            if (info.itemsize != sizeof(T))
                throw std::domain_error("buffer has incorrect itemsize: " + std::to_string(info.itemsize) +
                                        "; expected " + std::to_string(sizeof(T)));

            using format_desc_t = pybind11::format_descriptor<std::remove_const_t<T>>;
            auto expected_format = format_desc_t::format();
            assert(info.format.size() == 1 && expected_format.size() == 1);
            const char *int_formats = "bBhHiIlLqQnN";
            const char *int_char = std::strchr(int_formats, info.format[0]);
            const char *expected_int_char = std::strchr(int_formats, expected_format[0]);
            if (int_char && expected_int_char) {
                // just check upper/lower case for integer formats which indicates signedness (itemsize already checked)
                // direct format comparison in not enough, for details see
                // https://github.com/pybind/pybind11/issues/1806 and https://github.com/pybind/pybind11/issues/1908
                if (std::islower(*int_char) != std::islower(*expected_int_char))
                    throw std::domain_error("incompatible integer formats: " + info.format + " and " + expected_format);
            } else if (info.format != expected_format) {
                throw std::domain_error(
                    "buffer has incorrect format: " + info.format + "; expected " + expected_format);
            }
            return {std::move(info)};
        }

        struct typestr {
            enum byteorder_type { little_endian = '<', big_endian = '>', not_relevant = '|' };
            enum type_type {
                bit_field = 't',
                boolean = 'b',
                integer = 'i',
                unsigned_integer = 'u',
                floating_point = 'f',
                complex_floating_point = 'c',
                timedelta = 'm',
                datetime = 'M',
                object = 'O',
                string = 'S',
                unicode_string = 'U',
                other = 'V'
            };
            byteorder_type byteorder;
            type_type type;
            int size;
        };

        inline typestr parse_typestr(std::string const &src) {
            if (src.size() < 3)
                throw std::domain_error("invalid typestr: " + src);
            typestr res = {static_cast<typestr::byteorder_type>(src[0]),
                static_cast<typestr::type_type>(src[1]),
                std::atoi(src.substr(2).c_str())};
            switch (res.byteorder) {
            case typestr::little_endian:
            case typestr::big_endian:
            case typestr::not_relevant:
                break;
            default:
                throw std::domain_error("invalid typestr: " + src);
            }
            switch (res.type) {
            case typestr::bit_field:
            case typestr::boolean:
            case typestr::integer:
            case typestr::unsigned_integer:
            case typestr::floating_point:
            case typestr::complex_floating_point:
            case typestr::timedelta:
            case typestr::datetime:
            case typestr::object:
            case typestr::string:
            case typestr::unicode_string:
            case typestr::other:
                break;
            default:
                throw std::domain_error("invalid typestr: " + src);
            }
            return res;
        }

        struct descr_node;
        using descr = std::vector<std::shared_ptr<descr_node>>;

        struct descr_node {
            std::string name;
            std::string basic_name;
            std::variant<typestr, descr> type;
            std::vector<size_t> shape;
        };

        descr parse_descr(pybind11::list const &src);

        inline std::shared_ptr<descr_node> parse_descr_node(pybind11::tuple const &src) {
            descr_node dst;
            if (src.size() < 2)
                throw std::domain_error(pybind11::str("unexpected descr item: {}").format(src));
            if (pybind11::isinstance<pybind11::tuple>(src[0]))
                std::tie(dst.name, dst.basic_name) = src[0].cast<std::tuple<std::string, std::string>>();
            else
                dst.name = src[0].cast<std::string>();
            if (pybind11::isinstance<pybind11::list>(src[1]))
                dst.type = parse_descr(src.cast<pybind11::list>());
            else
                dst.type = parse_typestr(src[1].cast<std::string>());
            if (src.size() == 2) {
                dst.shape.push_back(1);
            } else {
                if (src.size() != 3)
                    throw std::domain_error(pybind11::str("unexpected descr item: {}").format(src));
                auto py_shape = src[3].cast<pybind11::tuple>();
                std::transform(py_shape.begin(), py_shape.end(), std::back_inserter(dst.shape), [](auto const &src) {
                    return src.template cast<size_t>();
                });
            }
            return std::make_shared<descr_node>(dst);
        }

        inline descr parse_descr(pybind11::list const &src) {
            descr res;
            std::transform(src.begin(), src.end(), std::back_inserter(res), [](auto const &src) {
                return parse_descr_node(src.template cast<pybind11::tuple>());
            });
            return res;
        }

        size_t size(descr const &);

        inline size_t size(typestr const &src) { return src.size; }

        inline size_t size(descr_node const &src) {
            size_t res = std::visit([](auto const &src) { return size(src); }, src.type);
            for (auto &&d : src.shape)
                res *= d;
            return res;
        }

        inline size_t size(descr const &src) {
            size_t res = 0;
            for (auto &&item : src)
                res += size(*item);
            return res;
        }

        template <class T, size_t Dim, class Kind = void, size_t UnitStrideDim = size_t(-1)>
        auto as_cuda_sid(pybind11::object const &src) {
            static_assert(std::is_trivially_copy_constructible_v<T>,
                "as_cuda_sid should be instantiated with the trivially copyable type");

            auto iface = src.attr("__cuda_array_interface__").cast<pybind11::dict>();

            // shape
            array<size_t, Dim> shape;
            {
                auto py_shape = iface["shape"].cast<pybind11::tuple>();
                if (py_shape.size() != Dim)
                    throw std::domain_error("cuda array has incorrect number of dimensions: " +
                                            std::to_string(py_shape.size()) + "; expected " + std::to_string(Dim));
                std::transform(py_shape.begin(), py_shape.end(), shape.begin(), [](auto &&src) {
                    return src.template cast<size_t>();
                });
            }

            // typestr
            {
                auto obj = parse_typestr(iface["typestr"].cast<std::string>());
                if (obj.type == typestr::bit_field)
                    throw std::domain_error("bit fields are not supported");
                // any other types are OK for us as soon as the sizeof is correct
                if (obj.size != sizeof(T))
                    throw std::domain_error("invalid item size");
            }

            // data
            auto data = iface["data"].cast<std::tuple<std::size_t, bool>>();
            T *ptr = reinterpret_cast<T *>(std::get<0>(data));
            constexpr bool writable = !std::is_const<T>();
            bool readonly = std::get<1>(data);
            assert(!writable || !readonly);

            // version
            auto version = iface["version"].cast<int>();

            // strides
            array<size_t, Dim> strides;
            if (version > 1 && iface.contains("strides") && !iface["strides"].is_none()) {
                auto py_strides = iface["strides"].cast<pybind11::tuple>();
                if (py_strides.size() != Dim)
                    throw std::domain_error("cuda array has incorrect number of strides: " +
                                            std::to_string(py_strides.size()) + "; expected " + std::to_string(Dim));
                std::transform(py_strides.begin(), py_strides.end(), strides.begin(), [](auto &&src) {
                    auto bytes = src.template cast<size_t>();
                    assert(bytes % sizeof(T) == 0);
                    return bytes / sizeof(T);
                });
            } else {
                size_t s = 1;
                for (int i = Dim - 1; i >= 0; --i) {
                    strides[i] = s;
                    s *= shape[i];
                }
            }

            // descr
            if (iface.contains("descr")) {
                // descr us not used.
                // We just fully parse it and ensure that the total size EQUALS to sizeof(T) as is prescribed
                // by the spec. I think (anstaf) it is an error in the spec. Total size should be LESS OR EQUAL
                // to sizeof(T) because of alignment.
                auto descr = parse_descr(iface["descr"].cast<pybind11::list>());
                if (size(descr) != sizeof(T))
                    throw std::domain_error(pybind11::str("invalid descr total size: {}").format(iface["descr"]));
            }

            // mask
            if (version > 0 && iface.contains("mask") && !iface["mask"].is_none())
                throw std::domain_error("__cuda_array_interface__.mask is not supported.");

            using sid::property;
            return sid::synthetic()
                .template set<property::origin>(sid::host_device::simple_ptr_holder<T *>{ptr})
                .template set<property::strides>(assign_unit_stride<UnitStrideDim>(std::move(strides), shape))
                .template set<property::strides_kind, kind<Dim, Kind>>()
                .template set<property::lower_bounds>(array<integral_constant<size_t, 0>, Dim>())
                .template set<property::upper_bounds>(shape);
        }
    } // namespace python_sid_adapter_impl_

    // Makes a SID from the `pybind11::buffer`.
    // Be aware that the return value is a move only object
    using python_sid_adapter_impl_::as_sid;

    using python_sid_adapter_impl_::as_cuda_sid;
} // namespace gridtools
