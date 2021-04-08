# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Collection, Dict, List, Set, Union

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import LeafNode
from gtc.common import BuiltInLiteral, DataType, LevelMarker, NativeFunction, UnaryOperator
from gtc.cuir import cuir


class CUIRCodegen(codegen.TemplatedGenerator):

    LocalScalar = as_fmt("{dtype} {name};")

    FieldDecl = as_fmt("{name}")

    ScalarDecl = as_fmt("{name}")

    Temporary = as_fmt("{name}")

    AssignStmt = as_fmt("{left} = {right};")

    FieldAccess = as_mako(
        "*${f'sid::multi_shifted<tag::{name}>({name}, m_strides, {offset})' if offset else name}"
    )

    def visit_IJCacheAccess(
        self, node: cuir.IJCacheAccess, symtable: Dict[str, Any], **kwargs: Any
    ) -> str:
        extent = symtable[node.name].extent
        if extent.i == extent.j == (0, 0):
            # cache is scalar
            assert node.offset.i == node.offset.j == 0
            return node.name
        if node.offset.i == node.offset.j == 0:
            return "*" + node.name
        offsets = (
            f"{o} * {d}_stride_{node.name}"
            for o, d in zip([node.offset.i, node.offset.j], "ij")
            if o != 0
        )
        return node.name + "[" + " + ".join(offsets) + "]"

    KCacheAccess = as_mako("${_this_generator.k_cache_var(name, _this_node.offset.k)}")

    ScalarAccess = as_fmt("{name}")

    CartesianOffset = as_mako(
        "${'' if _this_node.i == _this_node.j == _this_node.k == 0 else f'offsets({i}_c, {j}_c, {k}_c)'}"
    )

    BinaryOp = as_fmt("({left} {op} {right})")

    UNARY_OPERATOR_TO_CODE = {
        UnaryOperator.NOT: "!",
        UnaryOperator.NEG: "-",
        UnaryOperator.POS: "+",
    }

    UnaryOp = as_fmt("({_this_generator.UNARY_OPERATOR_TO_CODE[_this_node.op]}{expr})")

    TernaryOp = as_fmt("({cond} ? {true_expr} : {false_expr})")

    Cast = as_fmt("static_cast<{dtype}>({expr})")

    BUILTIN_LITERAL_TO_CODE = {
        BuiltInLiteral.TRUE: "true",
        BuiltInLiteral.FALSE: "false",
    }

    def visit_BuiltInLiteral(self, builtin: BuiltInLiteral, **kwargs: Any) -> str:
        try:
            return self.BUILTIN_LITERAL_TO_CODE[builtin]
        except KeyError as error:
            raise NotImplementedError("Not implemented BuiltInLiteral encountered.") from error

    Literal = as_mako("static_cast<${dtype}>(${value})")

    NATIVE_FUNCTION_TO_CODE = {
        NativeFunction.ABS: "std::abs",
        NativeFunction.MIN: "std::min",
        NativeFunction.MAX: "std::max",
        NativeFunction.MOD: "std::fmod",
        NativeFunction.SIN: "std::sin",
        NativeFunction.COS: "std::cos",
        NativeFunction.TAN: "std::tan",
        NativeFunction.ARCSIN: "std::asin",
        NativeFunction.ARCCOS: "std::acos",
        NativeFunction.ARCTAN: "std::atan",
        NativeFunction.SQRT: "std::sqrt",
        NativeFunction.POW: "std::pow",
        NativeFunction.EXP: "std::exp",
        NativeFunction.LOG: "std::log",
        NativeFunction.ISFINITE: "std::isfinite",
        NativeFunction.ISINF: "std::isinf",
        NativeFunction.ISNAN: "std::isnan",
        NativeFunction.FLOOR: "std::floor",
        NativeFunction.CEIL: "std::ceil",
        NativeFunction.TRUNC: "std::trunc",
    }

    def visit_NativeFunction(self, func: NativeFunction, **kwargs: Any) -> str:
        try:
            return self.NATIVE_FUNCTION_TO_CODE[func]
        except KeyError as error:
            raise NotImplementedError(
                f"Not implemented NativeFunction '{func}' encountered."
            ) from error

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    DATA_TYPE_TO_CODE = {
        DataType.BOOL: "bool",
        DataType.INT8: "std::int8_t",
        DataType.INT16: "std::int16_t",
        DataType.INT32: "std::int32_t",
        DataType.INT64: "std::int64_t",
        DataType.FLOAT32: "float",
        DataType.FLOAT64: "double",
    }

    def visit_DataType(self, dtype: DataType, **kwargs: Any) -> str:
        try:
            return self.DATA_TYPE_TO_CODE[dtype]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    IJExtent = as_fmt("extent<{i[0]}, {i[1]}, {j[0]}, {j[1]}>")

    HorizontalExecution = as_mako(
        """
        // ${id_}
        if (validator(${extent}())${' && ' + mask if _this_node.mask else ''}) {
            ${'\\n'.join(declarations)}
            ${'\\n'.join(body)}
        }
        """
    )

    def visit_AxisBound(self, node: cuir.AxisBound, **kwargs: Any) -> str:
        if node.level == LevelMarker.START:
            return f"{node.offset}"
        if node.level == LevelMarker.END:
            return f"k_size + {node.offset}"
        raise ValueError("Cannot handle dynamic levels")

    IJCacheDecl = as_mako(
        """
        % if _this_node.extent.i == _this_node.extent.j == (0, 0):
        // scalar ij-cache
        ${dtype} ${name};
        % else:
        // ij-cache in shared memory
        constexpr int ${name}_cache_data_size = (i_block_size_t() + ${-_this_node.extent.i[0] + _this_node.extent.i[1]}) * (j_block_size_t() + ${-_this_node.extent.j[0] + _this_node.extent.j[1]});
        __shared__ ${dtype} ${name}_cache_data[${name}_cache_data_size];
        constexpr int i_stride_${name} = 1;
        constexpr int j_stride_${name} = i_block_size_t() + ${-_this_node.extent.i[0] + _this_node.extent.i[1]};
        ${dtype} *${name} = ${name}_cache_data + (${-_this_node.extent.i[0]} + _i_block) * i_stride_${name} + (${-_this_node.extent.j[0]} + _j_block) * j_stride_${name};
        % endif
        """
    )

    KCacheDecl = as_mako(
        """
        % for var in _this_generator.k_cache_vars(_this_node):
            ${dtype} ${var};
        % endfor
        """
    )

    VerticalLoopSection = as_mako(
        """
        <%def name="sid_shift(step)">
            sid::shift(_ptr, sid::get_stride<dim::k>(m_strides), ${step}_c);
        </%def>
        <%def name="cache_shift(cache_vars)">
            % for dst, src in zip(cache_vars[:-1], cache_vars[1:]):
                ${dst} = ${src};
            % endfor
        </%def>
        // ${id_}
        % if order == cuir.LoopOrder.FORWARD:
            for (int _k_block = ${start}; _k_block < ${end}; ++_k_block) {
                ${'\\n__syncthreads();\\n'.join(horizontal_executions)}

                ${sid_shift(1)}
                % for k_cache in k_cache_decls:
                    ${cache_shift(_this_generator.k_cache_vars(k_cache))}
                % endfor
            }
        % elif order == cuir.LoopOrder.BACKWARD:
            for (int _k_block = ${end} - 1; _k_block >= ${start}; --_k_block) {
                ${'\\n__syncthreads();\\n'.join(horizontal_executions)}

                ${sid_shift(-1)}
                % for k_cache in k_cache_decls:
                    ${cache_shift(_this_generator.k_cache_vars(k_cache)[::-1])}
                % endfor
            }
        % else:
            if (_k_block >= ${start} && _k_block < ${end}) {
                ${'\\n__syncthreads();\\n'.join(horizontal_executions)}
            }
        % endif
        """
    )

    @staticmethod
    def k_cache_var(name: str, offset: int) -> str:
        return name + (f"p{offset}" if offset >= 0 else f"m{-offset}")

    @classmethod
    def k_cache_vars(cls, k_cache: cuir.KCacheDecl) -> List[str]:
        assert k_cache.extent
        return [
            cls.k_cache_var(k_cache.name, offset)
            for offset in range(k_cache.extent.k[0], k_cache.extent.k[1] + 1)
        ]

    def visit_VerticalLoop(
        self, node: cuir.VerticalLoop, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Collection[str]]:

        return self.generic_visit(
            node,
            fields=node.iter_tree().if_isinstance(cuir.FieldAccess).getattr("name").to_set(),
            k_cache_decls=node.k_caches,
            order=node.loop_order,
            symtable=symtable,
            **kwargs,
        )

    VerticalLoop = as_mako(
        """
        template <class Sid>
        struct loop_${id_}_f {
            sid::ptr_holder_type<Sid> m_ptr_holder;
            sid::strides_type<Sid> m_strides;
            int k_size;

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(const int _i_block,
                                               const int _j_block,
                                               Validator validator) const {
                auto _ptr = m_ptr_holder();
                sid::shift(_ptr,
                           sid::get_stride<sid::blocked_dim<dim::i>>(m_strides),
                           blockIdx.x);
                sid::shift(_ptr,
                           sid::get_stride<sid::blocked_dim<dim::j>>(m_strides),
                           blockIdx.y);
                sid::shift(_ptr,
                           sid::get_stride<dim::i>(m_strides),
                           _i_block);
                sid::shift(_ptr,
                           sid::get_stride<dim::j>(m_strides),
                           _j_block);
                % if order == cuir.LoopOrder.PARALLEL:
                const int _k_block = blockIdx.z;
                sid::shift(_ptr,
                           sid::get_stride<dim::k>(m_strides),
                           _k_block);
                % endif

                % for field in fields:
                    auto &&${field} = device::at_key<tag::${field}>(_ptr);
                % endfor

                % for ij_cache in ij_caches:
                    ${ij_cache}
                % endfor

                % for k_cache in k_caches:
                    ${k_cache}
                % endfor

                % for section in sections:
                    ${section}
                % endfor
            }
        };
        """
    )

    Kernel = as_mako(
        """
        % for vertical_loop in vertical_loops:
            ${vertical_loop}
        % endfor

        template <${', '.join(f'class Loop{vl.id_}' for vl in _this_node.vertical_loops)}>
        struct kernel_${id_}_f {
            % for vertical_loop in _this_node.vertical_loops:
                Loop${vertical_loop.id_} m_${vertical_loop.id_};
            % endfor

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(const int _i_block,
                                               const int _j_block,
                                               Validator validator) const {
                % for vertical_loop in _this_node.vertical_loops:
                    m_${vertical_loop.id_}(_i_block, _j_block, validator);
                % endfor
            }
        };

        """
    )

    def visit_Program(self, node: cuir.Program, **kwargs: Any) -> Union[str, Collection[str]]:
        def loop_start(vertical_loop: cuir.VerticalLoop) -> str:
            if vertical_loop.loop_order == cuir.LoopOrder.FORWARD:
                return self.visit(vertical_loop.sections[0].start, **kwargs)
            if vertical_loop.loop_order == cuir.LoopOrder.BACKWARD:
                return self.visit(vertical_loop.sections[0].end, **kwargs) + " - 1"
            return "0"

        def loop_fields(vertical_loop: cuir.VerticalLoop) -> Set[str]:
            return (
                vertical_loop.iter_tree().if_isinstance(cuir.FieldAccess).getattr("name").to_set()
            )

        def ctype(symbol: str) -> str:
            return self.visit(node.symtable_[symbol].dtype, **kwargs)

        return self.generic_visit(
            node,
            max_extent=self.visit(
                cuir.IJExtent.zero().union(*node.iter_tree().if_isinstance(cuir.IJExtent)), **kwargs
            ),
            loop_start=loop_start,
            loop_fields=loop_fields,
            ctype=ctype,
            symtable=node.symtable_,
            cuir=cuir,
            **kwargs,
        )

    Program = as_mako(
        """#include <algorithm>
        #include <array>
        #include <cstdint>
        #include <gridtools/common/cuda_util.hpp>
        #include <gridtools/common/host_device.hpp>
        #include <gridtools/common/hymap.hpp>
        #include <gridtools/common/integral_constant.hpp>
        #include <gridtools/sid/allocator.hpp>
        #include <gridtools/sid/block.hpp>
        #include <gridtools/sid/composite.hpp>
        #include <gridtools/sid/multi_shift.hpp>
        #include <gridtools/stencil/common/dim.hpp>
        #include <gridtools/stencil/common/extent.hpp>
        #include <gridtools/stencil/gpu/launch_kernel.hpp>
        #include <gridtools/stencil/gpu/shared_allocator.hpp>
        #include <gridtools/stencil/gpu/tmp_storage_sid.hpp>

        namespace ${name}_impl_{
            using namespace gridtools;
            using namespace literals;
            using namespace stencil;

            using domain_t = std::array<unsigned, 3>;
            using i_block_size_t = integral_constant<int, 64>;
            using j_block_size_t = integral_constant<int, 8>;

            template <class Storage>
            auto block(Storage storage) {
                return sid::block(std::move(storage),
                    tuple_util::make<hymap::keys<dim::i, dim::j>::values>(
                        i_block_size_t(), j_block_size_t()));
            }

            template <class I, class J, class K>
            GT_FUNCTION_DEVICE auto offsets(I i, J j, K k) {
                return tuple_util::device::make<hymap::keys<dim::i, dim::j, dim::k>::template values>(i, j, k);
            }

            namespace tag {
                % for p in set().union(*(loop_fields(v) for k in _this_node.kernels for v in k.vertical_loops)):
                    struct ${p} {};
                % endfor
            }

            % for kernel in kernels:
                ${kernel}
            % endfor

            auto ${name}(domain_t domain){
                return [domain](${','.join(f'auto&& {p}' for p in params)}){
                    auto tmp_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char[]>);
                    const int i_size = domain[0];
                    const int j_size = domain[1];
                    const int k_size = domain[2];
                    const int i_blocks = (i_size + i_block_size_t() - 1) / i_block_size_t();
                    const int j_blocks = (j_size + j_block_size_t() - 1) / j_block_size_t();

                    % for tmp in temporaries:
                        auto ${tmp} = gpu_backend::make_tmp_storage<${ctype(tmp)}>(
                            1_c,
                            i_block_size_t(),
                            j_block_size_t(),
                            ${max_extent}(),
                            i_blocks,
                            j_blocks,
                            k_size,
                            tmp_alloc);
                    % endfor

                    % for kernel in _this_node.kernels:

                        // kernel ${kernel.id_}
                        gpu_backend::shared_allocator shared_alloc_${kernel.id_};

                        % for vertical_loop in kernel.vertical_loops:
                            // vertical loop ${vertical_loop.id_}

                            assert((${loop_start(vertical_loop)}) >= 0 &&
                                   (${loop_start(vertical_loop)}) < k_size);
                            auto offset_${vertical_loop.id_} = tuple_util::make<hymap::keys<dim::k>::values>(
                                ${loop_start(vertical_loop)}
                            );

                            auto composite_${vertical_loop.id_} = sid::composite::make<
                                    ${', '.join(f'tag::{field}' for field in loop_fields(vertical_loop))}
                                >(

                            % for field in loop_fields(vertical_loop):
                                % if field in params:
                                    block(sid::shift_sid_origin(
                                        ${field},
                                        offset_${vertical_loop.id_}
                                    ))
                                % else:
                                    sid::shift_sid_origin(
                                        ${field},
                                        offset_${vertical_loop.id_}
                                    )
                                % endif
                                ${'' if loop.last else ','}
                            % endfor
                            );
                            using composite_${vertical_loop.id_}_t = decltype(composite_${vertical_loop.id_});
                            loop_${vertical_loop.id_}_f<composite_${vertical_loop.id_}_t> loop_${vertical_loop.id_}{
                                sid::get_origin(composite_${vertical_loop.id_}),
                                sid::get_strides(composite_${vertical_loop.id_}),
                                k_size
                            };

                        % endfor

                        kernel_${kernel.id_}_f<${', '.join(f'decltype(loop_{vl.id_})' for vl in kernel.vertical_loops)}> kernel_${kernel.id_}{
                            ${', '.join(f'loop_{vl.id_}' for vl in kernel.vertical_loops)}
                        };
                        gpu_backend::launch_kernel<${max_extent},
                            i_block_size_t::value, j_block_size_t::value>(
                            i_size,
                            j_size,
                            % if kernel.vertical_loops[0].loop_order == cuir.LoopOrder.PARALLEL:
                                k_size,
                            % else:
                                1,
                            %endif
                            kernel_${kernel.id_},
                            shared_alloc_${kernel.id_}.size());
                    % endfor

                    GT_CUDA_CHECK(cudaDeviceSynchronize());
                };
            }
        }

        using ${name}_impl_::${name};
        """
    )

    @classmethod
    def apply(cls, root: LeafNode, **kwargs: Any) -> str:
        if not isinstance(root, cuir.Program):
            raise ValueError("apply() requires gtcpp.Progam root node")
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code
