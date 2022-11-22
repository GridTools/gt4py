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

import numpy as np

import eve
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import LeafNode
from gtc.common import BuiltInLiteral, DataType, LevelMarker, NativeFunction, UnaryOperator
from gtc.cuir import cuir


class CUIRCodegen(codegen.TemplatedGenerator, eve.VisitorWithSymbolTableTrait):

    LocalScalar = as_fmt("{dtype} {name};")

    FieldDecl = as_fmt("{name}")

    ScalarDecl = as_fmt("{name}")

    Temporary = as_fmt("{name}")

    AssignStmt = as_fmt("{left} = {right};")

    MaskStmt = as_mako(
        """
        if (${mask}) {
            ${'\\n'.join(body)}
        }
        """
    )

    While = as_mako(
        """
        while (${cond}) {
            ${'\\n'.join(body)}
        }
        """
    )

    def visit_FieldAccess(self, node: cuir.FieldAccess, **kwargs: Any):
        if isinstance(node, cuir.KCacheAccess):
            return self.generic_visit(node, **kwargs)

        symtable: Dict[str, cuir.Decl] = kwargs["symtable"]

        def maybe_const(s):
            try:
                return f"{int(s)}_c"
            except ValueError:
                return s

        name = self.visit(node.name, **kwargs)
        offset = self.visit(node.offset, **kwargs)
        data_index = [self.visit(index, in_data_index=True, **kwargs) for index in node.data_index]

        decl = symtable[node.name]
        if isinstance(decl, cuir.Temporary) and decl.data_dims:
            data_index_str = "+".join(
                f"{index}*{int(np.prod(decl.data_dims[i + 1:], initial=1))}"
                for i, index in enumerate(data_index)
            )
            return f"{name}({offset})[{data_index_str}]"
        else:
            data_index_str = "".join(f", {maybe_const(index)}" for index in data_index)
            return f"{name}({offset}{data_index_str})"

    def visit_IJCacheAccess(
        self, node: cuir.IJCacheAccess, symtable: Dict[str, Any], **kwargs: Any
    ) -> str:
        decl = symtable[node.name]
        assert isinstance(decl, cuir.IJCacheDecl)
        extent = decl.extent
        assert extent is not None
        offsets = node.offset.to_dict()
        if extent.i == extent.j == (0, 0):
            # cache is scalar
            assert offsets["i"] == offsets["j"] == 0
            return node.name
        if offsets["i"] == offsets["j"] == 0:
            return "*" + node.name
        off = (
            f"{o} * {d}_stride_{node.name}"
            for o, d in zip((offsets["i"], offsets["j"]), "ij")
            if o != 0
        )
        return node.name + "[" + " + ".join(off) + "]"

    KCacheAccess = as_mako("${_this_generator.k_cache_var(name, _this_node.offset.k)}")

    ScalarAccess = as_fmt("{name}")

    CartesianOffset = as_fmt("{i}_c, {j}_c, {k}_c")

    VariableKOffset = as_fmt("0_c, 0_c, {k}")

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

    def visit_Literal(
        self, node: cuir.Literal, *, in_data_index: bool = False, **kwargs: Any
    ) -> str:
        value = self.visit(node.value, **kwargs)
        if in_data_index:
            return value
        else:
            dtype = self.visit(node.dtype, **kwargs)
            return f"static_cast<{dtype}>({value})"

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
        NativeFunction.SINH: "std::sinh",
        NativeFunction.COSH: "std::cosh",
        NativeFunction.TANH: "std::tanh",
        NativeFunction.ARCSINH: "std::asinh",
        NativeFunction.ARCCOSH: "std::acosh",
        NativeFunction.ARCTANH: "std::atanh",
        NativeFunction.SQRT: "std::sqrt",
        NativeFunction.POW: "std::pow",
        NativeFunction.EXP: "std::exp",
        NativeFunction.LOG: "std::log",
        NativeFunction.GAMMA: "std::tgamma",
        NativeFunction.CBRT: "std::cbrt",
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
            raise NotImplementedError(
                f"Not implemented DataType '{dtype.name}' encountered."
            ) from error

    IJExtent = as_fmt("extent<{i[0]}, {i[1]}, {j[0]}, {j[1]}>")

    HorizontalExecution = as_mako(
        """
        // HorizontalExecution ${id(_this_node)}
        if (validator(${extent}())) {
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
        // VerticalLoopSection ${id(_this_node)}
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

        fields = {
            name: data_dims
            for name, data_dims in node.walk_values()
            .if_isinstance(cuir.FieldAccess)
            .getattr("name", "data_index")
            .map(lambda x: (x[0], len(x[1])))
        }

        return self.generic_visit(
            node,
            fields=fields,
            k_cache_decls=node.k_caches,
            order=node.loop_order,
            symtable=symtable,
            **kwargs,
        )

    VerticalLoop = as_mako(
        """
        template <class Sid>
        struct loop_${id(_this_node)}_f {
            sid::ptr_holder_type<Sid> m_ptr_holder;
            sid::strides_type<Sid> m_strides;
            int i_size;
            int j_size;
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

                % for field, data_dims in fields.items():
                const auto ${field} = [&](auto i, auto j, auto k
                    % if field not in temp_names:
                    % for i in range(data_dims):
                    , auto dim_${i + 3}
                    % endfor
                    % endif
                    ) -> auto&& {
                    return *sid::multi_shifted<tag::${field}>(
                        device::at_key<tag::${field}>(_ptr),
                        m_strides,
                        hymap::keys<dim::i, dim::j, dim::k
                        % if field not in temp_names:
                        % for i in range(data_dims):
                        , integral_constant<int, ${i + 3}>
                        % endfor
                        % endif
                        >::make_values(i, j, k
                        % if field not in temp_names:
                        % for i in range(data_dims):
                        , dim_${i + 3}
                        % endfor
                        % endif
                        ));
                };
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

        template <${', '.join(f'class Loop{id(vl)}' for vl in _this_node.vertical_loops)}>
        struct kernel_${id(_this_node)}_f {
            % for vertical_loop in _this_node.vertical_loops:
            Loop${id(vertical_loop)} m_${id(vertical_loop)};
            % endfor

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(const int _i_block,
                                               const int _j_block,
                                               Validator validator) const {
                % for vertical_loop in _this_node.vertical_loops:
                m_${id(vertical_loop)}(_i_block, _j_block, validator);
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
                vertical_loop.walk_values().if_isinstance(cuir.FieldAccess).getattr("name").to_set()
            )

        def ctype(symbol: str) -> str:
            decl = kwargs["symtable"][symbol]
            dtype = self.visit(decl.dtype, **kwargs)
            if decl.data_dims:
                total_size = int(np.prod(decl.data_dims, initial=1))
                dtype = f"array<{dtype}, {total_size}>"
            return dtype

        return self.generic_visit(
            node,
            max_extent=self.visit(
                cuir.IJExtent.zero().union(*node.walk_values().if_isinstance(cuir.IJExtent)),
                **kwargs,
            ),
            loop_start=loop_start,
            loop_fields=loop_fields,
            ctype=ctype,
            cuir=cuir,
            temp_names={decl.name for decl in node.temporaries},
            **kwargs,
        )

    Positional = as_fmt("auto {name} = positional<dim::{axis_name}>();")

    Program = as_mako(
        """#include <algorithm>
        #include <array>
        #include <cstdint>
        #include <gridtools/common/array.hpp>
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
        #include <gridtools/stencil/gpu/tmp_storage_sid.hpp>
        % if positionals:
        #include <gridtools/stencil/positional.hpp>
        % endif

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
                    hymap::keys<dim::i, dim::j>::make_values(
                        i_block_size_t(), j_block_size_t()));
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
                    auto tmp_alloc = sid::device::cached_allocator(&cuda_util::cuda_malloc<char[]>);
                    const int i_size = domain[0];
                    const int j_size = domain[1];
                    const int k_size = domain[2];
                    const int i_blocks = (i_size + i_block_size_t() - 1) / i_block_size_t();
                    const int j_blocks = (j_size + j_block_size_t() - 1) / j_block_size_t();

                    % for decl in positionals:
                    ${decl}
                    % endfor

                    % for tmp in temporaries:
                    auto ${tmp} = gpu_backend::make_tmp_storage<${ctype(tmp)}>(
                        1_c,
                        i_block_size_t(),
                        j_block_size_t(),
                        ${max_extent}(),
                        i_blocks,
                        j_blocks,
                        k_size,
                        tmp_alloc
                    );
                    % endfor

                    % for kernel in _this_node.kernels:

                    // kernel ${id(kernel)}

                    % for vertical_loop in kernel.vertical_loops:
                    // vertical loop ${id(vertical_loop)}

                    assert((${loop_start(vertical_loop)}) >= 0 &&
                           (${loop_start(vertical_loop)}) < k_size);
                    auto offset_${id(vertical_loop)} = hymap::keys<dim::k>::make_values(
                        ${loop_start(vertical_loop)}
                    );

                    auto composite_${id(vertical_loop)} = sid::composite::keys<
                            ${', '.join(f'tag::{field}' for field in loop_fields(vertical_loop))}
                        >::make_values(

                    % for field in loop_fields(vertical_loop):
                        % if field in params:
                            block(sid::shift_sid_origin(
                                ${field},
                                offset_${id(vertical_loop)}
                            ))
                        % else:
                            sid::shift_sid_origin(
                                ${field},
                                offset_${id(vertical_loop)}
                            )
                        % endif
                        ${'' if loop.last else ','}
                    % endfor
                    );
                    using composite_${id(vertical_loop)}_t = decltype(composite_${id(vertical_loop)});
                    loop_${id(vertical_loop)}_f<composite_${id(vertical_loop)}_t> loop_${id(vertical_loop)}{
                        sid::get_origin(composite_${id(vertical_loop)}),
                        sid::get_strides(composite_${id(vertical_loop)}),
                        i_size,
                        j_size,
                        k_size
                    };

                    % endfor

                    kernel_${id(kernel)}_f<${', '.join(f'decltype(loop_{id(vl)})' for vl in kernel.vertical_loops)}> kernel_${id(kernel)}{
                        ${', '.join(f'loop_{id(vl)}' for vl in kernel.vertical_loops)}
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
                        kernel_${id(kernel)},
                        0);
                    % endfor
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
        if kwargs.get("format_source", True):
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code
