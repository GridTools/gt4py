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

import textwrap
from dataclasses import dataclass, field
from typing import Any, Collection, Optional, Set, Tuple, Union

from eve import SymbolTableTrait
from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator
from gtc import common
from gtc.numpy import npir


__all__ = ["NpirCodegen"]


def _dump_sequence(sequence, *, separator=", ", start="(", end=")") -> str:
    return f"{start}{separator.join(sequence)}{end}"


def _slice_string(ch: str, offset: int) -> str:
    return f"{ch}{offset:+d}:{ch.upper()}{offset:+d}" if offset != 0 else f"{ch}:{ch.upper()}"


ORIGIN_CORRECTED_VIEW_CLASS = textwrap.dedent(
    """\
    class Field:
        def __init__(self, field, offsets: Tuple[int, ...], dimensions: Tuple[bool, bool, bool]):
            ii = iter(range(3))
            self.idx_to_data = tuple(
                [next(ii) if has_dim else None for has_dim in dimensions]
                + list(range(sum(dimensions), len(field.shape)))
            )

            shape = [field.shape[i] if i is not None else 1 for i in self.idx_to_data]
            self.field_view = np.reshape(field.data, shape).view(np.ndarray)

            self.offsets = offsets

        @classmethod
        def empty(cls, shape, offset):
            return cls(np.empty(shape), offset, (True, True, True))

        def shim_key(self, key):
            new_args = []
            if not isinstance(key, tuple):
                key = (key, )
            for index in self.idx_to_data:
                if index is None:
                    new_args.append(slice(None, None))
                else:
                    idx = key[index]
                    offset = self.offsets[index]
                    if isinstance(idx, slice):
                        new_args.append(
                            slice(idx.start + offset, idx.stop + offset, idx.step) if offset else idx
                        )
                    else:
                        new_args.append(idx + offset)
            if not isinstance(new_args[2], (numbers.Integral, slice)):
                assert isinstance(new_args[0], slice) and isinstance(new_args[1], slice)
                new_args[:2] = np.broadcast_arrays(
                    np.expand_dims(
                        np.arange(new_args[0].start, new_args[0].stop),
                        axis=tuple(i for i in range(self.field_view.ndim) if i != 0)
                    ),
                    np.expand_dims(
                        np.arange(new_args[1].start, new_args[1].stop),
                        axis=tuple(i for i in range(self.field_view.ndim) if i != 1)
                    ),
                )
            return tuple(new_args)

        def __getitem__(self, key):
            return self.field_view.__getitem__(self.shim_key(key))

        def __setitem__(self, key, value):
            return self.field_view.__setitem__(self.shim_key(key), value)
    """
)


class NpirCodegen(TemplatedGenerator):
    @dataclass
    class BlockContext:
        locals_declared: Set[str] = field(default_factory=set)

        def add_declared(self, *args):
            self.locals_declared |= set(args)

    contexts = (SymbolTableTrait.symtable_merger,)

    FieldDecl = FormatTemplate(
        "{name} = Field({name}, _origin_['{name}'], ({', '.join(dimensions)}))"
    )

    TemporaryDecl = FormatTemplate(
        "{name} = Field.empty((_dI_ + {padding[0]}, _dJ_ + {padding[1]}, _dK_), ({', '.join(offset)}, 0))"
    )

    # LocalDecl is purposefully omitted.

    VarKOffset = FormatTemplate("lk + {k}")

    def visit_FieldSlice(
        self,
        node: npir.FieldSlice,
        *,
        is_serial: bool = False,
        **kwargs: Any,
    ) -> Union[str, Collection[str]]:
        offsets = [node.i_offset, node.j_offset] + [
            self.visit(node.k_offset, is_serial=is_serial, **kwargs)
            if isinstance(node.k_offset, npir.VarKOffset)
            else node.k_offset
        ]
        data_index = self.visit(node.data_index, is_serial=is_serial, inside_slice=True, **kwargs)

        if isinstance(offsets[2], str):
            k = offsets[2]
        elif is_serial:
            ko = offsets[2]
            k = "k_{}:k_{}".format(
                f"{ko:+d}" if ko != 0 else "",
                f"{ko+1:+d}" if ko + 1 != 0 else "",
            )
        else:
            k = _slice_string("k", offsets[2])
        all_args = [_slice_string(ch, offset) for ch, offset in zip(("i", "j"), offsets)] + [k]
        if node.name in kwargs.get("symtable", {}):
            decl = kwargs["symtable"][node.name]
            dimensions = decl.dimensions if isinstance(decl, npir.FieldDecl) else [True] * 3
            args = [axis for i, axis in enumerate(all_args) if dimensions[i]]
        else:
            args = all_args
        access_slice = ", ".join(args + list(data_index))

        return f"{node.name}[{access_slice}]"

    LocalScalarAccess = ParamAccess = FormatTemplate("{name}")

    def visit_DataType(self, node: common.DataType, **kwargs: Any) -> Union[str, Collection[str]]:
        return f"np.{node.name.lower()}"

    def visit_BuiltInLiteral(
        self, node: common.BuiltInLiteral, **kwargs
    ) -> Union[str, Collection[str]]:
        if node is common.BuiltInLiteral.TRUE:
            return "True"
        elif node is common.BuiltInLiteral.FALSE:
            return "False"
        else:
            return self.generic_visit(node, **kwargs)

    def visit_ScalarLiteral(
        self, node: npir.ScalarLiteral, *, inside_slice: bool = False, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        # This could be trivial, but it's convenient for reading if the dtype is omitted in slices.
        dtype = self.visit(node.dtype, inside_slice=inside_slice, **kwargs)
        value = self.visit(node.value, inside_slice=inside_slice, **kwargs)
        return f"{value}" if inside_slice else f"{dtype}({value})"

    ScalarCast = FormatTemplate("{dtype}({expr})")

    VectorCast = FormatTemplate("{expr}.astype({dtype})")

    def visit_NativeFunction(
        self, node: common.NativeFunction, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node == common.NativeFunction.MIN:
            return "np.minimum"
        elif node == common.NativeFunction.MAX:
            return "np.maximum"
        elif node == common.NativeFunction.POW:
            return "np.power"
        elif node == common.NativeFunction.GAMMA:
            return "scipy.special.gamma"
        return "np." + self.generic_visit(node, **kwargs)

    def visit_NativeFuncCall(
        self, node: npir.NativeFuncCall, *, mask: Optional[str] = None, **kwargs: Any
    ):
        kwargs["mask_arg"] = f", where={mask}" if mask else ""
        return self.generic_visit(node, mask=mask, **kwargs)

    NativeFuncCall = FormatTemplate("{func}({', '.join(arg for arg in args)}{mask_arg})")

    def visit_VectorAssign(
        self, node: npir.VectorAssign, *, ctx: "BlockContext", **kwargs: Any
    ) -> Union[str, Collection[str]]:
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)
        if not node.mask:
            return f"{left} = {right}"

        mask = self.visit(node.mask, **kwargs)
        if (
            isinstance(node.left, npir.LocalScalarAccess)
            and node.left.name not in ctx.locals_declared
        ):
            # Note: Have seen that LocalScalarAccess on LHS with dtype = None.
            # Until that is solved can get the dtype from the symtable.
            dtype = kwargs["symtable"][node.left.name].dtype
            default_val = f"{self.visit(dtype, **kwargs)}()"
            ctx.add_declared(node.left.name)
        else:
            default_val = left

        return f"{left} = np.where({mask}, {right}, {default_val})"

    VectorArithmetic = FormatTemplate("({left} {op} {right})")

    VectorLogic = FormatTemplate("np.bitwise_{op}({left}, {right})")

    def visit_UnaryOperator(
        self, node: common.UnaryOperator, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node is common.UnaryOperator.NOT:
            return "np.bitwise_not"
        return self.generic_visit(node, **kwargs)

    VectorUnaryOp = FormatTemplate("({op}({expr}))")

    VectorTernaryOp = FormatTemplate("np.where({cond}, {true_expr}, {false_expr})")

    def visit_LevelMarker(
        self, node: common.LevelMarker, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return "K" if node == common.LevelMarker.END else "k"

    def visit_AxisBound(self, node: common.AxisBound, **kwargs: Any) -> Union[str, Collection[str]]:
        if node.offset > 0:
            voffset = f" + {node.offset}"
        elif node.offset == 0:
            voffset = ""
        else:
            voffset = f" - {-node.offset}"
        return self.generic_visit(node, voffset=voffset, **kwargs)

    AxisBound = FormatTemplate("_d{level}_{voffset}")

    def visit_LoopOrder(self, node: common.LoopOrder, **kwargs) -> Union[str, Collection[str]]:
        if node is common.LoopOrder.FORWARD:
            return "for k_ in range(k, K):"
        elif node is common.LoopOrder.BACKWARD:
            return "for k_ in range(K-1, k-1, -1):"
        return ""

    def visit_Broadcast(
        self,
        node: npir.Broadcast,
        *,
        is_serial: bool,
        lower: Tuple[int, int],
        upper: Tuple[int, int],
        **kwargs: Any,
    ) -> Union[str, Collection[str]]:
        boundary = [upper - lower for lower, upper in zip(lower, upper)]
        shape = _dump_sequence(
            [f"_dI_ + {boundary[0]}", f"_dJ_ + {boundary[1]}"]
            + ["1" if is_serial else "K - k"]
            + ["1"] * (node.dims - 3)
        )
        return self.generic_visit(
            node, shape=shape, is_serial=is_serial, lower=lower, upper=upper, **kwargs
        )

    Broadcast = FormatTemplate("np.full({shape}, {expr})")

    def visit_VerticalPass(self, node: npir.VerticalPass, **kwargs):
        is_serial = node.direction != common.LoopOrder.PARALLEL
        has_variable_k = bool(node.iter_tree().if_isinstance(npir.VarKOffset).to_list())
        if has_variable_k:
            lk_stmt = "lk = {}".format(
                "k_" if is_serial else "np.arange(k, K)[np.newaxis, np.newaxis, :]"
            )
        else:
            lk_stmt = ""
        return self.generic_visit(node, is_serial=is_serial, lk_stmt=lk_stmt, **kwargs)

    VerticalPass = JinjaTemplate(
        textwrap.dedent(
            """\
            # --- begin vertical block ---{% set body_indent = 0 %}
            k, K = {{ lower }}, {{ upper }}
            {%- if direction %}
            {{ direction }}{% set body_indent = 4 %}{% endif %}
            {{ lk_stmt | indent(body_indent, first=True) }}{% for hblock in body %}
            {{ hblock | indent(body_indent, first=True) }}
            {% endfor %}# --- end vertical block ---
            """
        )
    )

    def visit_HorizontalBlock(
        self, node: npir.HorizontalBlock, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        lower = [-node.extent[0][0], -node.extent[1][0]]
        upper = [node.extent[0][1], node.extent[1][1]]
        return self.generic_visit(node, lower=lower, upper=upper, ctx=self.BlockContext(), **kwargs)

    HorizontalBlock = JinjaTemplate(
        textwrap.dedent(
            """\
            # --- begin horizontal block --
            i, I = _di_ - {{ lower[0] }}, _dI_ + {{ upper[0] }}
            j, J = _dj_ - {{ lower[1] }}, _dJ_ + {{ upper[1] }}

            {% for assign in body %}{{ assign }}
            {% endfor -%}
            # --- end horizontal block --

            """
        )
    )

    def visit_Computation(
        self, node: npir.Computation, *, ignore_np_errstate: bool = True, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        signature = ["*", *node.arguments, "_domain_", "_origin_"]
        return self.generic_visit(
            node,
            signature=", ".join(signature),
            data_view_class=ORIGIN_CORRECTED_VIEW_CLASS,
            ignore_np_errstate=ignore_np_errstate,
            **kwargs,
        )

    Computation = JinjaTemplate(
        textwrap.dedent(
            """\
            import numbers
            from typing import Tuple

            import numpy as np
            import scipy.special

            {{ data_view_class }}

            def run({{ signature }}):

                # --- begin domain boundary shortcuts ---
                _di_, _dj_, _dk_ = 0, 0, 0
                _dI_, _dJ_, _dK_ = _domain_
                # --- end domain padding ---

                {% for decl in api_field_decls %}{{ decl | indent(4) }}
                {% endfor %}
                {% for decl in temp_decls %}{{ decl | indent(4) }}
                {% endfor %}

                {% if ignore_np_errstate -%}
                with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                {%- else -%}
                with np.errstate():
                {%- endif %}

                {% for pass in vertical_passes %}
                {{ pass | indent(8) }}
                {% else %}
                    pass
                {% endfor %}
            """
        )
    )
