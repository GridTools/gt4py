# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from gt4py.next.iterator import ir
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon
from gt4py.next.type_system import type_specifications as ts


def argument_processing_function(
    args: tuple[Any, ...],
    offset_provider: Mapping[str, Any],
    metrics_level: int,
    runtime_return_value: Any,
) -> Any: ...


def _perfom_tuple_unpacking(
    tuple_argument: Union[tuple[Any, ...], str],
    tuple_len: int,
    unpacked_variables: Optional[list[str]],
) -> Union[tuple[Any, ...], list[str]]:
    if unpacked_variables is None:
        assert isinstance(tuple_argument, tuple) and len(tuple_argument) == tuple_len
        return tuple_argument
    else:
        assert isinstance(tuple_argument, str) and len(tuple_argument) > 0
        unpacked_arguments = [f"{tuple_argument}_{i}" for i in range(tuple_len)]
        unpacked_variables.append(f"{', '.join(unpacked_arguments)}, = {tuple_argument}")
        return unpacked_arguments


def _process_argument(
    argument: Union[Any, str],
    param_type: Optional[ts.TypeSpec],
    unpacked_variables: Optional[list[str]],
    back_end: str,
) -> Union[Any, str]:

    tuple_types: Optional[Sequence[Any]]
    if unpacked_variables is None:
        assert param_type is None
        eval_mode = True
        is_field = hasattr(argument, "__gt_origin__")
        is_scalar = True  # No other choice
        is_bool = isinstance(argument, (np.bool_, bool))
        tuple_types = (None,) * len(argument) if (is_tuple := isinstance(argument, tuple)) else None

    else:
        assert param_type is not None
        eval_mode = False
        is_field = isinstance(param_type, ts.FieldType)
        is_scalar = isinstance(param_type, ts.ScalarType)
        tuple_types = (
            param_type.types if (is_tuple := isinstance(param_type, ts.TupleType)) else None
        )
        is_bool = param_type.kind == ts.ScalarKind.BOOL if is_scalar else False  # type: ignore[attr-defined]

    if is_tuple:
        assert tuple_types is not None  # Make MyPy happy
        tuple_members = _perfom_tuple_unpacking(argument, len(tuple_types), unpacked_variables)
        processed_tuple_members: Union[str, tuple[Any, ...]] = tuple() if eval_mode else "("
        for tuple_member, tuple_member_type in zip(tuple_members, tuple_types):
            processed_tuple_members += _process_argument(  # type: ignore[operator]
                argument=tuple_member,
                param_type=tuple_member_type,
                unpacked_variables=unpacked_variables,
                back_end=back_end,
            )

        # The trailing commas are intentionally and are needed for symmetry reasons. See
        #  also the note in `_perform_argument_processing()`.
        return (processed_tuple_members,) if eval_mode else (processed_tuple_members + "),")  # type: ignore[operator]

    command: str
    if is_field:
        argument_name = "argument" if eval_mode else argument
        if (len(argument.domain) == 0) if eval_mode else (len(param_type.dims) == 0):  # type: ignore[union-attr]
            # GTFN wants a field, dace a scalar.
            command = f"{argument_name}" if back_end == "gtfn" else f"{argument_name}.as_scalar()"
        else:
            # Full array, the differences is where the origin comes from.
            origin_prop = "__gt_origin__" if back_end == "gtfn" else "__dace_origin__"
            command = f"({argument_name}.ndarray, ({argument_name}.{origin_prop}))"

    elif is_scalar:
        command = "{cast}({arg})".format(
            arg="argument" if eval_mode else argument,
            cast=("bool" if is_bool and back_end == "gtfn" else ""),
        )
    else:
        assert isinstance(argument, str)
        raise ValueError(f"Parameter `{argument}` had unexpected type `{param_type}`")

    command += ", "  # Comma is intentional.
    return eval(command) if eval_mode else command


def _process_offset_providers(
    offset_provider: Mapping[str, Any],
    eval_mode: bool,
) -> Union[str, tuple[Any, ...]]:
    # Assumes that the order of the offset providers is stable.
    # And that there is no difference between GTFN and DaCe
    processed_offset_providers: Union[str, tuple[Any, ...]] = tuple() if eval_mode else ""
    for table_name in offset_provider:
        command = f"(offset_provider['{table_name}'].ndarray, (0, 0)), "
        processed_offset_providers += eval(command) if eval_mode else command  # type: ignore[operator]
    return processed_offset_providers


def _process_metric_arguments(
    metric_level: Union[str, int],
    compute_time_argument: Union[str, Any],
    eval_mode: bool,
    back_end: str,
    use_metrics: bool,
) -> Union[str, Any]:

    if not use_metrics:
        return tuple() if eval_mode else ""

    # This is currently different and should probably be unified somehow.
    if back_end == "gtfn":
        raise NotImplementedError("Implement me")
    else:
        if eval_mode:
            assert not (isinstance(metric_level, str) or isinstance(compute_time_argument, str))
            return (metric_level, compute_time_argument)
        else:
            assert isinstance(metric_level, str) and isinstance(compute_time_argument, str)
            return f"{metric_level}, {compute_time_argument}, "


def _perform_argument_processing(
    args: tuple[Any, ...],
    offset_provider: Mapping[str, Any],
    metric_level: int,
    runtime_return_value: Any,
    use_metrics: bool,  # Not in generated signature, but must be provided.
    back_end: str,  # Not part but must be provided.
) -> tuple[Any]:

    # NOTE: Although the loop bellow looks like it could be turned into into a tuple
    #   comprehension it would lead to an error. This is related to the output type
    #   of `_process_argument()`. In essence it boils down to `append()` (comprehension)
    #   and `extend()` (loop).
    processed_arguments: tuple[Any, ...] = tuple()
    for arg in args:
        processed_arguments += _process_argument(  # type: ignore[operator]
            argument=arg, param_type=None, unpacked_variables=None, back_end=back_end
        )
    processed_arguments += _process_offset_providers(  # type: ignore[operator]
        offset_provider=offset_provider, eval_mode=True
    )
    processed_arguments += _process_metric_arguments(  # type: ignore[operator]
        metric_level=metric_level,
        compute_time_argument=runtime_return_value,
        eval_mode=True,
        back_end=back_end,
        use_metrics=use_metrics,
    )
    return processed_arguments


def write_convert_code(
    update_function_name: str,
    prog: ir.Program,
    use_metrics: bool,
    offset_provider: Mapping[str, Any],
    back_end: str,
    eval_mode: bool,
) -> str:
    back_end = back_end.lower()
    assert back_end in ["gtfn", "dace"]

    # Now writing the code
    code_lines: list[str] = []

    if eval_mode:
        code_lines.append(
            "from gt4py.next.program_processors.runners.dace.workflow import nanobinings as gtx_wfdnanobindings"
        )
    code_lines.append(f"def {update_function_name}(")
    code_lines.append("\targs,")
    code_lines.append("\toffset_provider,")
    code_lines.append(f"\t{gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL},")
    code_lines.append(f"\t{gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME},")
    code_lines.append("):")

    if eval_mode:
        # The code generated function only dispatches to the real function.
        code_lines.append("\treturn gtx_wfdnanobindings._perform_argument_processing(")
        code_lines.append("\t\targs=args,")
        code_lines.append("\t\toffset_provider=offset_provider,")
        code_lines.append(f"\t\tmetric_level={gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL},")
        code_lines.append(f"\t\truntime_return_value={gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME},")
        code_lines.append(f"\t\tuse_metrics={use_metrics},")
        code_lines.append(f"\t\tback_end='{back_end}',")
        code_lines.append("\t)")
    else:
        # The code generated function performs the processing.
        expanded_args: list[str] = [f"__gtx_expanded_names_{param.id}" for param in prog.params]
        unpacked_variables: list[str] = [", ".join(expanded_args) + ", = args"]
        process_arg_stmt = ""
        for arg_name, param in zip(expanded_args, prog.params):
            process_arg_stmt += _process_argument(
                argument=arg_name,
                param_type=param.type,
                unpacked_variables=unpacked_variables,
                back_end=back_end,
            )
        process_arg_stmt += _process_offset_providers(  # type: ignore[operator]
            offset_provider=offset_provider, eval_mode=False
        )
        process_arg_stmt += _process_metric_arguments(
            metric_level=gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL,
            compute_time_argument=gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME,
            eval_mode=False,
            back_end=back_end,
            use_metrics=use_metrics,
        )
        code_lines.extend(("\t" + s for s in unpacked_variables))
        code_lines.append(f"\treturn ({process_arg_stmt})")

    return "\n".join(code_lines)
