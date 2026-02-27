import functools
import logging
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import backend as gtx_backend

log = logging.getLogger(__name__)


def dict_values_to_list(d: dict[str, Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def customize_backend(
    program: gtx_typing.Program | gtx.typing.FieldOperator | None,
    backend: gtx_typing.Backend | None,
) -> gtx_typing.Backend | None:
    program_name = program.__name__ if program is not None else ""
    backend_name = backend.name if backend is not None else "embedded"
    if backend is None or isinstance(backend, gtx_backend.Backend):
        log.info(f"Using backend '{backend_name}' for '{program_name}'.")
        return backend

    custom_backend = backend
    log.info(f"Using backend '{backend_name}' for '{program_name}'.")
    return custom_backend


def setup_program(
    program: gtx_typing.Program,
    backend: gtx_typing.Backend | None,
    constant_args: dict[str, gtx.Field | gtx_typing.Scalar] | None = None,
    variants: dict[str, list[gtx_typing.Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: gtx_typing.OffsetProvider | None = None,
) -> Callable[..., None]:
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    backend = customize_backend(program, backend)

    bound_static_args = {k: v for k, v in constant_args.items() if gtx.is_scalar_type(v)}
    static_args_program = program.with_backend(backend)
    if backend is not None:
        static_args_program = static_args_program.with_compilation_options(enable_jit=False)
        static_args_program.compile(
            **dict_values_to_list(horizontal_sizes),
            **dict_values_to_list(vertical_sizes),
            **variants,
            **dict_values_to_list(bound_static_args),
            offset_provider=offset_provider,
        )

    return functools.partial(
        static_args_program,
        **constant_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )
