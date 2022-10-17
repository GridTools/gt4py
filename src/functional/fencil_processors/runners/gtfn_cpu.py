# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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


import dataclasses
from typing import Any, Final, Optional, Callable

import numpy as np

from functional.fencil_processors import processor_interface as fpi  # fencil processor interface
from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import bindings, build
from functional.fencil_processors.codegens.gtfn import gtfn_module
from functional.fencil_processors.source_modules import cpp_gen, source_modules
from functional.iterator import ir as itir
from functional import common


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if hasattr(arg, "__array__"):
        return np.asarray(arg)
    else:
        return arg


from functional.ffront.symbol_makers import make_symbol_type_from_value

from eve.utils import content_hash

@dataclasses.dataclass(frozen=True)
class GTFNExecutor(fpi.FencilExecutor):
    language_settings: source_modules.LanguageWithHeaderFilesSettings = cpp_gen.CPP_DEFAULT

    name: Optional[str] = None

    _cache: dict[int, Callable] = dataclasses.field(repr=False, init=False, default_factory=dict)

    def __call__(self, fencil: itir.FencilDefinition, *args: Any, offset_provider, lift_mode: Optional[str] = None, column_axis: Optional = None) -> None:
        """
        Execute the iterator IR fencil with the provided arguments.

        The fencil is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``FencilExecutorFunction`` for details.
        """
        # TODO(tehrengruber): poor mans cache, until we have a better solution.
        cache_key = hash((
            fencil,
            # TODO(tehrengruber): as the resulting frontend types contain lists they are
            #  not hashable. As a workaround we just use content_hash here.
            content_hash(tuple(make_symbol_type_from_value(arg) for arg in args)),
            id(offset_provider),
            lift_mode,
            column_axis))

        if not cache_key in self._cache:
            # TODO(ricoh): a pipeline runner might enhance readability as well as discourage
            #  custom logic between steps.
            impl = self._cache[cache_key] = build.CMakeProject(
                source_module=(
                    source_module := gtfn_module.GTFNSourceModuleGenerator(self.language_settings)(
                        fencil, *args,
                        offset_provider=offset_provider,
                        lift_mode=lift_mode,
                        column_axis=column_axis
                    )
                ),
                bindings_module=bindings.create_bindings(source_module),
                #cache_strategy=cache.Strategy.SESSION,
                cache_strategy=cache.Strategy.PERSISTENT,
            ).get_implementation()
        else:
            impl = self._cache[cache_key]

        return impl(*[convert_arg(arg) for arg in args], *[op.tbl for op in offset_provider.values() if isinstance(op, common.Connectivity)])

    @property
    def __name__(self) -> str:
        return self.name or repr(self)


run_gtfn: Final[fpi.FencilProcessorProtocol[None, fpi.FencilExecutor]] = GTFNExecutor(
    name="run_gtfn"
)
