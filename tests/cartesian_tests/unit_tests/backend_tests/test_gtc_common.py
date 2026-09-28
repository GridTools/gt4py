# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from gt4py.cartesian.stencil_builder import StencilBuilder
from gt4py.eve import formatting


FORMATTED_MARK = "// formatted\n"


def sample_stencil(in_field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        in_field += 1  # type: ignore


@pytest.mark.parametrize("format_source", [True, False])
def test_make_extension_sources_formats_only_when_enabled(format_source, monkeypatch):
    monkeypatch.setattr(formatting, "format_cpp_source", lambda source: FORMATTED_MARK + source)
    builder = StencilBuilder(sample_stencil, backend="gt:cpu_ifirst").with_options(
        name="sample_stencil", module=__name__, format_source=format_source
    )

    sources = builder.backend._make_extension_sources()
    all_sources = [source for group in sources.values() for source in group.values()]
    assert {"computation", "bindings"} <= sources.keys()
    assert all_sources
    assert all(source.startswith(FORMATTED_MARK) == format_source for source in all_sources)
    assert not any(source.startswith(FORMATTED_MARK * 2) for source in all_sources)
