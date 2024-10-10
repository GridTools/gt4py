# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.otf.binding import nanobind

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    program_source_example,
)


def test_bindings(program_source_example):
    module = nanobind.create_bindings(program_source_example)
    assert module.library_deps[0].name == "nanobind"
