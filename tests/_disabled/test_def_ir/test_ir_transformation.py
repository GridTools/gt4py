# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import itertools

from gt4py import analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir

from gt4py.definitions import StencilID
from .def_ir_stencil_definitions import REGISTRY as def_ir_registry

id_version_counter = itertools.count()


@pytest.fixture()
def id_version():
    return str(id_version_counter.__next__())


def analyze(name):
    module_name = "_test_module." + name
    stencil_name = name + "_stencil"
    options = gt_definitions.BuildOptions(name=stencil_name, module=module_name, rebuild=True)

    definition_ir_factory = def_ir_registry[name]
    definition_ir = definition_ir_factory()

    iir = gt_analysis.transform(definition_ir, options)

    return iir


def generate(def_ir, backend, *, id_version):
    module_name = "_test_module." + def_ir.name
    stencil_name = def_ir.name + "_stencil"
    options = gt_definitions.BuildOptions(name=stencil_name, module=module_name, rebuild=True)

    stencil_id = StencilID("{}.{}".format(options.module, options.name), id_version)

    if options.rebuild:
        # Force recompilation
        stencil_class = None
    else:
        # Use cached version (if id_version matches)
        stencil_class = backend.load(stencil_id, None, options)

    if stencil_class is None:
        stencil_class = backend.generate(stencil_id, def_ir, None, options)

    stencil_implementation = stencil_class()

    return stencil_implementation


@pytest.mark.parametrize("name", def_ir_registry.names)
def test_ir_transformation(name):
    iir = analyze(name)


@pytest.mark.parametrize(
    ["name", "backend"],
    itertools.product(
        def_ir_registry.names,
        [
            gt_backend.from_name(name)
            for name in gt_backend.REGISTRY.names
            if gt_backend.from_name(name).storage_info["device"] == "cpu"
        ],
    ),
)
def test_code_generation(name, backend, *, id_version):
    def_ir = analyze(name)
    generate(def_ir, backend, id_version=id_version)
