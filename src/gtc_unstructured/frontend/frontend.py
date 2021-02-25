# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

import ast
import inspect
import textwrap

import devtools

from gtc_unstructured.frontend.gtscript_to_gtir import GTScriptToGTIR, NodeCanonicalizer
from gtc_unstructured.frontend.py_to_gtscript import PyToGTScript
from gtc_unstructured.irs import common, gtir
from gtc_unstructured.irs.gtir_to_nir import GtirToNir
from gtc_unstructured.irs.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops
from gtc_unstructured.irs.nir_passes.merge_neighbor_loops import find_and_merge_neighbor_loops
from gtc_unstructured.irs.nir_to_usid import NirToUsid
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator

from . import built_in_functions, built_in_types
from .gtscript_ast import Argument, External


# todo(tehrengruber): the frontend as written here will disappear at some point as the `PassManager` in Eve and
#  build stages in GT4Py provide most of the functionality here. Please keep this class as reduced as possible in
#  the meantime.
class GTScriptCompilationTask:
    def __init__(self, definition):
        self.definition = definition
        self.source = None
        self.python_ast = None
        self.gtscript_ast = None
        self.gtir = None
        self.cpp_code = None

    def _get_arguments(self):
        """
        Populate symbol table by extracting the argument types from scope the function is embedded in.
        """
        # assert self.gtscript_ast is not None
        args = []
        sig = inspect.signature(self.definition)
        for name, param in sig.parameters.items():
            args.append(Argument(name=name, type_=param.annotation))
        return args

    def _generate_gtscript_ast(self):
        externals = [
            External(name="dtype", value=common.DataType.FLOAT64),
            External(name="Vertex", value=common.LocationType.Vertex),
            External(name="Edge", value=common.LocationType.Edge),
            External(name="Cell", value=common.LocationType.Cell),
            External(name="Field", value=built_in_types.Field),
            External(name="Connectivity", value=built_in_types.Connectivity),
            External(name="LocalField", value=built_in_types.LocalField),
        ] + [
            External(name=name, value=getattr(built_in_functions, name))
            for name in built_in_functions.__all__
        ]

        self.source = textwrap.dedent(inspect.getsource(self.definition))
        self.python_ast = ast.parse(self.source).body[0]
        self.gtscript_ast = PyToGTScript().transform(
            self.python_ast,
            node_init_args={"externals": externals, "arguments": self._get_arguments()},
        )

        return self.gtscript_ast

    def _generate_gtir(self):
        # Canonicalization
        NodeCanonicalizer.apply(self.gtscript_ast)

        # Transform into GTIR
        self.gtir = GTScriptToGTIR.apply(self.gtscript_ast)

        return self.gtir

    def _generate_cpp(self, *, debug=False, code_generator=UsidGpuCodeGenerator):
        # Code generation
        nir_comp = GtirToNir().visit(self.gtir)
        nir_comp = find_and_merge_horizontal_loops(nir_comp)
        # nir_comp = find_and_merge_neighbor_loops(nir_comp)
        usid_comp = NirToUsid().visit(nir_comp)

        if debug:
            devtools.debug(nir_comp)
            devtools.debug(usid_comp)

        self.cpp_code = code_generator.apply(usid_comp)

        return self.cpp_code

    def generate(self, *, debug=False, code_generator=UsidGpuCodeGenerator):
        """
        Generate c++ code of the stencil.
        """
        if isinstance(self.definition, gtir.Computation):
            # accept GTIR as input
            self.gtir = self.definition
        else:
            # default case, assume a gtscript function definition is passed
            self._generate_gtscript_ast()
            self._generate_gtir()
        self._generate_cpp(debug=debug, code_generator=code_generator)
        return self.cpp_code
