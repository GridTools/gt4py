# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

import os

import numpy as np

from gt4py import backend as gt_backend
from gt4py import storage as gt_storage
from gt4py.storage import StorageDefaults
from . import pyext_builder


class GTCPUBackend(gt_backend.BaseGTBackend):
    @classmethod
    def generate_extension(cls, stencil_id, implementation_ir, storage_defaults, options):
        pyext_opts = dict(
            verbose=options.backend_opts.pop("verbose", False),
            clean=options.backend_opts.pop("clean", False),
            debug_mode=options.backend_opts.pop("debug_mode", False),
            add_profile_info=options.backend_opts.pop("add_profile_info", False),
        )

        # Generate source
        gt_pyext_generator = cls.PYEXT_GENERATOR_CLASS(
            cls.get_pyext_class_name(stencil_id),
            cls.get_pyext_module_name(stencil_id),
            cls._CPU_ARCHITECTURE,
            storage_defaults,
            options,
        )
        gt_pyext_sources = gt_pyext_generator(implementation_ir)

        # Build extension module
        pyext_build_path = os.path.relpath(cls.get_pyext_build_path(stencil_id))
        os.makedirs(pyext_build_path, exist_ok=True)
        sources = []
        for key, source in gt_pyext_sources.items():
            src_file_name = os.path.join(pyext_build_path, key)
            src_ext = src_file_name.split(".")[-1]
            if src_ext != "hpp":
                if src_ext == "src":
                    src_file_name = src_file_name.replace("src", "cpp")
                sources.append(src_file_name)

            with open(src_file_name, "w") as f:
                f.write(source)

        pyext_target_path = cls.get_stencil_package_path(stencil_id)
        qualified_pyext_name = cls.get_pyext_module_name(stencil_id, qualified=True)
        module_name, file_path = pyext_builder.build_gtcpu_ext(
            qualified_pyext_name,
            sources=sources,
            build_path=pyext_build_path,
            target_path=pyext_target_path,
            **pyext_opts,
        )
        assert module_name == qualified_pyext_name

        return module_name, file_path


@gt_backend.register
class GTX86Backend(GTCPUBackend):

    name = "gtx86"
    options = gt_backend.BaseGTBackend.GT_BACKEND_OPTS
    compute_device = "cpu"
    assert_specified_layout = True
    storage_defaults = StorageDefaults(layout_map=(0, 1, 2))

    _CPU_ARCHITECTURE = "x86"


@gt_backend.register
class GTMCBackend(GTCPUBackend):

    name = "gtmc"
    options = gt_backend.BaseGTBackend.GT_BACKEND_OPTS
    compute_device = "cpu"
    assert_specified_layout = True
    storage_defaults = StorageDefaults(alignment=8, layout_map=(0, 2, 1))

    _CPU_ARCHITECTURE = "mc"
