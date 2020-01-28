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

import os

import numpy as np

from gt4py import backend as gt_backend
from . import pyext_builder


def cuda_layout(mask):
    ctr = reversed(range(sum(mask)))
    return tuple([next(ctr) if m else None for m in mask])


class PythonGTCUDAGenerator(gt_backend.PythonGTGenerator):
    def generate_synchronization(self, field_names):
        return "\n".join([f + ".host_to_device()" for f in field_names])

    def generate_mark_modified(self, output_field_names):
        return "\n".join([f + "._set_device_modified()" for f in output_field_names])


def cuda_is_compatible_layout(field):
    stride = 0
    layout_map = cuda_layout(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def cuda_is_compatible_type(field):
    # ToDo: find a better way to remove the import cycle
    from gt4py.storage.storage import GPUStorage, ExplicitlySyncedGPUStorage

    return isinstance(field, (GPUStorage, ExplicitlySyncedGPUStorage))


@gt_backend.register
class GTCUDABackend(gt_backend.BaseGTBackend):
    GENERATOR_CLASS = PythonGTCUDAGenerator
    name = "gtcuda"
    options = gt_backend.BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": cuda_layout,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }

    @classmethod
    def generate_extension(cls, stencil_id, performance_ir, options):
        pyext_opts = dict(
            verbose=options.backend_opts.pop("verbose", True),
            clean=options.backend_opts.pop("clean", False),
            debug_mode=options.backend_opts.pop("debug_mode", False),
            add_profile_info=options.backend_opts.pop("add_profile_info", False),
        )

        # Generate source
        gt_pyext_generator = cls.PYEXT_GENERATOR_CLASS(
            cls.get_pyext_class_name(stencil_id),
            cls.get_pyext_module_name(stencil_id),
            "cuda",
            options,
        )
        gt_pyext_sources = gt_pyext_generator(performance_ir)

        # Build extension module
        pyext_build_path = cls.get_pyext_build_path(stencil_id)
        os.makedirs(pyext_build_path, exist_ok=True)
        sources = []
        for key, source in gt_pyext_sources.items():
            src_file_name = os.path.join(pyext_build_path, key)
            src_ext = src_file_name.split(".")[-1]
            if src_ext != "hpp":
                if src_ext == "src":
                    src_file_name = src_file_name.replace("src", "cu")
                sources.append(src_file_name)

            with open(src_file_name, "w") as f:
                f.write(source)

        pyext_target_path = cls.get_stencil_package_path(stencil_id)
        qualified_pyext_name = cls.get_pyext_module_name(stencil_id, qualified=True)
        module_name, file_path = pyext_builder.build_gtcuda_ext(
            qualified_pyext_name,
            sources=sources,
            build_path=pyext_build_path,
            target_path=pyext_target_path,
            **pyext_opts,
        )
        assert module_name == qualified_pyext_name

        return module_name, file_path
