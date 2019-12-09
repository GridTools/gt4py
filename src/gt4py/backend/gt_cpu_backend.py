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


def make_x86_layout_map(mask):
    ctr = iter(range(sum(mask)))
    if len(mask) < 3:
        layout = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask = [*mask[3:], *mask[:3]]
        layout = [next(ctr) if m else None for m in swapped_mask]

        layout = [*layout[-3:], *layout[:-3]]

    return tuple(layout)


def x86_is_compatible_layout(field):
    stride = 0
    layout_map = make_x86_layout_map(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def gtcpu_is_compatible_type(field):
    return isinstance(field, np.ndarray)


def make_mc_layout_map(mask):
    ctr = reversed(range(sum(mask)))
    if len(mask) < 3:
        layout = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask = list(mask)
        tmp = swapped_mask[1]
        swapped_mask[1] = swapped_mask[2]
        swapped_mask[2] = tmp

        layout = [next(ctr) if m else None for m in swapped_mask]

        tmp = layout[1]
        layout[1] = layout[2]
        layout[2] = tmp

    return tuple(layout)


def mc_is_compatible_layout(field):
    stride = 0
    layout_map = make_mc_layout_map(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


class GTCPUBackend(gt_backend.BaseGTBackend):
    @classmethod
    def generate_extension(cls, stencil_id, performance_ir, options):
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
            options,
        )
        gt_pyext_sources = gt_pyext_generator(performance_ir)

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
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    _CPU_ARCHITECTURE = "x86"


@gt_backend.register
class GTMCBackend(GTCPUBackend):

    name = "gtmc"
    options = gt_backend.BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 8,
        "device": "cpu",
        "layout_map": make_mc_layout_map,
        "is_compatible_layout": mc_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    _CPU_ARCHITECTURE = "mc"
