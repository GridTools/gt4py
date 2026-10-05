# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the On-The-Fly (OTF) compilation workflow for the GTIR-DaCe backend.

The main module is `backend`, that exports the backends for CPU and GPU devices.
The `backend` module uses `factory` to define a workflow that implements the
`CompilePipeline` recipe. The different stages are implemeted in separate modules:
- `translation` for lowering of GTIR to SDFG and applying SDFG transformations
- `compilation` for compiling the SDFG into a program
- `decoration` to parse the program arguments and pass them to the program call

The toolchain builders create every step from one `DaCeConfig`, and wrap the
translation step in a persistent `CachedStep` unless the config disables it,
thus they provide caching of the GTIR program.
"""
