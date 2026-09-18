#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""
Python dev-scripts package.

Each public module exposing a global ``cli`` (a ``typer.Typer`` app) is
auto-discovered as a sub-command of the ``scripts/run`` toolbox. Dependencies
for all the modules in this package should be declared in the 'scripts'
dependency group (in 'pyproject.toml').

"""
