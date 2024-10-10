# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Global configuration of test generation and execution with pytest."""

import pytest

from . import definitions


NODE_MAKERS = []
FROZEN_NODE_MAKERS = []
INVALID_NODE_MAKERS = []


# Automatic creation of pytest fixtures from maker functions in .definitions
for key, value in definitions.__dict__.items():
    if key.startswith("make_"):
        name = key[5:]
        exec(
            f"""
@pytest.fixture
def {name}_maker():
    yield definitions.make_{name}

@pytest.fixture
def {name}({name}_maker):
    yield {name}_maker()

@pytest.fixture
def fixed_{name}({name}_maker):
    yield {name}_maker(fixed=True)
"""
        )

        if "_node" in key:
            if "_invalid" in key:
                INVALID_NODE_MAKERS.append(value)
            elif "_frozen" in key:
                FROZEN_NODE_MAKERS.append(value)
            else:
                NODE_MAKERS.append(value)


@pytest.fixture(params=NODE_MAKERS)
def sample_node_maker(request):
    return request.param


@pytest.fixture(params=NODE_MAKERS)
def sample_node(request):
    return request.param()


@pytest.fixture(params=FROZEN_NODE_MAKERS)
def frozen_sample_node_maker(request):
    return request.param


@pytest.fixture(params=FROZEN_NODE_MAKERS)
def frozen_sample_node(request):
    return request.param()


@pytest.fixture(params=INVALID_NODE_MAKERS)
def invalid_sample_node_maker(request):
    return request.param
