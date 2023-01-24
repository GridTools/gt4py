from types import SimpleNamespace

import pytest

from eve import NodeTranslator
from eve.utils import UIDs
from functional.iterator import ir
from functional.iterator.transforms.reduction_utils import _get_partial_offsets, register_ir

from .test_unroll_reduce import basic_reduction


register_ir(ir)


def test_get_partial_offsets(basic_reduction):
    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=False)}
    partial_offsets = _get_partial_offsets(basic_reduction.args)

    assert partial_offsets == ["dim"]


# TODO move some of the tests from unroll_reduce
