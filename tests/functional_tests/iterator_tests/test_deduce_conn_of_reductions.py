from types import SimpleNamespace

import pytest

from eve.utils import UIDs
from functional.iterator import ir
from functional.iterator.transforms.deduce_conn_of_reductions import DeduceConnOfReductions

from .test_unroll_reduce import basic_reduction


def test_deduce_conn_of_reduction(basic_reduction):
    offset_provider = {"dim": SimpleNamespace(max_neighbors=3, has_skip_values=False)}
    actual = DeduceConnOfReductions.apply(basic_reduction, offset_provider=offset_provider)

    assert actual.annex.reduction_offsets == ["dim"]


# TODO move some of the tests from unroll_reduce
