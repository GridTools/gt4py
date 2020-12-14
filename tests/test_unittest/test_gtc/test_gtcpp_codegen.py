import pytest

from gt4py.gtc.gtcpp import gtcpp_codegen
from gt4py.gtc.gtcpp.gtcpp import GTLevel


@pytest.mark.parametrize("root,expected", [(GTLevel(splitter=0, offset=5), 5)])
def test_offset_limit(root, expected):
    assert gtcpp_codegen._offset_limit(root) == expected
