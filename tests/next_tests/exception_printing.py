from gt4py.next.ffront.decorator import field_operator
from gt4py.next.errors import *

@field_operator
def testee(a) -> float:
    return 1

testee(1, offset_provider={})