from gt4py.next.errors import CompilerError
from gt4py.eve import SourceLocation


loc = SourceLocation("/source/file.py", 5, 2, end_line=5, end_column=9)
msg = "a message"


def test_message():
    assert CompilerError(loc, msg).message == msg


def test_location():
    assert CompilerError(loc, msg).location_trace[0] == loc


