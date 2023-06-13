from gt4py.next.errors import CompilationError
from gt4py.eve import SourceLocation


loc = SourceLocation(5, 2, "/source/file.py", end_line=5, end_column=9)
msg = "a message"


def test_message():
    assert CompilationError(loc, msg).msg == msg


def test_location():
    assert CompilationError(loc, msg).location == loc


