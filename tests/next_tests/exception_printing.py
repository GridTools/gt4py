from gt4py.next.errors import *
import inspect
from gt4py.eve import SourceLocation


frameinfo = inspect.getframeinfo(inspect.currentframe())
loc = SourceLocation(frameinfo.lineno, 1, frameinfo.filename, end_line=frameinfo.lineno, end_column=5)
raise CompilerError(loc, "this is an error message")