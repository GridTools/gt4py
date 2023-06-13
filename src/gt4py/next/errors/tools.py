import pathlib
from gt4py.eve import SourceLocation


def get_source_from_location(location: SourceLocation):
    try:
        source_file = pathlib.Path(location.source)
        source_code = source_file.read_text()
        source_lines = source_code.splitlines(False)
        start_line = location.line
        end_line = location.end_line + 1 if location.end_line else start_line + 1
        relevant_lines = source_lines[(start_line-1):(end_line-1)]
        return "\n".join(relevant_lines)
    except Exception as ex:
        raise ValueError("failed to get source code for source location") from ex