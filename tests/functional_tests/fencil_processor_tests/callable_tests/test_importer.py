import pathlib
import tempfile
import textwrap

from functional.fencil_processors.callables import importer


def test_import_callables():
    src_module = textwrap.dedent(
        """\
    def function(a, b):
        return a + b
    """
    )
    with tempfile.TemporaryDirectory() as folder:
        file = pathlib.Path(folder) / "module.py"
        file.write_text(src_module)
        functions = importer.import_callables(file)
        assert "function" in functions
