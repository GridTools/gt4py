import pathlib
import tempfile
import textwrap

from functional.fencil_processors.callables import modules


def test_load_binding():
    src_module = textwrap.dedent(
        """\
    def function(a, b):
        return a + b
    """
    )
    with tempfile.TemporaryDirectory() as folder:
        file = pathlib.Path(folder) / "module.py"
        file.write_text(src_module)
        functions = modules.load_module(file)
        assert "function" in functions
