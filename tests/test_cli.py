import re
import sys

from click.testing import CliRunner
import pytest
import numpy

from gt4py import storage
from gt4py import cli
from gt4py import gtsimport


def has_cupy():
    try:
        import cupy

        return True
    except ImportError:
        return False


@pytest.fixture
def clirunner():
    yield CliRunner()


@pytest.fixture
def clean_imports():
    impdata = [sys.path.copy(), sys.meta_path.copy(), sys.modules.copy()]
    yield
    sys.path, sys.meta_path, sys.modules = impdata


@pytest.fixture
def simple_stencil(tmp_path, clean_imports):
    """Provide a gtscript file with a simple stencil."""
    module_file = tmp_path / "stencil.gtpy"
    module_file.write_text(
        (
            "## using-dsl: gtscript\n"
            "\n"
            "\n"
            "@lazy_stencil()\n"
            "def init_1(input_field: Field[float]):\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        input_field = 1\n"
        )
    )
    yield module_file


@pytest.fixture
def features_stencil(tmp_path, clean_imports):
    """Provide a gtscript file with a stencil using all the features."""
    module_file = tmp_path / "features_stencil.gtpy"
    module_file.write_text(
        (
            "## using-dsl: gtscript\n"
            "import numpy\n"
            "\n"
            "\n"
            "CONSTANT = 5.9\n"
            "\n"
            "\n"
            "@function\n"
            "def some_operation(field_a, field_b, constant=CONSTANT):\n"
            "    return field_a + constant * field_b\n"
            "\n"
            "\n"
            "@lazy_stencil(externals={'my_constant': CONSTANT})\n"
            "def some_stencil(field_a: Field[numpy.float], field_b: Field[numpy.float]):\n"
            "    from __externals__ import my_constant\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        field_a = some_operation(field_a, field_b, constant=my_constant)\n"
            "\n"
            "\n"
            "@lazy_stencil(externals={'fill_value': CONSTANT})\n"
            "def fill(field_a: Field[numpy.float]):\n"
            "    from __externals__ import fill_value\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        field_a = fill_value\n"
        )
    )
    yield module_file


@pytest.fixture
def library_stencil(tmp_path):
    """Provide a gtscript file that brings it's own library."""
    module_file = tmp_path / "library_stencil"


@pytest.fixture(scope="function")
def reset_importsys():
    print("storing data")
    stored_sys_path = sys.path.copy()
    stored_metapath = sys.meta_path.copy()
    stored_modules = sys.modules.copy()
    yield
    print("resetting data")
    sys.path = stored_sys_path
    sys.meta_path = stored_metapath
    sys.modules = stored_modules


def test_gtpyc_help(clirunner):
    """Calling the CLI with --help should print the usage message."""
    result = clirunner.invoke(cli.gtpyc, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Usage: " in result.output


def test_gtpyc_compile_debug(clirunner, simple_stencil, tmp_path):
    """Compile the test stencil using the debug backend and check for compiled source files."""
    output_path = tmp_path / "gtpyc_test_debug"
    result = clirunner.invoke(
        cli.gtpyc,
        [f"--output-path={output_path}", "--backend=debug", str(simple_stencil)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    print(list(output_path.iterdir()))
    py_files = [path.name for path in output_path.iterdir() if path.suffix == ".py"]
    assert "init_1.py" in py_files


def test_gtpyc_compile_gtx86_pyext(clirunner, simple_stencil, tmp_path, clean_imports):
    """Compile the test stencil binary python extension using the gtx86 backend and check the resulting source structure."""
    output_path = tmp_path / "gtpyc_test_gtx86"
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=gtx86",
            "--bindings=python",
            "--compile-bindings",
            "-O",
            "verbose=True",
            str(simple_stencil),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    print(result.output)
    print(list(output_path.iterdir()))
    py_files = [path for path in output_path.iterdir() if path.suffix == ".py"]
    stencil_modules = [path for path in py_files if path.name.startswith("init_1")]
    ext_modules = [
        path
        for path in output_path.iterdir()
        if path.suffix == ".so" and path.name.startswith("_init_1")
    ]
    build_paths = [
        path for path in output_path.iterdir() if path.name.startswith("src_") and path.is_dir()
    ]
    assert ext_modules
    assert stencil_modules
    assert build_paths
    build_path = build_paths[0]
    print([path.name for path in build_path.iterdir()])
    cpp_files = [path.name for path in build_path.iterdir() if path.suffix == ".cpp"]
    hpp_files = [path.name for path in build_path.iterdir() if path.suffix == ".hpp"]
    assert "computation.cpp" in cpp_files
    assert "bindings.cpp" in cpp_files
    assert "computation.hpp" in hpp_files
    sys.path.append(str(output_path))
    import init_1

    print(init_1)
    print(init_1.__dict__.keys())
    assert init_1.init_1


def test_gtpyc_nopy_gtx86(clirunner, simple_stencil, tmp_path):
    """Only generate the c++ files."""
    output_path = tmp_path / "gtpyc_test_nopy"
    result = clirunner.invoke(
        cli.gtpyc,
        [f"--output-path={output_path}", "--backend=gtx86", str(simple_stencil)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    src_path = output_path / f"src_init_1"
    src_files = [path.name for path in src_path.iterdir()]
    assert set(["computation.hpp", "computation.cpp"]) == set(src_files)


def test_gtpyc_withpy_gtx86(clirunner, simple_stencil, tmp_path):
    """Only generate source files but include python bindings."""
    output_path = tmp_path / "gtpyc_test_nopy"
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=gtx86",
            "--bindings=python",
            str(simple_stencil),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    src_path = output_path / f"src_init_1"
    output_contents = [path.name for path in output_path.iterdir()]
    src_contents = [path.name for path in src_path.iterdir()]
    assert set(["init_1.py", "src_init_1"]) == set(output_contents)
    assert set(["bindings.cpp", "computation.cpp", "computation.hpp"]) == set(src_contents)


def test_sub_and_multi(clirunner, features_stencil, tmp_path):
    """generate two or more stencils, one using a subroutine."""
    output_path = tmp_path / "gtpyc_test_submulti"
    result = clirunner.invoke(
        cli.gtpyc,
        [f"--output-path={output_path}", "--backend=debug", str(features_stencil),],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    src_files = set(path.name for path in output_path.iterdir())
    assert src_files.issuperset({"some_stencil.py", "fill.py"})


@pytest.mark.parametrize(
    "backend",
    [
        ("--backend=debug",),
        pytest.param(("--backend=gtx86", "--bindings=python", "--compile-bindings")),
        pytest.param(
            ("--backend=gtcuda", "--bindings=python", "--compile-bindings"),
            marks=pytest.mark.skipif(not has_cupy(), reason="cupy not installed"),
        ),
    ],
)
def test_externals_run_with_storage(
    clirunner, features_stencil, tmp_path, reset_importsys, backend
):
    """override externals on commandline."""
    import json
    import traceback

    output_path = tmp_path / "gtpyc_test_externals"
    my_constant = 3.14
    fill_value = 9.0
    externals = json.dumps({"my_constant": my_constant, "fill_value": fill_value})
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=debug",
            f"--externals={externals}",
            str(features_stencil),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, traceback.print_tb(result.exc_info[2])
    some_stencil_f = output_path / "some_stencil.py"
    fill_f = output_path / "fill.py"
    assert re.findall(
        rf'["\'\s]*my_constant["\'\s]*[:=]["\'\s]*{my_constant}["\'\s]*',
        some_stencil_f.read_text(),
    )
    assert re.findall(
        rf'["\'\s]*fill_value["\'\s]*[:=]["\'\s]*{fill_value}["\'\s]*', fill_f.read_text()
    )

    sys.path.append(str(fill_f.parent))
    import fill as fill_m

    stencil = fill_m.fill()

    data = storage.empty(
        backend=stencil.backend, shape=(2, 2, 1), dtype=float, default_origin=((0, 0, 0))
    )
    stencil(data)
    assert (data == numpy.full((2, 2, 1), fill_value)).all()


@pytest.mark.parametrize(
    "backend",
    [
        ("--backend=debug",),
        ("--backend=numpy",),
        pytest.param(
            ("--backend=gtx86", "--bindings=python", "--compile-bindings"),
            marks=pytest.mark.xfail(
                reason="using numpy arrays for storatges is not yet implemented for gtx86"
            ),
        ),
        pytest.param(
            ("--backend=gtcuda", "--bindings=python", "--compile-bindings"),
            marks=[
                pytest.mark.skipif(not has_cupy(), reason="cupy not installed"),
                pytest.mark.xfail(
                    reason="using numpy arrays for storatges is not yet implemented for gtx86"
                ),
            ],
        ),
    ],
)
def test_externals_run_with_array(clirunner, features_stencil, tmp_path, reset_importsys, backend):
    """override externals on commandline."""
    import json
    import traceback

    output_path = tmp_path / "gtpyc_test_externals"
    my_constant = 42
    fill_value = 8.0
    externals = json.dumps({"my_constant": my_constant, "fill_value": fill_value})
    args = [
        f"--output-path={output_path}",
        *backend,
        f"--externals={externals}",
        str(features_stencil),
    ]
    print(args)
    result = clirunner.invoke(cli.gtpyc, args, catch_exceptions=False)
    fill_f = output_path / "fill.py"
    print(list(output_path.iterdir()))
    sys.path.append(str(output_path))
    import fill as fill_m

    print(fill_m.__dict__.keys())
    stencil = fill_m.fill()

    data = numpy.empty(shape=(2, 2, 1), dtype=float)
    stencil(data, origin=(0, 0, 0))
    assert (data == numpy.full((2, 2, 1), fill_value)).all()
