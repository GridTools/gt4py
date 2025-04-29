# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import pytest

from gt4py.cartesian import gtscript_imports


@pytest.fixture(scope="function")
def reset_importsys():
    # This is only shown if a test fails. In that case it can be helpful just to check
    # it is not caused by caching in the import system or similar.
    print("storing import system configuration")
    stored_sys_path = sys.path.copy()
    stored_metapath = sys.meta_path.copy()
    stored_modules = sys.modules.copy()
    yield
    # Also only visible in case of failure.
    print("resetting import system configuration")
    sys.path = stored_sys_path
    sys.meta_path = stored_metapath
    sys.modules = stored_modules


@pytest.fixture(params=gtscript_imports.GTS_EXTENSIONS)
def extension(request):
    yield request.param


@pytest.fixture
def make_single_file(tmp_path, extension, reset_importsys):
    """
    Provide a function to write single gtscript module with a given name.

    The name should be unique per test function, to avoid any import caching effects.
    """

    def run(module_name):
        gts_file = tmp_path / f"{module_name}{extension}"
        gts_file.write_text(
            (
                "# [GT] using-dsl: gtscript\n"
                "\n"
                "\n"
                "@lazy_stencil()\n"
                "def single_file(a: Field[float]):\n"
                "    with computation(PARALLEL), interval(...):\n"
                "        a = 1.\n"
                "\n"
                "\n"
                "SENTINEL = 5\n"
            )
        )
        return gts_file

    yield run


@pytest.fixture
def make_two_files(tmp_path, extension, reset_importsys):
    """
    Provide a factory for two gtscript modules next to each other.

    The name prefix given must be unique per test function.
    the two modules can be imported as <prefix>_one and <prefix>_two.

    * module one imports module two
    * module two has a function that imports module one

    the resulting file structure::

        <tmp-dir>
        |- <prefix>_one.<ext>
        +- <prefix>_two.<ext>
    """

    def run(prefix):
        module_one = tmp_path / f"{prefix}_one{extension}"
        module_two = tmp_path / f"{prefix}_two{extension}"

        module_one.write_text(
            (
                "# [GT] using-dsl: gtscript\n"
                "import numpy\n"  # make sure thirdparty pkg can be imported
                f"import {prefix}_two\n"
                f"from {prefix}_two import square\n"
                "\n"
                "\n"
                "FT = Field[numpy.float64]\n"  # ensure gtscript import still works
                "SENTINEL = 43\n"
                "\n"
                "\n"
                "@lazy_stencil()\n"
                "def a_square_b(a: FT, b: FT):\n"
                "    with computation(PARALLEL), interval(...):\n"
                "        a = square(b)\n"
            )
        )

        module_two.write_text(
            (
                "# [GT] using-dsl: gtscript\n"
                "import numpy"  # make sure thirdparty pkg can be imported
                "\n"
                "\n"
                "FT = Field[numpy.float32]\n"  # ensure gtscript import still works
                "SENTINEL = 47\n"
                "\n"
                "\n"
                "def get_sentinel_one():\n"
                f"    from {prefix}_one import SENTINEL as sentinel_one\n"
                "    return sentinel_one\n"
                "\n"
                "\n"
                "@function\n"
                "def square(field):\n"
                "    return field * field\n"
            )
        )
        return module_one, module_two

    yield run


@pytest.fixture
def make_package(tmp_path, extension, reset_importsys):
    """
    Provide a factory for a gtscript file and package.

    test-function unique prefix required.


    the resulting file structure::

        <tmp-dir>/
        |- <prefix>.<ext>
        |- <prefix>_lib/
        |  |- __init__.py
        |  |- <prefix>_sub1.<ext>
        |  +- <prefix>_sub2/
        |  |  |- __init__.py
        |  |  +- <prefix>_sub_sub.<ext>
    """

    def run(prefix):
        module_file = tmp_path / f"{prefix}{extension}"
        lib_path = tmp_path / f"{prefix}_lib"
        lib_init = lib_path / "__init__.py"
        sub_1_path = lib_path / f"{prefix}_sub1{extension}"
        sub_2_path = lib_path / f"{prefix}_sub2"
        sub_2_init = sub_2_path / "__init__.py"
        sub_sub_path = sub_2_path / f"{prefix}_sub_sub{extension}"

        # set up folder structure
        lib_path.mkdir()
        sub_2_path.mkdir()

        # write files
        module_file.write_text(
            (
                "# [GT] using-dsl: gtscript\n\n\n"
                f"from {prefix}_lib import CONST, sf1, sss\n"
                "SENTINEL = 1\n\n\n"
                "@function\n"
                "def mf(a, *, const):\n"
                "    return SENTINEL * const\n\n\n"
                '@lazy_stencil(externals={"C": CONST})\n'
                "def ms(a: Field[float]):\n"
                "    from __externals__ import C\n"
                "    with computation(PARALLEL), interval(...):\n"
                "        b = mf(a, const=C)\n"
                "        a = sf1(b)\n"
            )
        )
        lib_init.write_text(
            (
                f"from . import {prefix}_sub2\n"
                f"from .{prefix}_sub1 import sf1\n"
                f"from .{prefix}_sub2 import CONST, sss\n"
            )
        )
        sub_1_path.write_text(
            (
                "# [GT] using-dsl: gtscript\n\n"
                f"from .{prefix}_sub2.{prefix}_sub_sub import SENTINEL as SS_SENT\n\n\n"
                "@function\n"
                "def sf1(b):\n"
                "    return b + SS_SENT\n\n\n"
                "SENTINEL = 2\n"
            )
        )
        sub_2_init.write_text(
            (
                "# [GT] using-dsl: gtscript\n\n"
                f"from ..{prefix}_sub1 import SENTINEL as S1_SENT\n"
                f"from . import {prefix}_sub_sub\n"
                f"from .{prefix}_sub_sub import sss\n\n\n"
                "CONST = 3.14\n"
            )
        )
        sub_sub_path.write_text(
            (
                "# [GT] using-dsl: gtscript\n\n\n"
                "@lazy_stencil()\n"
                "def sss(a: Field[float]):\n"
                "    with computation(PARALLEL), interval(...):\n"
                "        a = 0\n\n\n"
                "SENTINEL = 3\n"
            )
        )
        return module_file

    yield run


def test_single_file_import(make_single_file, reset_importsys):
    """Test using basic import statement for a gtscript module."""
    single_file = make_single_file("sfile_imp")
    gtscript_imports.enable(search_path=[single_file.parent])
    assert str(single_file.parent) in sys.path

    import sfile_imp

    assert sfile_imp.SENTINEL == 5


def test_single_file_nosearchpath(make_single_file):
    """Test import statement without giving a search path."""
    single_file = make_single_file("sfile_nsp")
    gtscript_imports.enable()
    sys.path.append(single_file.parent)

    import sfile_nsp

    assert sfile_nsp.SENTINEL == 5


def test_single_file_builddir(tmp_path, make_single_file):
    """Test importer with build folder."""
    single_file = make_single_file("sfile_bf")
    gen_dir = tmp_path / "generate"
    gtscript_imports.enable(search_path=[single_file.parent], generate_path=gen_dir)

    import sfile_bf

    assert sfile_bf.SENTINEL == 5
    assert (gen_dir / "sfile_bf.py").exists()


def test_single_file_insource(make_single_file):
    """Test importer with in-source generation."""
    single_file = make_single_file("sfile_is")
    gtscript_imports.enable(search_path=[single_file.parent], in_source=True)

    import sfile_is

    assert sfile_is.SENTINEL == 5
    assert (single_file.parent / "sfile_is.py").exists()


def test_single_file_import_module(make_single_file):
    """Test importlib.import_module in combination with importer."""
    from importlib import import_module

    single_file = make_single_file("sfile_im")
    gtscript_imports.enable(search_path=[single_file.parent])
    print(single_file.stem)
    print(single_file.suffixes)
    sfile_im = import_module(single_file.stem.split(".")[0])
    assert sfile_im.SENTINEL == 5


def test_single_file_load(make_single_file):
    """Test the finder's load_module method."""
    single_file = make_single_file("sfile_load")
    finder = gtscript_imports.enable(search_path=[single_file.parent])
    sf_module = finder.load_module("sfile_load")
    assert sf_module.SENTINEL == 5


def test_twofile_import(make_two_files):
    """Test importing from the same folder."""
    file_one, file_two = make_two_files("simple")
    gtscript_imports.enable(search_path=[file_one.parent])

    import simple_one

    assert simple_one.SENTINEL == 43
    assert simple_one.simple_two.SENTINEL == 47

    from simple_one import simple_two

    assert simple_two.get_sentinel_one() == simple_one.SENTINEL


def test_pkg_import(make_package):
    """Test importing from package in same folder."""
    main_file = make_package("pkg_simple")
    gtscript_imports.enable(search_path=[main_file.parent])

    import pkg_simple

    assert pkg_simple.SENTINEL == 1
    assert pkg_simple.CONST == 3.14
    assert pkg_simple.mf
    assert pkg_simple.ms
    assert pkg_simple.sf1
    assert pkg_simple.sss
