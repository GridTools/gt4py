# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The pybind11 ``run_computation`` binding rendered by ``bindings_main_template``."""

from gt4py.cartesian.backend.gtc_common import bindings_main_template


ENTRY_PARAMS = [
    ("py::buffer", "py"),
    ("std::array<gt::int_t,3>", "py_origin"),
    ("py::buffer", "out"),
    ("std::array<gt::int_t,4>", "out_origin"),
    ("double", "alpha"),
]
SID_PARAMS = ["sid_py", "sid_out", "gridtools::stencil::global_parameter(alpha)"]


def _render() -> str:
    return bindings_main_template().render_values(
        name="computation",
        module_name="stencil_pyext",
        entry_params=ENTRY_PARAMS,
        sid_params=SID_PARAMS,
    )


def test_binding_takes_py_args_and_casts_each_parameter_in_order():
    code = _render()
    expected = [("std::array<gt::uint_t, 3>", "domain"), *ENTRY_PARAMS, ("py::object", "exec_info")]

    assert 'm.def("run_computation", [](py::args gt_pyext_args)' in code
    assert f"gt_pyext_args.size() != {len(expected)}" in code

    positions = []
    for index, (ctype, name) in enumerate(expected):
        cast = f'{ctype} {name} = gt_pyext_cast_arg<{ctype}>(gt_pyext_args, {index}, "{name}");'
        assert cast in code
        positions.append(code.index(cast))
    assert positions == sorted(positions)

    # the body is unchanged: the computation is called on the SIDs built from the parameters
    assert f"computation(domain)({','.join(SID_PARAMS)});" in code


def test_binding_docstring_carries_the_typed_signature():
    code = _render()
    signature = (
        "run_computation(std::array<gt::uint_t, 3> domain, py::buffer py, "
        "std::array<gt::int_t,3> py_origin, py::buffer out, std::array<gt::int_t,4> out_origin, "
        "double alpha, py::object exec_info)"
    )
    assert f'"Runs the given computation: {signature}"' in code
