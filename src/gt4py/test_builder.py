import sys
import io
import os
import jinja2
import numpy as np
import gt4py.backend as gt_backend


class TestBuilder:
    def __init__(self, stencil_object):
        self.stencil_short_name = stencil_object.options["name"]
        self.stencil_unique_name = stencil_object.__class__.__name__
        self.module = stencil_object.__class__.__module__

    def write_test(
        self,
        backend: str,
        domain: tuple,
        origins: dict,
        shapes: dict,
        field_args: dict,
        parameter_args: dict,
        out_indices=[],
    ):
        components = self.module.split(".")[1:]
        if "GT_CACHE_DIR_NAME" in os.environ:
            unit_test_dir = os.environ["GT_CACHE_DIR_NAME"]
        else:
            unit_test_dir = os.path.join(os.getcwd(), ".gt_cache")

        cpython_id = "py{major}{minor}_{api}".format(
            major=sys.version_info.major, minor=sys.version_info.minor, api=sys.api_version
        )

        unit_test_name = "unit_test.cpp"
        unit_test_dir = (
            os.path.join(
                unit_test_dir, cpython_id, backend.replace(":", ""), os.sep.join(components)
            )
            + "_pyext_BUILD"
        )

        data_dir = os.path.join(unit_test_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        arg_fields = []
        out_fields = []

        if len(out_indices) < 1:
            out_indices = [
                field_idx for field_idx, field_arg in enumerate(field_args) if "out" in field_arg
            ]
            if len(out_indices) < 1:
                out_indices = [
                    field_idx
                    for field_idx, field_arg in enumerate(field_args)
                    if field_arg in origins
                ]

        for field_idx, field_arg in enumerate(field_args):
            field = field_args[field_arg]
            str_io = io.StringIO()
            np.savetxt(str_io, field.data.flatten())

            data_path = os.path.join(data_dir, f"{field_arg}.csv")
            data_file = open(data_path, "w")
            data_file.write(str_io.getvalue().rstrip().replace("\n", ","))

            if field_arg in origins and field_arg in shapes:
                arg_fields.append(
                    dict(
                        name=field_arg,
                        dtype=str(field.dtype),
                        origin=origins[field_arg],
                        shape=shapes[field_arg],
                        stride=field.strides,
                        size=field.size,
                    )
                )

            if field_idx in out_indices:
                out_fields.append(dict(name=field_arg, dtype=str(field.dtype), size=field.size))

        parameters = []
        for param_arg in parameter_args:
            param = parameter_args[param_arg]
            parameters.append(dict(name=param_arg, dtype=type(param).__name__, value=str(param)))

        template_args = dict(
            arg_fields=arg_fields,
            domain=tuple(domain),
            parameters=parameters,
            out_fields=out_fields,
            stencil_short_name=self.stencil_short_name,
            stencil_unique_name=self.stencil_unique_name,
            test_path=unit_test_dir,
            backend=backend,
        )

        template_dir = gt_backend.GTPyExtGenerator.TEMPLATE_DIR
        template_file = open(os.path.join(template_dir, f"{unit_test_name}.in"), "r")
        template = jinja2.Template(template_file.read())
        unit_test_source = template.render(**template_args)

        unit_test_path = os.path.join(unit_test_dir, unit_test_name)
        unit_test_file = open(unit_test_path, "w")
        unit_test_file.write(unit_test_source)

        # 2nd pass: get paths and references to output field data...
        out_idx = 0
        for field_idx, field_arg in enumerate(field_args):
            if field_idx in out_indices:
                out_fields[out_idx]["data"] = field_args[field_arg]
                out_fields[out_idx]["path"] = os.path.join(
                    data_dir, out_fields[out_idx]["name"] + "_out.csv"
                )
                out_idx += 1

        return out_fields

    def write_output(self, out_fields: list, overwrite=True):
        for out_field in out_fields:
            if overwrite or not os.path.exists(out_field["path"]):
                str_io = io.StringIO()
                np.savetxt(str_io, out_field["data"].data.flatten())
                data_file = open(out_field["path"], "w")
                data_file.write(str_io.getvalue().rstrip().replace("\n", ","))
