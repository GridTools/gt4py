import dace
from gt4py import utils as gt_utils
from gt4py import backend as gt_backend
from gt4py.definitions import StencilID
import gt4py

REGISTRY = gt_utils.Registry()


def register(dace_program):
    REGISTRY.register(dace_program._name, dace_program)
    return dace_program


def register(iir_factory):
    if iir_factory.__name__.startswith("make_"):
        name = iir_factory.__name__[len("make_") :]
    else:
        raise ValueError("Name of stencil factory must start with 'make_'")
    return REGISTRY.register(name, iir_factory)


def build_dace_stencil(name, options, backend="dace", *, id_version="xxxxxx"):
    if isinstance(backend, str):
        backend = gt_backend.from_name(backend)
    if not issubclass(backend, gt_backend.Backend):
        raise TypeError("Backend must be a backend identifier string or a gt4py Backend class.")

    iir_factory = REGISTRY[name]
    iir = iir_factory()
    stencil_id = StencilID("{}.{}".format(options.module, options.name), id_version)

    # if options.rebuild:
    #     Force recompilation
    # stencil_class = None
    # else:
    #     Use cached version (if id_version matches)
    # stencil_class = backend.load(stencil_id, None, options)
    #
    # if stencil_class is None:
    # stencil_class = backend.build(stencil_id, iir, None, options)
    stencil_class = backend.load(stencil_id, None, options)
    # stencil_class = None
    if stencil_class is None:
        stencil_class = backend.build(stencil_id, iir, None, options)

    stencil_implementation = stencil_class()

    return stencil_implementation


I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


def make_implementation(
    name: str,
    args_list: list,
    fields_with_description: dict,
    parameters_with_type: dict,
    dace_program,
    domain=None,
    k_axis_splitters=None,
    externals=None,
    sources=None,
):
    from gt4py.analysis.passes import DataTypePass

    api_signature = gt4py.ir.utils.make_api_signature(args_list)

    domain = domain or gt4py.ir.Domain.LatLonGrid()
    # if k_axis_splitters is not None:
    #     # Assume: ["var_name"] or  [("var_name", index)]
    #     refs = []
    #     for item in k_axis_splitters:
    #         if isinstance(item, tuple):
    #             refs.append(VarRef(name=item[0], index=Index([item[1]])))
    #         else:
    #             refs.append(VarRef(name=item))
    #     axis_splitters = {domain.sequential_axis.name: refs}
    # else:
    #     axis_splitters = {}
    axis_splitters = None

    fields_decls = {}
    fields_extents = {}
    for field_name, description in fields_with_description.items():
        extent = description.pop("extent", gt4py.ir.Extent.zeros())
        description.setdefault("layout_id", repr(extent))
        fields_extents[field_name] = gt4py.ir.Extent(extent)
        fields_decls[field_name] = gt4py.ir.utils.make_field_decl(name=field_name, **description)

    parameter_decls = {}
    for key, value in parameters_with_type.items():
        if isinstance(value, tuple):
            assert len(value) == 2
            data_type = value[0]
            length = value[1]
        else:
            data_type = value
            length = 0
        parameter_decls[key] = gt4py.ir.VarDecl(
            name=key, data_type=gt4py.ir.DataType.from_dtype(data_type), length=length, is_api=True
        )

    implementation = gt4py.ir.StencilImplementation(
        name=name,
        api_signature=api_signature,
        domain=domain,
        axis_splitters_var=axis_splitters,
        fields=fields_decls,
        parameters=parameter_decls,
        multi_stages=[],
        fields_extents=fields_extents,
        externals=externals,
        sources=sources,
    )
    #
    data_type_visitor = DataTypePass.CollectDataTypes()
    data_type_visitor(implementation)
    implementation.dace_program = dace_program
    return implementation


@register
def make_horizontal_diffusion():
    @dace.program
    def horizontal_diffusion(
        in_field: dace.float64[I + 4, J + 4, K],
        out_field: dace.float64[I, J, K],
        coeff: dace.float64[I, J, K],
    ):
        lap_field = dace.define_local(["I + 2", "J + 2", K], dtype=in_field.dtype)
        flx_field = dace.define_local(["I + 1", J, K], dtype=in_field.dtype)
        fly_field = dace.define_local([I, "J + 1", K], dtype=in_field.dtype)

        @dace.map
        def lap(i: _[-1 : I + 1], j: _[-1 : J + 1], k: _[0:K]):
            in_field_r << in_field[i + 2 + 1, j + 2, k]
            in_field_l << in_field[i + 2 - 1, j + 2, k]
            in_field_f << in_field[i + 2, j + 2 + 1, k]
            in_field_b << in_field[i + 2, j + 2 - 1, k]
            in_field_c << in_field[i + 2, j + 2, k]
            lf >> lap_field[i + 1, j + 1, k]

            lf = 4.0 * in_field_c - (in_field_l + in_field_r + in_field_f + in_field_b)

        @dace.map
        def flx(i: _[-1:I], j: _[0:J], k: _[0:K]):
            lf_r << lap_field[i + 1 + 1, j + 1, k]
            lf_c << lap_field[i + 1, j + 1, k]
            if_r << in_field[i + 2 + 1, j + 2, k]
            if_c << in_field[i + 2, j + 2, k]
            ff >> flx_field[i + 1, j, k]
            res = lf_r - lf_c
            ff = 0 if (res * (if_r - if_c)) > 0 else res

        @dace.map
        def fly(i: _[0:I], j: _[-1:J], k: _[0:K]):
            lf_f << lap_field[i + 1, j + 1 + 1, k]
            lf_c << lap_field[i + 1, j + 1, k]
            if_f << in_field[i + 2, j + 2 + 1, k]
            if_c << in_field[i + 2, j + 2, k]
            ff >> fly_field[i, j + 1, k]
            res = lf_f - lf_c
            ff = 0 if (res * (if_f - if_c)) > 0 else res

        @dace.map
        def out(i: _[0:I], j: _[0:J], k: _[0:K]):
            if_c << in_field[i + 2, j + 2, k]
            cf << coeff[i, j, k]
            fx_f_c << flx_field[i + 1, j, k]
            fx_f_l << flx_field[i + 1 - 1, j, k]
            fy_f_c << fly_field[i, j + 1, k]
            fy_f_b << fly_field[i, j + 1 - 1, k]
            of >> out_field[i, j, k]
            of = if_c - cf * (fx_f_c - fx_f_l + fy_f_c - fy_f_b)

    implementation = make_implementation(
        "horizontal_diffusion",
        args_list=["in_field", "coeff", "out_field"],
        fields_with_description={
            "in_field": dict(is_api=True, extent=[(-2, 2), (-2, 2), (0, 0)]),
            "coeff": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "out_field": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        domain=None,
        dace_program=horizontal_diffusion,
    )
    # from ..test_def_ir.test_ir_transformation import analyze
    # ref_iir = analyze("horizontal_diffusion")
    return implementation


@register
def make_tridiagonal_solver():
    @dace.program
    def tridiagonal_solver(
        inf: dace.float64[I, J, K],
        diag: dace.float64[I, J, K],
        sup: dace.float64[I, J, K],
        rhs: dace.float64[I, J, K],
        out: dace.float64[I, J, K],
    ):
        for k in range(0, 1):

            @dace.map
            def fwd_top(i: _[0:I], j: _[0:J]):
                s_in << sup[i, j, k]
                r_in << rhs[i, j, k]
                d << diag[i, j, k]
                s >> sup[i, j, k]
                r >> rhs[i, j, k]
                s = s_in / d
                r = r_in / d

        for k in range(1, K):

            @dace.map
            def fwd_body(i: _[0:I], j: _[0:J]):
                s_in << sup[i, j, k]
                d << diag[i, j, k]
                s_l << sup[i, j, k - 1]
                i_c << inf[i, j, k]
                r_l << rhs[i, j, k - 1]
                r_in << rhs[i, j, k]
                s >> sup[i, j, k]
                r >> rhs[i, j, k]
                s = s_in / (d - s_l * i_c)
                r = (r_in - i_c * r_l) / (d - s_l * i_c)

        for k in range(K - 1, K - 2, -1):

            @dace.map
            def back_bottom(i: _[0:I], j: _[0:J]):
                r << rhs[i, j, k]
                o >> out[i, j, k]
                o = r

        for k in range(K - 2, -1, -1):

            @dace.map
            def back_body(i: _[0:I], j: _[0:J]):
                r << rhs[i, j, k]
                s << sup[i, j, k]
                o_h << out[i, j, k + 1]
                o >> out[i, j, k]
                o = r - s * o_h

    implementation = make_implementation(
        "tridiagonal_solver",
        args_list=["inf", "diag", "sup", "rhs", "out"],
        fields_with_description={
            "inf": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "diag": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "sup": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "rhs": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "out": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        dace_program=tridiagonal_solver,
    )
    return implementation


if __name__ == "__main__":
    from gt4py import backend as gt_backend
    from gt4py import definitions as gt_definitions
    from dace import SDFG

    opts = gt_definitions.BuildOptions(name="dacetest", module="__main__")
    iir = make_horizontal_diffusion()
    # print(iir.sdfg.filename)
    # so_file = '/home/gronerl/gt4py/tests/test_dace/.dacecache/horizontal_diffusion/build/libhorizontal_diffusion.so'
    # sdfg_file = '/home/gronerl/gt4py/tests/test_dace/.dacecache/horizontal_diffusion/program.sdfg'
    # from dace.codegen.compiler import CompiledSDFG, ReloadableDLL
    # dll = ReloadableDLL(so_file, "horizontal_diffusion")
    # sdfg = SDFG.from_file(sdfg_file)
    # compiled_sdfg = CompiledSDFG(sdfg, dll)
    # compiled_sdfg(I=5, J=5, K=5)
    backend = gt_backend.from_name("dace")
    stencil_id = gt_definitions.StencilID("{}.{}".format(opts.module, opts.name), "x")
    backend.build(stencil_id, iir, None, opts)
