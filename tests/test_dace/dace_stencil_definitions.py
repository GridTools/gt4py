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
    sdfg,
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
    implementation.sdfg = sdfg
    return implementation


@register
def make_horizontal_diffusion():
    sdfg = dace.SDFG("horizontal_diffusion")
    sdfg.add_array("in_field", shape=[I + 4, J + 4, K], dtype=dace.float64)
    sdfg.add_array("out_field", shape=[I, J, K], dtype=dace.float64)
    sdfg.add_array("coeff", shape=[I, J, K], dtype=dace.float64)
    sdfg.add_transient("lap_field", shape=["I + 2", "J + 2", K], dtype=dace.float64)
    sdfg.add_transient("flx_field", shape=["I + 1", "J", K], dtype=dace.float64)
    sdfg.add_transient("fly_field", shape=["I", "J + 1", K], dtype=dace.float64)

    # in_field = state.add_read("in_field")
    # lap_field = state.add_write('lap_field')

    #     @dace.map
    #     def lap(i: _[-1 : I + 1], j: _[-1 : J + 1], k: _[0:K]):
    #         in_field_r << in_field[i + 2 + 1, j + 2, k]
    #         in_field_l << in_field[i + 2 - 1, j + 2, k]
    #         in_field_f << in_field[i + 2, j + 2 + 1, k]
    #         in_field_b << in_field[i + 2, j + 2 - 1, k]
    #         in_field_c << in_field[i + 2, j + 2, k]
    #         lf >> lap_field[i + 1, j + 1, k]
    #
    #         lf = 4.0 * in_field_c - (in_field_l + in_field_r + in_field_f + in_field_b)

    lap_state = sdfg.add_state()
    lap, lap_entry, lap_exit = lap_state.add_mapped_tasklet(
        "lap",  # name
        dict(i="-1:I+1", j="-1:J+1", k="0:K"),  # map range
        dict(
            in_f=dace.Memlet.simple("in_field", "i+1:i+4, j+1:j+4, k", num_accesses=5)
        ),  # input memlets
        """
lap_f = 4.0 * in_f[1, 1] - (in_f[0, 1] + in_f[2, 1] + in_f[1, 0] + in_f[1, 2])

        """,
        dict(
            lap_f=dace.Memlet.simple("lap_field", "i+1, j+1, k", num_accesses=1)
        ),  # output memlets
    )

    lap_state.add_edge(
        lap_state.add_read("in_field"),
        None,
        lap_entry,
        None,
        memlet=dace.Memlet.simple("in_field", "0:I+4, 0:J+4, 0:K"),
    )
    lap_state.add_edge(
        lap_exit,
        None,
        lap_state.add_write("lap_field"),
        None,
        memlet=dace.Memlet.simple("lap_field", "0:I+2, 0:J+2, 0:K"),
    )
    lap_state.fill_scope_connectors()

    #     @dace.map
    #     def flx(i: _[-1:I], j: _[0:J], k: _[0:K]):
    #         lf_r << lap_field[i + 1 + 1, j + 1, k]
    #         lf_c << lap_field[i + 1, j + 1, k]
    #         if_r << in_field[i + 2 + 1, j + 2, k]
    #         if_c << in_field[i + 2, j + 2, k]
    #         ff >> flx_field[i + 1, j, k]
    #         res = lf_r - lf_c
    #         ff = 0 if (res * (if_r - if_c)) > 0 else res
    flx_state = sdfg.add_state()
    flx, flx_entry, flx_exit = flx_state.add_mapped_tasklet(
        "flx",  # name
        dict(i="-1:I", j="0:J", k="0:K"),  # map range
        dict(
            in_f=dace.Memlet.simple("in_field", "i+2:i+4, j+2, k", num_accesses=2),
            lap_f=dace.Memlet.simple("lap_field", "i+1:i+3, j+1, k", num_accesses=2),
        ),  # input memlets
        """
res = lap_f[1] - lap_f[0]
flx_f = 0 if (res * (in_f[1] - in_f[0])) > 0 else res
        """,
        dict(flx_f=dace.Memlet.simple("flx_field", "i+1, j, k", num_accesses=1)),  # output memlets
    )
    flx_state.add_edge(
        flx_state.add_read("in_field"),
        None,
        flx_entry,
        None,
        memlet=dace.Memlet.simple("in_field", "1:I+3, 2:J+2, 0:K"),
    )
    flx_state.add_edge(
        flx_state.add_read("lap_field"),
        None,
        flx_entry,
        None,
        memlet=dace.Memlet.simple("lap_field", "0:I+2, 1:J+1, 0:K"),
    )
    flx_state.add_edge(
        flx_exit,
        None,
        flx_state.add_write("flx_field"),
        None,
        memlet=dace.Memlet.simple("flx_field", "0:I+1, 0:J, 0:K"),
    )
    flx_state.fill_scope_connectors()
    #     @dace.map
    #     def fly(i: _[0:I], j: _[-1:J], k: _[0:K]):
    #         lf_f << lap_field(1)[i + 1, j + 1 + 1, k]
    #         lf_c << lap_field[i + 1, j + 1, k]
    #         if_f << in_field[i + 2, j + 2 + 1, k]
    #         if_c << in_field[i + 2, j + 2, k]
    #         ff >> fly_field[i, j + 1, k]
    #         res = lf_f - lf_c
    #         ff = 0 if (res * (if_f - if_c)) > 0 else res
    fly_state = sdfg.add_state()
    fly, fly_entry, fly_exit = fly_state.add_mapped_tasklet(
        "fly",  # name
        dict(i="0:I", j="-1:J", k="0:K"),  # map range
        dict(
            in_f=dace.Memlet.simple("in_field", "i+2, j+2:j+4, k", num_accesses=2),
            lap_f=dace.Memlet.simple("lap_field", "i+1, j+1:j+3, k", num_accesses=2),
        ),  # input memlets
        """
res = lap_f[1] - lap_f[0]
fly_f = 0 if (res * (in_f[1] - in_f[0])) > 0 else res
        """,
        dict(fly_f=dace.Memlet.simple("fly_field", "i, j+1, k")),  # output memlets
    )
    fly_state.add_edge(
        fly_state.add_read("in_field"),
        None,
        fly_entry,
        None,
        memlet=dace.Memlet.simple("in_field", "2:I+2, 1:J+3, 0:K"),
    )
    fly_state.add_edge(
        fly_state.add_read("lap_field"),
        None,
        fly_entry,
        None,
        memlet=dace.Memlet.simple("lap_field", "1:I+1, 0:J+2, 0:K"),
    )
    fly_state.add_edge(
        fly_exit,
        None,
        fly_state.add_write("fly_field"),
        None,
        memlet=dace.Memlet.simple("fly_field", "0:I, 0:J+1, 0:K"),
    )
    fly_state.fill_scope_connectors()

    #     @dace.map
    #     def out(i: _[0:I], j: _[0:J], k: _[0:K]):
    #         if_c << in_field[i + 2, j + 2, k]
    #         cf << coeff[i, j, k]
    #         fx_f_c << flx_field[i + 1, j, k]
    #         fx_f_l << flx_field[i + 1 - 1, j, k]
    #         fy_f_c << fly_field[i, j + 1, k]
    #         fy_f_b << fly_field[i, j + 1 - 1, k]
    #         of >> out_field[i, j, k]
    #         of = if_c - cf * (fx_f_c - fx_f_l + fy_f_c - fy_f_b)
    out_state = sdfg.add_state()
    out, out_entry, out_exit = out_state.add_mapped_tasklet(
        "out",  # name
        dict(i="0:I", j="0:J", k="0:K"),  # map range
        dict(
            in_f=dace.Memlet.simple("in_field", "i+2, j+2, k"),
            c=dace.Memlet.simple("coeff", "i, j, k"),
            flx_f=dace.Memlet.simple("flx_field", "i:i+2, j, k"),
            fly_f=dace.Memlet.simple("fly_field", "i, j:j+2, k"),
        ),  # input memlet
        """
out_f = in_f - c * (flx_f[1] - flx_f[0] + fly_f[1] - fly_f[0])
    """,
        dict(out_f=dace.Memlet.simple("out_field", "i, j, k")),  # output memlets
    )
    out_state.add_edge(
        out_state.add_read("in_field"),
        None,
        out_entry,
        None,
        memlet=dace.Memlet.simple("in_field", "2:I+2, 2:J+2, 0:K"),
    )
    out_state.add_edge(
        out_state.add_read("coeff"),
        None,
        out_entry,
        None,
        memlet=dace.Memlet.simple("coeff", "0:I, 0:J, 0:K"),
    )
    out_state.add_edge(
        out_state.add_read("flx_field"),
        None,
        out_entry,
        None,
        memlet=dace.Memlet.simple("flx_field", "0:I+1, 0:J, 0:K"),
    )
    out_state.add_edge(
        out_state.add_read("fly_field"),
        None,
        out_entry,
        None,
        memlet=dace.Memlet.simple("fly_field", "0:I, 0:J+1, 0:K"),
    )
    out_state.add_edge(
        out_exit,
        None,
        out_state.add_write("out_field"),
        None,
        memlet=dace.Memlet.simple("out_field", "0:I, 0:J, 0:K"),
    )
    sdfg.fill_scope_connectors()
    #     sdfg.add_edge(lap_state, flx_state, dace.InterstateEdge())
    #     sdfg.add_edge(flx_state, fly_state, dace.InterstateEdge())
    #     sdfg.add_edge(fly_state, out_state, dace.InterstateEdge())
    #
    #     # @dace.program
    #     # def horizontal_diffusion(
    #     #     in_field: dace.float64[I + 4, J + 4, K],
    #     #     out_field: dace.float64[I, J, K],
    #     #     coeff: dace.float64[I, J, K],
    #     # ):
    #     #     lap_field = dace.define_local(["I + 2", "J + 2", K], dtype=in_field.dtype)
    #     #     flx_field = dace.define_local(["I + 1", J, K], dtype=in_field.dtype)
    #     #     fly_field = dace.define_local([I, "J + 1", K], dtype=in_field.dtype)
    #     #

    #     #
    #
    #     #

    #     #

    sdfg.add_edge(lap_state, flx_state, edge=dace.InterstateEdge())
    sdfg.add_edge(flx_state, fly_state, edge=dace.InterstateEdge())
    sdfg.add_edge(fly_state, out_state, edge=dace.InterstateEdge())
    sdfg.apply_strict_transformations()
    sdfg.validate()

    import json

    with open("tmp.sdfg", "w") as sdfgfile:
        json.dump(sdfg.to_json(), sdfgfile)
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
        sdfg=sdfg,
    )
    # from ..test_def_ir.test_ir_transformation import analyze
    # ref_iir = analyze("horizontal_diffusion")
    return implementation


@register
def make_tridiagonal_solver():

    sdfg = dace.SDFG("tridiagonal_solver")
    sdfg.add_array("inf", shape=[I, J, K], dtype=dace.float64)
    sdfg.add_array("diag", shape=[I, J, K], dtype=dace.float64)
    sdfg.add_array("sup", shape=[I, J, K], dtype=dace.float64)
    sdfg.add_array("rhs", shape=[I, J, K], dtype=dace.float64)
    sdfg.add_array("out", shape=[I, J, K], dtype=dace.float64)
    #     for k in range(0, 1):
    #
    #         @dace.map
    #         def fwd_top(i: _[0:I], j: _[0:J]):
    #             s_in << sup[i, j, k]
    #             r_in << rhs[i, j, k]
    #             d << diag[i, j, k]
    #             s >> sup[i, j, k]
    #             r >> rhs[i, j, k]
    #             s = s_in / d
    #             r = r_in / d
    state = sdfg.add_state()
    sup_in = state.add_read("sup")
    rhs_in = state.add_read("rhs")
    diag_in = state.add_read("diag")
    sup_out = state.add_write("sup")
    rhs_out = state.add_write("rhs")
    _, state_entry, state_exit = state.add_mapped_tasklet(
        "fwd_top",  # name
        dict(i="0:I", j="0:J"),  # map range
        dict(
            s_in=dace.Memlet.simple("sup", "i, j, k", num_accesses=1),
            r_in=dace.Memlet.simple("rhs", "i, j, k", num_accesses=1),
            d=dace.Memlet.simple("diag", "i, j, k", num_accesses=2),
        ),  # input memlets
        """
s = s_in / d
r = r_in / d
        """,
        dict(
            s=dace.Memlet.simple("sup", "i, j, k", num_accesses=1),
            r=dace.Memlet.simple("rhs", "i, j, k", num_accesses=1),
        ),  # output memlets
    )
    state.add_edge(
        sup_in, None, state_entry, None, memlet=dace.Memlet.simple("sup", "0:I, 0:J, 0:1")
    )
    state.add_edge(
        rhs_in, None, state_entry, None, memlet=dace.Memlet.simple("rhs", "0:I, 0:J, 0:1")
    )
    state.add_edge(
        diag_in, None, state_entry, None, memlet=dace.Memlet.simple("diag", "0:I, 0:J, 0:1")
    )
    state.add_edge(
        state_exit, None, sup_out, None, memlet=dace.Memlet.simple("sup", "0:I, 0:J, 0:1")
    )
    state.add_edge(
        state_exit, None, rhs_out, None, memlet=dace.Memlet.simple("rhs", "0:I, 0:J, 0:1")
    )
    entry_state1, _, exit_state1 = sdfg.add_loop(None, state, None, "k", "0", "k<1", "k+1")
    #     for k in range(1, K):
    #
    #         @dace.map
    #         def fwd_body(i: _[0:I], j: _[0:J]):
    #             s_in << sup[i, j, k]
    #             d << diag[i, j, k]
    #             s_l << sup[i, j, k - 1]
    #             i_c << inf[i, j, k]
    #             r_l << rhs[i, j, k - 1]
    #             r_in << rhs[i, j, k]
    #             s >> sup[i, j, k]
    #             r >> rhs[i, j, k]
    #             s = s_in / (d - s_l * i_c)
    #             r = (r_in - i_c * r_l) / (d - s_l * i_c)
    state = sdfg.add_state()
    sup_in = state.add_read("sup")
    rhs_in = state.add_read("rhs")
    diag_in = state.add_read("diag")
    inf_in = state.add_read("inf")
    sup_out = state.add_write("sup")
    rhs_out = state.add_write("rhs")
    _, state_entry, state_exit = state.add_mapped_tasklet(
        "fwd_body",  # name
        dict(i="0:I", j="0:J"),  # map range
        dict(
            s_in=dace.Memlet.simple("sup", "i, j, k-1:k+1", num_accesses=3),
            r_in=dace.Memlet.simple("rhs", "i, j, k-1:k+1", num_accesses=2),
            i_in=dace.Memlet.simple("inf", "i, j, k", num_accesses=3),
            d=dace.Memlet.simple("diag", "i, j, k", num_accesses=2),
        ),  # input memlets
        """
s = s_in[1] / (d - s_in[0] * i_in)
r = (r_in[1] - i_in * r_in[0]) / (d - s_in[0] * i_in)
        """,
        dict(
            s=dace.Memlet.simple("sup", "i, j, k", num_accesses=1),
            r=dace.Memlet.simple("rhs", "i, j, k", num_accesses=1),
        ),  # output memlets
    )
    state.add_edge(
        sup_in, None, state_entry, None, memlet=dace.Memlet.simple("sup", "0:I, 0:J, k-1:k+1")
    )
    state.add_edge(
        rhs_in, None, state_entry, None, memlet=dace.Memlet.simple("rhs", "0:I, 0:J, k-1:k+1")
    )
    state.add_edge(
        diag_in, None, state_entry, None, memlet=dace.Memlet.simple("diag", "0:I, 0:J, k")
    )
    state.add_edge(
        inf_in, None, state_entry, None, memlet=dace.Memlet.simple("inf", "0:I, 0:J, k")
    )
    state.add_edge(
        state_exit, None, sup_out, None, memlet=dace.Memlet.simple("sup", "0:I, 0:J, k")
    )
    state.add_edge(
        state_exit, None, rhs_out, None, memlet=dace.Memlet.simple("rhs", "0:I, 0:J, k")
    )
    entry_state2, _, exit_state2 = sdfg.add_loop(None, state, None, "k", "1", "k<K", "k+1")

    #     for k in range(K - 1, K - 2, -1):
    #
    #         @dace.map
    #         def back_bottom(i: _[0:I], j: _[0:J]):
    #             r << rhs[i, j, k]
    #             o >> out[i, j, k]
    #             o = r
    state = sdfg.add_state()
    rhs_in = state.add_read("rhs")
    out_out = state.add_write("out")
    _, state_entry, state_exit = state.add_mapped_tasklet(
        "back_bottom",  # name
        dict(i="0:I", j="0:J"),  # map range
        dict(r_in=dace.Memlet.simple("rhs", "i, j, k", num_accesses=1),),  # input memlets
        """
o = r_in
        """,
        dict(o=dace.Memlet.simple("out", "i, j, k", num_accesses=1),),  # output memlets
    )
    state.add_edge(
        rhs_in, None, state_entry, None, memlet=dace.Memlet.simple("rhs", "0:I, 0:J, k")
    )
    state.add_edge(
        state_exit, None, out_out, None, memlet=dace.Memlet.simple("out", "0:I, 0:J, k")
    )
    entry_state3, _, exit_state3 = sdfg.add_loop(None, state, None, "k", "K-1", "k>K-2", "k-1")

    #     for k in range(K - 2, -1, -1):
    #
    #         @dace.map
    #         def back_body(i: _[0:I], j: _[0:J]):
    #             r << rhs[i, j, k]
    #             s << sup[i, j, k]
    #             o_h << out[i, j, k + 1]
    #             o >> out[i, j, k]
    #             o = r - s * o_h
    state = sdfg.add_state()
    rhs_in = state.add_read("rhs")
    sup_in = state.add_read("sup")
    out_in = state.add_read("out")
    out_out = state.add_write("out")
    _, state_entry, state_exit = state.add_mapped_tasklet(
        "back_body",  # name
        dict(i="0:I", j="0:J"),  # map range
        dict(
            r_in=dace.Memlet.simple("rhs", "i, j, k", num_accesses=1),
            s_in=dace.Memlet.simple("sup", "i, j, k", num_accesses=1),
            o_in=dace.Memlet.simple("out", "i, j, k+1", num_accesses=1),
        ),  # input memlets
        """
o = r_in - s_in * o_in
        """,
        dict(o=dace.Memlet.simple("out", "i, j, k", num_accesses=1),),  # output memlets
    )
    state.add_edge(
        rhs_in, None, state_entry, None, memlet=dace.Memlet.simple("rhs", "0:I, 0:J, k")
    )
    state.add_edge(
        sup_in, None, state_entry, None, memlet=dace.Memlet.simple("sup", "0:I, 0:J, k")
    )
    state.add_edge(
        out_in, None, state_entry, None, memlet=dace.Memlet.simple("out", "0:I, 0:J, k+1")
    )
    state.add_edge(
        state_exit, None, out_out, None, memlet=dace.Memlet.simple("out", "0:I, 0:J, k")
    )
    entry_state4, _, exit_state4 = sdfg.add_loop(None, state, None, "k", "K-2", "k>-1", "k-1")

    sdfg.add_edge(exit_state1, entry_state2, dace.InterstateEdge())
    sdfg.add_edge(exit_state2, entry_state3, dace.InterstateEdge())
    sdfg.add_edge(exit_state3, entry_state4, dace.InterstateEdge())
    sdfg.fill_scope_connectors()
    sdfg.validate()
    sdfg.apply_strict_transformations()
    import json

    with open("tmp.sdfg", "w") as sdfgfile:
        json.dump(sdfg.to_json(), sdfgfile)
    # sdfg.fill_scope_connectors()
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
        sdfg=sdfg,
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
