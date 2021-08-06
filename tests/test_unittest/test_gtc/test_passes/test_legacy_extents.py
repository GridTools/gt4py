# flake8: noqa: F841
from gt4py import gtscript as gs
from gt4py.backend import from_name
from gt4py.gtscript import PARALLEL, computation, interval
from gt4py.stencil_builder import StencilBuilder
from gtc.passes.gtir_legacy_extents import compute_legacy_extents
from gtc.passes.gtir_pipeline import prune_unused_parameters


def prepare_gtir(builder: StencilBuilder):
    return builder.gtir_pipeline.full(skip=[prune_unused_parameters])


def test_noextents():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[0, 0, 0]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_single_pos_offset():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[1, 0, 0]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_single_neg_offset():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[0, -1, 0]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_single_k_offset():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[0, 0, 1]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_offset_chain():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[1, 0, 1]
        with computation(PARALLEL), interval(...):
            field_b = field_a[1, 0, 0]
        with computation(PARALLEL), interval(...):
            tmp = field_b[0, -1, 0] + field_b[0, 1, 0]
            field_a = tmp[0, 0, 0] + tmp[0, 0, -1]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_ij():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[gs.IJ, float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[0, 1]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_j():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[gs.J, float]):
        with computation(PARALLEL), interval(...):
            field_a = field_b[1] + field_b[-2]

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_unreferenced():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            field_a = 1.0

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext


def test_field_if():
    def stencil(field_a: gs.Field[float], field_b: gs.Field[float]):
        with computation(PARALLEL), interval(...):
            if field_b[0, 1, 0] < 0.1:
                if field_b[1, 0, 0] > 1.0:
                    field_a = 0
                else:
                    field_a = 1
            else:
                tmp = -field_b[0, 1, 0]
                field_a = tmp

    builder = StencilBuilder(stencil, backend=from_name("debug"))
    old_ext = builder.implementation_ir.fields_extents
    legacy_ext = compute_legacy_extents(prepare_gtir(builder))

    for name, ext in old_ext.items():
        assert legacy_ext[name] == ext
