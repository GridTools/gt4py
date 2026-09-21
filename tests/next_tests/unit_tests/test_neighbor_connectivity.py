# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import textwrap
import typing

import numpy as np
import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import (
    DimensionIndex,
    DimensionKind,
    LocalDimensionIndex,
    NeighborConnectivity,
    NeighborConnectivityType,
)
from gt4py.next.ffront import transform_utils
from gt4py.next.type_system import type_specifications as ts, type_translation


class Vertex(DimensionIndex): ...


class Edge(DimensionIndex): ...


class KDim(DimensionIndex, kind=DimensionKind.VERTICAL): ...


class V2E(NeighborConnectivity[Vertex, Edge], max_neighbors=4, min_neighbors=3):
    class Local(LocalDimensionIndex): ...


class E2V(NeighborConnectivity[Edge, Vertex]):
    class Local(LocalDimensionIndex): ...


class LsqCoeff(LocalDimensionIndex, size=3): ...


def _declare(source: str) -> dict:
    """
    Run `source` as the body of a throwaway module.

    Declarations have to be at module level, so error cases cannot simply be written inside
    the test function: the `<locals>` check would fire before the one under test.
    """
    namespace = {
        "__name__": __name__,
        "typing": typing,
        "DimensionIndex": DimensionIndex,
        "DimensionKind": DimensionKind,
        "LocalDimensionIndex": LocalDimensionIndex,
        "NeighborConnectivity": NeighborConnectivity,
        "Vertex": Vertex,
        "Edge": Edge,
        "KDim": KDim,
        "V2E": V2E,
        "ConstListDim": common.ConstListDim,
    }
    exec(textwrap.dedent(source), namespace)
    return namespace


class TestDeclaration:
    def test_owner_and_dimensions(self):
        assert V2E.Local.owner is V2E
        assert V2E.origin is Vertex
        assert V2E.codomain is Edge
        assert V2E.Local.kind is DimensionKind.LOCAL
        assert issubclass(V2E.Local, DimensionIndex)

    def test_counts(self):
        assert (V2E.Local.max_neighbors, V2E.Local.min_neighbors) == (4, 3)
        assert (E2V.Local.max_neighbors, E2V.Local.min_neighbors) == (None, None)

    def test_ownerless_local(self):
        assert LsqCoeff.owner is None
        assert (LsqCoeff.max_neighbors, LsqCoeff.min_neighbors) == (3, 3)
        assert LsqCoeff.kind is DimensionKind.LOCAL

    def test_counts_from_local_size(self):
        ns = _declare(
            """
            class C2E(NeighborConnectivity[Vertex, Edge]):
                class Local(LocalDimensionIndex, size=3): ...
            """
        )
        assert (ns["C2E"].Local.max_neighbors, ns["C2E"].Local.min_neighbors) == (3, 3)

    def test_identity(self):
        assert V2E.tag == f"{__name__}.V2E"
        assert V2E.Local.tag == f"{__name__}.V2E.Local"
        assert common.resolve(V2E.Local.tag) is V2E.Local
        assert str(V2E) == "V2E"
        assert repr(V2E) == V2E.tag

    def test_pickle_by_reference(self):
        assert pickle.loads(pickle.dumps(V2E)) is V2E
        assert pickle.loads(pickle.dumps(V2E.Local)) is V2E.Local

    def test_hashable(self):
        assert {V2E: 1}[V2E] == 1

    def test_type_parameter_subscription(self):
        alias = NeighborConnectivity[Vertex, Edge]
        assert typing.get_origin(alias) is NeighborConnectivity
        assert typing.get_args(alias) == (Vertex, Edge)

    def test_bool_is_not_a_neighbor_index(self):
        with pytest.raises(TypeError):
            V2E[True]

    def test_not_instantiable(self):
        with pytest.raises(TypeError, match="cannot be instantiated"):
            V2E()


class TestDeclarationErrors:
    @pytest.mark.parametrize(
        "source, match",
        [
            (
                """
                class C(NeighborConnectivity[Vertex, Edge]): ...
                """,
                "must declare its local dimension",
            ),
            (
                """
                class C(NeighborConnectivity[Vertex, Edge]):
                    class Local(DimensionIndex): ...
                """,
                "must declare its local dimension",
            ),
            (
                """
                class C(NeighborConnectivity[Vertex, Edge], max_neighbors=5):
                    Local: typing.TypeAlias = V2E.Local
                """,
                "counts are declared by its owner",
            ),
            (
                """
                class C(NeighborConnectivity):
                    class Local(LocalDimensionIndex): ...
                """,
                "must derive from 'NeighborConnectivity\\[Origin, Codomain\\]'",
            ),
            (
                """
                class C(V2E):
                    class Local(LocalDimensionIndex): ...
                """,
                "must derive from 'NeighborConnectivity\\[Origin, Codomain\\]'",
            ),
            (
                """
                class C(NeighborConnectivity[V2E.Local, Edge]):
                    class Local(LocalDimensionIndex): ...
                """,
                "'Origin' must be a non-local dimension",
            ),
            (
                """
                class C(NeighborConnectivity[Vertex, int]):
                    class Local(LocalDimensionIndex): ...
                """,
                "'Codomain' must be a non-local dimension",
            ),
            (
                """
                class C(NeighborConnectivity[Vertex, Edge], max_neighbors=2, min_neighbors=3):
                    class Local(LocalDimensionIndex): ...
                """,
                "exceeds 'max_neighbors'",
            ),
            (
                """
                class C(NeighborConnectivity[Vertex, Edge], max_neighbors=4):
                    class Local(LocalDimensionIndex, size=3): ...
                """,
                "contradicts the size",
            ),
            (
                """
                class L(LocalDimensionIndex, kind=DimensionKind.HORIZONTAL): ...
                """,
                "cannot have kind",
            ),
            (
                """
                class L(LocalDimensionIndex, size=1.5): ...
                """,
                "must be an integer",
            ),
            (
                """
                class L(DimensionIndex, kind=DimensionKind.LOCAL): ...
                """,
                "subclassing 'LocalDimensionIndex'",
            ),
        ],
    )
    def test_rejected(self, source, match):
        with pytest.raises(TypeError, match=match):
            _declare(source)

    def test_negative_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            _declare(
                """
                class C(NeighborConnectivity[Vertex, Edge], max_neighbors=-1):
                    class Local(LocalDimensionIndex): ...
                """
            )

    def test_adopting_an_ownerless_local(self):
        ns = _declare(
            """
            class Coeff(LocalDimensionIndex, size=3): ...

            class C(NeighborConnectivity[Vertex, Edge]):
                Local: typing.TypeAlias = Coeff
            """
        )
        assert ns["Coeff"].owner is ns["C"]
        assert (ns["Coeff"].max_neighbors, ns["Coeff"].min_neighbors) == (3, 3)

    def test_sharing_a_local_dimension(self):
        ns = _declare(
            """
            class V2EShared(NeighborConnectivity[Vertex, Edge]):
                Local: typing.TypeAlias = V2E.Local
            """
        )
        shared = ns["V2EShared"]
        assert shared.Local is V2E.Local
        assert V2E.Local.owner is V2E
        assert V2E.offset_tag == V2E.Local.tag
        assert shared.offset_tag == shared.tag
        assert shared.__gt_type__().tag == shared.tag

    def test_non_integer_index(self):
        with pytest.raises(TypeError, match="indexed by an integer"):
            V2E[Vertex]

    def test_base_is_not_a_declaration(self):
        with pytest.raises(TypeError, match="not a connectivity declaration"):
            NeighborConnectivity.__gt_type__()

    def test_function_local_declaration(self):
        with pytest.raises(TypeError, match="module level"):

            class C(NeighborConnectivity[Vertex, Edge]):
                Local: typing.TypeAlias = LsqCoeff

    def test_subclass_of_owned_local_is_ownerless(self):
        ns = _declare(
            """
            class Other(V2E.Local): ...
            """
        )
        assert ns["Other"].owner is None

    def test_local_dimension_cannot_be_staggered(self):
        with pytest.raises(TypeError, match="cannot be staggered"):
            common.Staggered[V2E.Local]


def _table_type(
    domain=(Vertex, V2E.Local),
    codomain=Edge,
    max_neighbors=4,
    skip_value=common._DEFAULT_SKIP_VALUE,
    dtype=np.int32,
) -> NeighborConnectivityType:
    return NeighborConnectivityType(
        domain=domain,
        codomain=codomain,
        skip_value=skip_value,
        dtype=core_defs.dtype(dtype),
        max_neighbors=max_neighbors,
    )


class TestCheckNeighborTable:
    def test_matching_type(self):
        common.check_neighbor_table(V2E, _table_type())

    def test_matching_table(self):
        from gt4py.next import constructors

        table = constructors.as_connectivity(
            domain={Edge: 2, E2V.Local: 2}, codomain=Vertex, data=np.array([[0, 1], [1, 2]])
        )
        common.check_neighbor_table(E2V, table)

    def test_undeclared_counts_accept_any_table(self):
        common.check_neighbor_table(
            E2V, _table_type(domain=(Edge, E2V.Local), codomain=Vertex, max_neighbors=7)
        )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"domain": (Vertex, E2V.Local)}, "its domain is"),
            ({"domain": (Edge, V2E.Local)}, "its domain is"),
            ({"codomain": Vertex}, "its codomain is"),
            ({"dtype": np.float64}, "is not integral"),
            ({"max_neighbors": 5}, "expected max_neighbors=4"),
            ({"skip_value": None}, "requires a skip value"),
        ],
    )
    def test_mismatch(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            common.check_neighbor_table(V2E, _table_type(**kwargs))

    def test_min_neighbors_exceeds_table(self):
        ns = _declare(
            """
            class MinOnly(NeighborConnectivity[Vertex, Edge], min_neighbors=5):
                class Local(LocalDimensionIndex): ...
            """
        )
        min_only = ns["MinOnly"]
        for skip_value in (None, common._DEFAULT_SKIP_VALUE):
            with pytest.raises(ValueError, match="min_neighbors=5 exceeds"):
                common.check_neighbor_table(
                    min_only,
                    _table_type(
                        domain=(Vertex, min_only.Local), max_neighbors=3, skip_value=skip_value
                    ),
                )

    def test_bool_table_is_not_integral(self):
        with pytest.raises(ValueError, match="is not integral"):
            common.check_neighbor_table(V2E, _table_type(dtype=bool))

    def test_not_a_neighbor_table(self):
        with pytest.raises(ValueError, match="expected a neighbor table"):
            common.check_neighbor_table(V2E, common.CartesianConnectivity(Vertex, 1))

    def test_skip_value_without_missing_neighbors(self):
        ns = _declare(
            """
            class Full(NeighborConnectivity[Vertex, Edge], max_neighbors=2, min_neighbors=2):
                class Local(LocalDimensionIndex): ...
            """
        )
        full = ns["Full"]
        with pytest.raises(ValueError, match="has skip value"):
            common.check_neighbor_table(
                full, _table_type(domain=(Vertex, full.Local), max_neighbors=2)
            )


class TestFrontendIntegration:
    def test_from_value_is_an_offset(self):
        # NOTE: pins the `__gt_type__` branch of `from_value` ahead of the dimension branch; a
        # connectivity declaration is a class, like a dimension.
        assert type_translation.from_value(V2E) == ts.OffsetType(
            source=Edge, target=(Vertex, V2E.Local), tag=V2E.Local.tag
        )

    def test_field_offset_is_derived_once(self):
        assert V2E.__gt_field_offset__() is V2E.__gt_field_offset__()
        assert V2E.__gt_field_offset__().value == V2E.Local.tag

    def test_neighbor_index_accepts_numpy_integers(self):
        from gt4py.next import constructors, embedded

        table = constructors.as_connectivity(
            domain={Vertex: 2, V2E.Local: 4},
            codomain=Edge,
            data=np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
        )
        with embedded.context.update(offset_provider={V2E.Local.tag: table}):
            assert np.array_equal(V2E[np.int32(1)].asnumpy(), V2E[1].asnumpy())

    def test_legacy_field_offset_has_local(self):
        from gt4py.next import FieldOffset

        assert FieldOffset("V2E", source=Edge, target=(Vertex, V2E.Local)).Local is V2E.Local

    def test_attribute_errors_are_dsl_errors(self):
        from gt4py.next import errors, field_operator
        from gt4py.next.ffront.func_to_foast import FieldOperatorParser
        from gt4py.next import Dims, Field

        def origin_of(a: Field[Dims[Edge], float]) -> Field[Dims[Vertex], float]:
            return a(V2E.origin)

        with pytest.raises(errors.DSLError, match="has no attribute 'origin'"):
            FieldOperatorParser.apply_to_function(origin_of)

    def test_fingerprint_covers_the_declaration(self):
        from gt4py.next import fingerprinting

        def fingerprint_of(source: str) -> str:
            # lenient: `_declare` classes are not importable, as in a re-run notebook cell
            return fingerprinting.lenient_fingerprinter(_declare(source)["C"])

        base = fingerprint_of(
            """
            class C(NeighborConnectivity[Vertex, Edge]):
                class Local(LocalDimensionIndex): ...
            """
        )
        swapped = fingerprint_of(
            """
            class C(NeighborConnectivity[Edge, Vertex]):
                class Local(LocalDimensionIndex): ...
            """
        )
        counted = fingerprint_of(
            """
            class C(NeighborConnectivity[Vertex, Edge], max_neighbors=3):
                class Local(LocalDimensionIndex): ...
            """
        )
        assert len({base, swapped, counted}) == 3
        assert fingerprinting.strict_fingerprinter(V2E) != fingerprinting.strict_fingerprinter(E2V)

    def test_grid_type_deduction(self):
        assert (
            transform_utils._deduce_grid_type(None, [Vertex, V2E]) is common.GridType.UNSTRUCTURED
        )
        with pytest.raises(ValueError, match="CARTESIAN"):
            transform_utils._deduce_grid_type(common.GridType.CARTESIAN, [V2E])


def test_redefined_declaration_with_an_adopted_local(monkeypatch):
    """Re-running a cell must re-own the adopted local dimension, not become a sharer."""
    import sys
    import types as pytypes

    module = pytypes.ModuleType("_readopted_connectivity_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = textwrap.dedent(
        """
        import typing

        from gt4py.next.common import DimensionIndex, LocalDimensionIndex, NeighborConnectivity

        class V(DimensionIndex): ...
        class E(DimensionIndex): ...
        class V2EDim(LocalDimensionIndex, size={n}): ...
        class V2E(NeighborConnectivity[V, E], max_neighbors={n}):
            Local: typing.TypeAlias = V2EDim
        """
    )
    exec(source.format(n=4), module.__dict__)
    assert module.V2E.offset_tag == module.V2EDim.tag

    # the redefinition takes ownership over again, and its counts are checked against `size=`
    exec(source.format(n=2), module.__dict__)
    assert module.V2EDim.owner is module.V2E
    assert module.V2E.offset_tag == module.V2EDim.tag
    assert module.V2EDim.max_neighbors == 2


def test_local_dimension_of():
    shared = _declare(
        """
        class V2EShared(NeighborConnectivity[Vertex, Edge]):
            Local: typing.TypeAlias = V2E.Local
        """
    )["V2EShared"]
    assert common.local_dimension_of(V2E) is V2E.Local
    assert common.local_dimension_of(shared) is V2E.Local
    with pytest.raises(TypeError, match="not a connectivity declaration"):
        common.local_dimension_of(NeighborConnectivity)


class TestFieldOffsetDeprecation:
    def test_unstructured_field_offset_warns(self):
        from gt4py.next import FieldOffset

        with pytest.warns(DeprecationWarning, match="NeighborConnectivity"):
            FieldOffset(V2E.Local.tag, source=Edge, target=(Vertex, V2E.Local))

    def test_derived_and_cartesian_field_offsets_do_not_warn(self, recwarn):
        from gt4py.next import FieldOffset

        class_ns = _declare(
            """
            class C2E(NeighborConnectivity[Vertex, Edge]):
                class Local(LocalDimensionIndex): ...
            """
        )
        class_ns["C2E"].__gt_field_offset__()
        FieldOffset("Koff", source=KDim, target=(KDim,))
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
