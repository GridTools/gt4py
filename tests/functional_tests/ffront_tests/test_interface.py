"""
Basic Interface Tests
=====================

    - declare a connectivity
    - create and run a stencil
        - field args declaration
        - scalar args
    - create and run a fencil
        - pass fields
        - pass connectivities (at run time, later at compile time too)
        - out field
        - think about ways to pass backend/connectivities etc 
            (in function signature / in functor config method / with block)
    - built-in field operators
        - arithmetics
        - shift
        - neighbor reductions
        - math functions: abs(), max(), min, mod(), sin(), cos(), tan(), arcsin(), arccos(), arctan(),
            sqrt(), exp(), log(), isfinite(), isinf(), isnan(), floor(), ceil(), trunc()
    - evaluation test cases


"""
import pytest


pytestmark = pytest.mark.skip(reason="incomplete")


def test_copy_lower():
    Vertex = Dimension("Vertex")

    @field_operator
    def copy_field(inp: Field[Vertex]):
        return inp

    ## check the source to source
    assert inspect.getsource(copy_field.__call__) == canonicalize(
        """
        def copy_field(...):
            return lift(deref(inp))
    """
    )

    ## lowering
    assert isinstance(generateIR(copy_field.__call__), FunctionDefinition)


def test_field_declaration(vertex_field, vertex_field_v_e):
    Vertex = Dimension("Vertex")
    V_E = Dimension("Edge Neighbors of Vertices")

    @field_operator
    def ddd(something: Field[Vertex, V_E]):
        return something(V_E[0])  # V_E[0] -> Offset

    assert ddd.api_fields["something"].type == "Field"
    assert ddd(vertex_field, offset_providers={"V_E": vertex_field_v_e})
