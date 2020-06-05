## using-dsl: gtscript


CONST_A = 5.0


@lazy_stencil(externals={"CONST_A": CONST_A})
def add_const(tgt: Field[float]):
    from __externals__ import CONST_A, CONST_B

    with computation(PARALLEL), interval(...):
        tgt = CONST_A + CONST_B
