import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
import copy
from gt4py.gtscript import Field

# config stuff
# backend = "gt:cpu_ifirst"
backend = "numpy"
dtype = np.float32
if backend == "numpy":
    backend_opts = {
        "rebuild": True,
    }
else:
    backend_opts = {
        "rebuild": False,
        "verbose": True,
        "_validate_args": True
    }


Nx, Ny = 2, 2
Nz = 2

n = 4
m = 6
n_dtype = (dtype, (n,))
m_dtype = (dtype, (m,))
dtype_matrix = (dtype, (n,m))

n_field_np = np.random.rand(n)
m_field_np = np.random.rand(m)
matrix_np = np.random.random(n*m).reshape(n, m)

matrix = gt.storage.from_array(
    data= matrix_np,
    backend = backend,
    dtype=dtype_matrix,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))

n_field = gt.storage.from_array(
    data= n_field_np,
    backend = backend,
    dtype=n_dtype,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))

m_field = gt.storage.from_array(
    data= m_field_np,
    backend = backend,
    dtype=m_dtype,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))


out_field = gt.storage.zeros(
    backend=backend,
    dtype=n_dtype,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))

coeff = 2.0


@gtscript.stencil(backend=backend, **backend_opts)
def test_stencil(
        matrix: gtscript.Field[dtype_matrix],
        # c: float,
        vec_n: gtscript.Field[n_dtype],
        vec_m: gtscript.Field[m_dtype]
        # out: gtscript.Field[n_dtype]
        # vec_m: gtscript.Field[m_dtype]
):
    # BUG: when using gtscript.Field instead of import Field 
    # tmp: gtscript.Field[(np.float64, (2,))] =  0
    tmp: Field[(np.float64, (4,))] =  2
    with computation(PARALLEL), interval(...):
        vec_n = 2 * tmp
        

test_stencil(matrix, n_field, m_field)
# print(f'{coeff = }\n{n_field = }\n{m_field = }\n{matrix = }\n')
print(f'{n_field = }\n')
tmp = np.einsum('ijklm, ijkm -> ijkl', matrix, m_field)
# np.testing.assert_allclose(np.asarray(n_field), tmp, rtol=1e-5, atol=1e-8)
# print(f'{m_field =}')
# print(f'{tmp = }')


