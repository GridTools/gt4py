import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
import copy

# config stuff
backend = "gt:cpu_ifirst"
# backend = "numpy"
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

n_field = np.random.rand(n)
matrix_np = np.random.random(n*m).reshape(n, m)

matrix = gt.storage.from_array(
    data= matrix_np,
    backend = backend,
    dtype=dtype_matrix,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))

in_field_1 = gt.storage.from_array(
    data= n_field,
    backend = backend,
    dtype=n_dtype,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))

in_field_2 = 3 * gt.storage.ones(
    backend=backend,
    dtype=m_dtype,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))


out_field = gt.storage.zeros(
    backend=backend,
    dtype=n_dtype,
    shape=(Nx, Ny, Nz), # todo: maybe try 2D
    default_origin=(0, 0, 0))

coeff = 2.0

@gtscript.function
def mult_coeff(vec_1, vec_2):
    tmp: gtscript.Field[(np.float64, (2,))] =  0
    return tmp


@gtscript.stencil(backend=backend, **backend_opts)
def test_stencil(
        matrix: gtscript.Field[dtype_matrix],
        # c: float,
        vec_n: gtscript.Field[m_dtype],
        out: gtscript.Field[n_dtype],
        # vec_m: gtscript.Field[m_dtype]
):
    # tmp: gtscript.Field[(np.float64, (2,))] =  0
    with computation(PARALLEL), interval(...):
        out = matrix @ vec_n
        

test_stencil(matrix, in_field_2, out_field)
print(f'{coeff = }\n{in_field_1 = }\n{in_field_2 = }\n{matrix = }\n{out_field = }')
tmp = np.einsum('ijklm, ijkm -> ijkl', matrix, in_field_2)
np.testing.assert_allclose(np.asarray(out_field), tmp, rtol=1e-5, atol=1e-8)
print(f'{tmp = }')


