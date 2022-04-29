import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript

# config stuff
backend = "numpy"
dtype = np.float64
backend_opts = {
    "rebuild": True,
    "_validate_args": False
}

Nx, Ny = 1, 1

# # intialize storages
# inp_vec = gt.storage.ones(
#     backend=backend,
#     dtype=(dtype, (2,)),
#     shape=(Nx, Ny, 1), # todo: maybe try 2D
#     default_origin=(0, 0, 0))

# out_vec = gt.storage.zeros(
#     backend=backend,
#     dtype=(dtype, (2,)),
#     shape=(Nx, Ny, 1), # todo: maybe try 2D
#     default_origin=(0, 0, 0))
    
# matrix = gt.storage.zeros(
#     backend=backend,
#     dtype=(dtype, (2, 2)),
#     shape=(Nx, Ny, 1), # todo: maybe try 2D
#     default_origin=(0, 0, 0))


# M = np.array([1, 2, 3, 4]).reshape(Nx, Ny, 1, 2, 2)
# res = np.dot(M.reshape(2, 2), np.ones(2)).reshape(Nx, Ny, 1, 1, 2)

# matrix[:] = M

# @gtscript.function
# def matmul_2(matrix, inp_vec, out_vec):
#     a = matrix[0,0,0][0, 0] * inp_vec[0,0,0][0] + matrix[0,0,0][0, 1] * inp_vec[0,0,0][1]
#     b = matrix[0,0,0][1, 0] * inp_vec[0,0,0][0] + matrix[0,0,0][1, 1] * inp_vec[0,0,0][1]
#     return a, b

# @gtscript.stencil(backend=backend, **backend_opts)
# def mul_by_2(
#     M: gtscript.Field[(dtype, (2, 2))],
#     inp_vec: gtscript.Field[(dtype, (2,))],
#     out_vec: gtscript.Field[(dtype, (2,))]
# ):
#     with computation(PARALLEL), interval(...):
#         a, b = matmul_2(M, inp_vec, out_vec)

# mul_by_2(matrix, inp_vec, out_vec, domain=(Nx, Ny, 1))
# print(f'{M.reshape(2, 2)} * {inp_vec.reshape(2)} = {out_vec.reshape(2)}')

# dtype_field_1 = (dtype, (5,))
# dtype_field_2 = (dtype, (3,))
# dtype_field = dtype

n = 2
<<<<<<< HEAD
m = 3
n_dtype = (dtype, (n,))
m_dtype = (dtype, (m,))
dtype_matrix = (dtype, (n,m))
# dtype_field = dtype

n_field = np.arange(n)


matrix_np = np.arange(n*m).reshape(n, m)


matrix = gt.storage.from_array(
    data= matrix_np,
    backend = backend,
    dtype=dtype_matrix,
    shape=(Nx, Ny, 1), # todo: maybe try 2D
    default_origin=(0, 0, 0))

in_field_1 = gt.storage.from_array(
    data= n_field,
    backend = backend,
    dtype=n_dtype,
=======
dtype_field = (dtype, (n,))

in_np = np.arange(n)

in_field_1 = gt.storage.from_array(
    data= in_np,
    backend = backend,
    dtype=dtype_field,
>>>>>>> 109f883 (intial ops working)
    shape=(Nx, Ny, 1), # todo: maybe try 2D
    default_origin=(0, 0, 0))

in_field_2 = 3 * gt.storage.ones(
    backend=backend,
<<<<<<< HEAD
    dtype=m_dtype,
=======
    dtype=dtype_field,
>>>>>>> 109f883 (intial ops working)
    shape=(Nx, Ny, 1), # todo: maybe try 2D
    default_origin=(0, 0, 0))


out_field = gt.storage.zeros(
    backend=backend,
    dtype=dtype_field,
    shape=(Nx, Ny, 1), # todo: maybe try 2D
    default_origin=(0, 0, 0))

coeff = 2.0

@gtscript.function
def mult_coeff(vec_1, vec_2):
    tmp: gtscript.Field[(np.float64, (2,))] =  0
    return tmp


@gtscript.stencil(backend=backend, **backend_opts)
def test_stencil(
        vec_1: gtscript.Field[dtype_field],
        vec_2: gtscript.Field[dtype_field],
        out_vec: gtscript.Field[dtype_field],
        c: float
):
    # tmp: gtscript.Field[(np.float64, (2,))] =  vec_1
    with computation(PARALLEL), interval(...):
        out_vec = mult_coeff(vec_1, vec_2)
        # out_vec = tmp

# @gtscript.stencil(backend=backend, **backend_opts)
# def test_stencil(
#     vec_1: gtscript.Field[(dtype, (2,))],
#     vec_2: gtscript.Field[(dtype, (2,))],
#     out_vec: gtscript.Field[(dtype, (2,))],
#     matrix: gtscript.Field[(dtype, (2,2))],
#     c: float
# ):
#     tmp: Field[IJK, (np.float_, (2,))] = 
#     with computation(PARALLEL), interval(...):
#         out_vec[0,0,0] = vec_1[0,0,0]
#         out_vec[0,0,0] = c * vec_1[0,0,0]
#         out_vec[0,0,0] = vec_1[0,0,0] * vec_2[0,0,0]
#         out_vec[0,0,0] = vec_1[0,0,0] / vec_2[0,0,0]
#         out_vec[0,0,0] = vec_1[0,0,0] + vec_2[0,0,0]
#         out_vec[0,0,0] = vec_1[0,0,0] - vec_2[0,0,0]

#         out_vec[0,0,0] = c * (vec_1[0,0,0] - vec_2[0,0,0])
#         out_vec[0,0,0] = c * ([vec_1[0,0,0] - vec_2[0,0,0], vec_1[0,0,0] - vec_2[0,0,0]])
#         out_vec[0,0,0] = vec_1[0,0,0] * vec_2[0,0,0] * vec_2[0,0,0]

#         out_vec[0,0,0] = matmul(matrix, vec_1)
#         out_vec[0,0,0] = matmul(matrix, vec_1)[1:-1]

#         out_vec[0,0,0][:-2] = vec_1[1:-1]
#         out_vec[0,0,0][:-2] = vec_1[1:-1] + vec_2[0:-2]




test_stencil(in_field_1, in_field_2, out_field, coeff)
print(f'{coeff = }\n{in_field_1 = }\n{in_field_2 = }\n{out_field = }')
