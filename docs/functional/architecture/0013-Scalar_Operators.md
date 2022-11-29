# Scalar operators

How about "elementwise operators"?

## What are scalar operators?

Scalar operators are functions that take scalar arguments and return a scalar result. They can be used to implement arbitrary pointwise operations on fields.

As an example, let's calculate the pointwise arithmetic-geometric mean of two fields:

```python
@scalar_operator
def agm(x: float, y: float, tol: float) -> float:
    an = x
    gn = y
    while abs(x - y) > tol:
        an, gn = 0.5*(an + gn), math.sqrt(an * gn)
    return an

@field_operator
def use_agm(x: Field[[D], float], y: Field[[D], float]) -> Field[[D], float]:
    return agm(x, y)
```

In the code above, the function `agm` is executed once for every element pair of the `(x, y)` field pair, and a field of the same size as both `x` and `y` is produced. The function `agm` is essentially just a complicated pointwise binary operator, like `+` or `-`.


## Why do we want scalar operators?

Gt4py is concerned with the efficient parallelization of code, which is much easier to do when the source code can be turned into a simple data flow graph. To ensure this, the syntax of field operators is restricted and several Python constructs, such as loops, cannot be used.

However, gt4py is not concerned with the optimziation of a single thread in the parallel grid, as that is delegated to platform compilers and has a smaller interference with parallel optimizations. Consequently, gt4py could allow fewer restrictions on the syntax in a scalar operator that describes only a single thread of execution. Unlike field operators, scalar operators can easily support conditionals, loops and mutable variables.

Scalar operators can be used to implement features that, using the pure field operator syntax, would be difficult, impractical, or downright impossible. In weather and climate the general targets for scalar operators would be iterative algorithms and solvers. 

## Implementation options

### Within or outside the toolchain

Due to their execution model, scalar operators can be implemented both within and outside the iterator IR compilation process.

#### Within the toolchain

When implemented within the toolchain, scalar operators are parsed just like field operators and scan operators, then get lowered to iterator IR, and finally C++ code is emitted. Since iterator IR is a functional language, the lowering must convert statement-based conditionals and loops into expression-based equivalents to create valid iterator IR.

Statement-based:
```python
e = 0
f = 1
for n in range(1, 10):
    e = e + 1/f
    f = f * n
```

Expression-based:
```python
e, f = do_for(
    rn=(1, 10),
    init=(0, 1),
    body=lambda n, e, f: (e + 1/f, f * n)
)
```

#### Outside the toolchain

Since the code generated from scalar operators won't use any GridTools features, the Python to C++ transpilation can happen outside the iterator IR toolchain. The transpiled C++ also does not need to be in a functional form, although that's not the only challenge when bridging the Python syntax to C++. 

When bypassing the toolchain, any internal or third party tool can be used to translate Python to C or C++. The C++ code, which will be a single function, is simply inserted into the generated GTFN C++ and is called as a regular C++ function.

### Options to bypass the toolchain

#### Requirements

First, the Python-to-C++ transpiler must support both CPU and CUDA compilation.

Second, the transpiler must support non-trivial Python code, and it must either emit an error or produce correct code. The main difficulty here is matching the scoping rules of Python and C++. Take the following piece of Python code:

```python
if condition:
    value = 5
else:
    value = 6.2
use(value)
```

At the point of `use`ing `value`, it may be `int(5)` `float(6.2)`, or in other cases even `None`. The proper translation of this to C++ may be:

```c++
std::variant<std::monostate, long, double> value;
if (condition) {
    value = 5;
} else {
    value = 6.2;
}
use(value);
```

Since this would probably be too slow for high performance computing, we can settle for:
```c++
std::common_type_t<long, double> value;
if (condition) {
    value = 5;
} else {
    value = 6.2;
}
use(value);
```

Another alternative is to mandate the declaration of variables which makes the transpilation substantially simpler. However, this still requires compiler diagnostics to tell the user what to declare:
```python
value: float
if condition:
    value = 5
else:
    value = 6.2
use(value)
```

#### Option 1: DaCe

While DaCe is a data-parallel optimization framework, it also has builtin capabalities to translate Python to C++ code, furthermore, it also supports CUDA. Unfortunately, DaCe does not produce correct C++ code for the scoping cases, neither does it emit its own diagnostics. The generated C++ code fails to compile, however, so it won't just produce invalid results. DaCe works as expected when explicit variable declarations are used, but work needs to be done to get diagnostics.

#### Option 2: Pythran

Pythran, being dedicated to Python to C++ compilation, handled the cases we've tested just fine. If it wasn't for its lack of support for CUDA, it would be an easy-to-use solution. Unfortunately, modifying Pythran to suit our requirements is not necessarily trivial, just like modifying DaCe.

#### Option 3: Roll our own transpiler

Using Python's AST module and libraries like `cgen`, all we have to implement is an AST to AST translation. Although this part is simple, handling the scoping rules would a hundred percent be our responsibility. Just like DaCe and Pythran, this method is also not trivial.

## Conclusion

The current prototype implementation uses Pythran which allows it to be very simple, but makes it unusable with the CUDA backend. As a result, the prototype cannot be extended to work in production, but it can still be used to write some code with scalar operators and evaluate the scalar operator concept as a whole.


