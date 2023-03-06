# DaCe backend

## Motivation

The DaCe framework uses a graph representation for computer programs. The graphs can natively express parallel workloads in the form of so-called *maps*. A DaCe map consists of a parallel execution domain, similar to a CUDA grid, and a tasklet, similar to a CUDA kernel body. As such, DaCe maps are also very similar to gt4py's stencil closures at the iterator IR level, as well as to gt4py's field view constructs such as adding two fields.

Due to the similarities, it makes sense to translate gt4py constructs to DaCe graphs. This way, we could leverage DaCe's optimization capabilities to produce fast-running stencil binaries.

## Field view vs. iterator view

The two main data-parallel constructs in gt4py are operations on fields in the field view and executing stencils on a domain in iterator view. These constructs translate naturally to DaCe maps, however, they are mutually exclusive at the moment, therefore we have to chose one. In the future, field view and iterator view might be seamlessly used together, which changes the exclusive relationship of the two approaches.

### Translating field view to DaCe maps

In field view, the primary constructs are field operators, which are characterized as a data-flow graph of trivial operations over several fields. For example, adding two fields and then taking the square root of the sum corresponds to a data-flow graph with two nodes, the addition and the square root. The nodes themselves execute a trivial computation, such as adding two numbers for every element of the input fields. The addition node can be translated to a single DaCe map that iterates over the size of the input fields and executes a trivial tasklet with the code `out[i] = a[i] + b[i]`.

Field operators can be more complex than a linear data-flow graph of two nodes, but no matter how large and complex the data-flow graph becomes, the nodes of it will represent simple, fundamental operations such elementwise arithmetic or the application of a connectivity table.  

Equivalence table of field view and DaCe constructs:

| Field view                  | DaCe                                   |
|-----------------------------|----------------------------------------|
| Program                     | Sequence of *states*                   |
| Field operator              | Data-flow graph within a *state*       |
| Operation on entire fields  | Data-parallel *map* over entire fields |
| Operation on single element | *Tasklet* inside a *map*               |

### Translating ITIR to DaCe maps

In iterator view, the primary constructs are stencils and stencil closures. Stencils represent an arbitrarily complex procedure to calculate a single element of an output array. A stencil closure represents the application of a stencil over a certain domain, thereby computing not just a single element of the output array but all elements of it in parallel. The natural mapping to DaCe is to translate a stencil closure to a DaCe map and to translate the stencil to the tasklet executed inside the map.

Equivalence table of field view and DaCe constructs:

| Iterator view    | DaCe                                   |
|------------------|----------------------------------------|
| Program (fencil) | Sequence of *states*                   |
| Stencil closure  | Data-parallel *map* over entire domain |
| Stencil function | *Tasklet* inside a *map*               |

### Comparison of the approaches

#### Optimization aspects

The core representation of our computational programs is a complex data flow graph with simple nodes. While this form is suitable for the generation of program binaries and running the JIT compiled application, doing so is not optimal as this representation tends to produce binaries with excessive memory allocation and access. (Consider a sequential data flow graph of simple elementwise operations. This would allocate a temporary field between each computation, consuming extra memory and moving results between registers and main memory after every trivial computation.)

While not optimal for code generation, this representation is great for optimization because it's easy to analyze as the data flow is explicit and the operations on the data are simple enough to understand its memory access patterns. Consequently, this representation is ideal for both DaCe and gt4py's iterator IR based optimizer.
