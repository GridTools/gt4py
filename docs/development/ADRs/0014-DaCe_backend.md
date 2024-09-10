# DaCe backend

## Motivation

The DaCe framework uses a graph representation for computer programs. The graphs can natively express parallel workloads in the form of so-called _maps_. A DaCe map consists of a parallel execution domain, similar to a CUDA grid, and a tasklet, similar to a CUDA kernel body. As such, DaCe maps are also very similar to gt4py's stencil closures at the iterator IR level, as well as to gt4py's field view constructs such as adding two fields.

Due to the similarities, it makes sense to translate gt4py constructs to DaCe graphs. This way, we can leverage DaCe's optimization capabilities to produce fast-running stencil binaries.

## Field view vs. iterator view

The two main gt4py data-parallel constructs are operations on fields in the field view and executing stencils on a domain in iterator view. These constructs translate naturally to DaCe maps, however, they are mutually exclusive at the moment, therefore we have to chose one. In the future, field view and iterator view might be seamlessly used together, which changes the exclusive relationship of the two approaches.

### Translating field view to DaCe maps

In field view, the primary constructs are field operators, which are characterized as a data-flow graph of trivial operations over several fields. For example, adding two fields and then taking the square root of the sum corresponds to a data-flow graph with two nodes, the addition and the square root. The nodes themselves execute a trivial computation, such as adding two numbers for every element of the input fields. The addition node can be translated to a single DaCe map that iterates over the size of the input fields and executes a trivial tasklet with the code `out[i] = a[i] + b[i]`.

Field operators can be more complex than a linear data-flow graph of two nodes, but no matter how large and complex the data-flow graph becomes, the nodes of it will represent simple, fundamental operations such elementwise arithmetic or the application of a connectivity table.

Equivalence table of field view and DaCe constructs:

| Field view                  | DaCe                                   |
| --------------------------- | -------------------------------------- |
| Program                     | Sequence of _states_                   |
| Field operator              | Data-flow graph within a _state_       |
| Operation on entire fields  | Data-parallel _map_ over entire fields |
| Operation on single element | _Tasklet_ inside a _map_               |

### Translating ITIR to DaCe maps

In iterator view, the primary constructs are stencils and stencil closures. Stencils represent an arbitrarily complex procedure to calculate a single element of an output array. A stencil closure represents the application of a stencil over a certain domain, thereby computing not just a single element of the output array but all elements of it in parallel. The natural mapping to DaCe is to translate a stencil closure to a DaCe map and to translate the stencil body to the tasklet executed inside the map.

Equivalence table of field view and DaCe constructs:

| Iterator view    | DaCe                                   |
| ---------------- | -------------------------------------- |
| Program (fencil) | Sequence of _states_                   |
| Stencil closure  | Data-parallel _map_ over entire domain |
| Stencil body     | _Tasklet_ inside a _map_               |

### Comparison of the approaches

#### Optimization aspects

The core representation of our computational programs is a complex data flow graph with simple nodes. While this form is suitable for the generation of program binaries and running the JIT-compiled application, doing so is not optimal as this representation tends to produce binaries with excessive memory allocation and memory access. (Consider a sequential data flow graph of simple elementwise operations. This would allocate a temporary field between each computation, consuming extra memory and moving results between registers and main memory after every trivial computation.)

While not optimal for code generation, this representation is great for optimization because it's easy to analyze as the data flow is explicit and the operations on the data are simple enough to understand its memory access patterns. Consequently, this representation is ideal for both DaCe and gt4py's iterator IR based optimizer.

When **lowering from field view**, DaCe graphs of such properties will be naturally generated, benefiting the most from DaCe's optimization capabilities. When **using iterator IR**, there is a substantial overlap between DaCe's and gt4py's optimization passes. Although the lowering from field view to iterator view already discards some information that DaCe could use for optimization, the bulk of the workload can still be passed on to DaCe by disabling all iterator IR optimizations and generating temporaries for all `lift` operations. This mode is useful to compare the optimization capabilities of DaCe and iterator IR, and also to assess the effectiveness of DaCe in gt4py's scientific domain. Depending on the interplay of the optimization passes, a different set of iterator IR passes can be enabled to achieve peak performance when using DaCe.

#### Interfacing with gt4py's design

Currently, gt4py contains two levels of intermediate representations:

- field view ASTs: a thin layer that bridges the field view frontend and the iterator IR core
- iterator IR: the core IR that serves as a target for all frontends, as a source for all backends, and contains most of gt4py's optimization passes

Gt4py was designed with multiple frontends and multiple backends in mind. Currently, there is only one frontend, the field view frontend, but there are plans to add an iterator view frontend as well. There are already several backends, including the GTFN backend and the embedded iterator IR backend. All frontends are lowered to iterator IR, and all backends consume iterator. As such, choosing field view as a source for the DaCe backend would mean that the DaCe backend would have to be implemented for every future frontend separate, which is not scalable.

#### Conclusion

While targeting DaCe from the field view ASTs is much more straightforward mapping than targeting DaCe from iterator IR, and it also preserves more information for DaCe to use for optimization, it is problematic from a gt4py design standpoint. As a first attempt, we will implement a DaCe backend that consumes iterator IR, and evaluate DaCe's optimization capabilities in this scenario.

## The ITIR to DaCe translation prototype

After a cycle, we have a working implementation of the iterator IR to DaCe SDFG conversion, as well as the infrastructure to execute gt4py stencils directly via DaCe. The implementation is a prototype, and it's not feature complete.

### Major limitations

#### Tuples

**Problem**: it's not possible and also not practical to generate DaCe tasklets that contain tuple expressions, thus iterator IR tuples are not currently supported.

**Solution**: in ITIR, the sources of tuple values are the `deref`, the `symref`, and the `make_tuple` expressions, each returning a tuple of values. These expressions can be transformed into tasklets that have multiple outputs. The only valid expression on their result is `tuple_get`, which means we can select the result immediately by its index. As long as the tuple is homogeneous, dynamic indexing is valid.

#### Typing

**Problem**: When converting iterator IR stencils to SDFG tasklets, the results of every `deref` expression and subsequent expressions on values have to be known. Currently, these types are **hardcoded** as `float64`.

**Solution**: The iterator type inference pass should provide the concrete type of all value-expressions.

#### Partial support for iterator IR grammar

- First order functions: in DaCe SDFG's, first order functions could be represented by associating a nested SDFG (function equivalent in DaCe) with an access node (variable equivalent in DaCe), which is not a thing as far as I know. There is no practical solution to support this in the SDFG lowering, such ITIR will always be rejected and should not be generated from ITIR passes.
- Lambdas: in DaCe SDFG's, an immediate call to an instantiation of a lambda function can be represented as a nested SDFG. Currently, the DaCe backend does not support this, but the solution should be fairly straightforward. Lambda support is necessary as they are essential for CSE in ITIR.

### Missing features

#### Reductions

Reductions are being reworked for ITIR, potentially also for DaCe, so support is postponed until objectives are clear.

#### Scans

**Problem**: DaCe does not have a builtin construct for scans, thus it will have to be implemented using maps, states and tasklets.

**Solution**: Use a DaCe map for the parallel dimensions of the scan, and implement the scan dimension as a nested SDFG with a sequential loop represented by a state machine.

### Non-feature improvements

#### Temporaries for lifts

**Problem**: the ITIR to DaCe SDFG conversion does not accept ITIR that contains `lifts`. Currently, lifts are eliminated by forced inlining, which sometimes blows up memory accesses.

**Solution**: implement support for conversion of ITIR with temporaries to SDFG. Then, we can enable ITIR heuristics to optimize memory accesses, or we can insert a temporary everywhere to let DaCe do the optimization.

#### Library nodes for indirect indexing

**Problem**: the current method of using tasklets to represent index manipulation by connectivities introduces additional states into the SDFG which hurts the optimization opportunities.

**Solution**: instead of using inter-state edges, using library nodes specifically for index manipulation could keep the entire stencil closure in a single state, improving optimizations. The library nodes have to be implemented and added to DaCe.

#### Compile-time strides

**Problem**: currently, all arrays have fully dynamic strides. This is very bad for the CPU binaries, because dynamic strides prevent vectorization. The GPU backends may also suffer, but much less so.

**Solution**: introduce static strides into the type system and use those types within the DaCe backend as appropriate.
