# Architecture

Implements the iterator view as described [here](https://github.com/GridTools/concepts/wiki/Iterator-View).

## Iterator view program in Python

A program for the iterator view consists of Python functions decorated with `@fundef` and an entry point, *fencil*, which is a Python function decorated with `@fendef`. The *fencil* must only contain calls to the `closure(...)` function.

Legal functions much not have side-effects, however, e.g., for debugging purposes, side-effects can be used in embedded execution.

There are 2 modes of execution: *embedded* (direct execution in Python) and *tracing* (trace function calls -> Eve IR representation -> code generation).
The implementations of *embedded* and *tracing* are decoupled by registering themselves (dependency inversion) in `builtins.py` (contains dispatch functions for all builtins of the model) and `runtime.py` (contains dispatch mechanism for the `fendef` and `fundef` decorators and the `closure(...)` function).

The builtins dispatcher is implemented in `dispatcher.py`. Implementations are registered with a key (`str`) (currently `tracing` and `embedded`). The active implementation is selected by pushing a key to the dispatcher stack.

`fundef` returns a wrapper around the function, which dispatches `__call__` to a hook if a predicate is met (used for *tracing*). By default the original function is called (used in *embedded* mode).

`fendef` return a wrapper that dispatches to a registered function. If `backend` is in the keyword arguments and not `None` `fendef_codegen` will be called, otherwise `fundef_embedded` will be called.

## Embedded execution

Embedded execution is implemented in the file `embedded.py`.

Sketch:
- fields are np arrays with named axes; names are instances of `CartesianAxis`
- in `closure()`, the stencil is executed for each point in the domain, the fields are wrapped in an iterator pointing to the current point of execution.
- `shift()` is lazy (to allow lift implementation), offsets are accumulated in the iterator and only executed when `deref()` is called.
- as described in the design, offsets are abstract; on fencil execution the `offset_provider` keyword argument needs to be specified, which is a dict of `str` to either `CartesianAxis` or `NeighborTableOffsetProvider`
- if `column_axis` keyword argument is specified on fencil execution (or in the fencil decorator), all operations will be done column wise in the give axis; `column_axis` needs to be specified if `scan` is used

## Tracing

An iterator view program is traced (implemented in `tracing.py`) and represented in a tree structure defined by the nodes in (`ir.py`).

Sketch:
- Each builtin returns a `FunctionCall` node representing the builtin.
- Foreach `fundef`, the signature of the wrapped function is extracted, then it is invoked with `Sym` nodes as arguments.
- Expressions involving an `Expr` node (e.g. `Sym`) are converted to appropriate builtin calls, e.g. `4. + Sym(id='foo')` is converted to `FunCall(fun=SymRef(id='plus'), args=...)`
- In appropriate places values are converted to nodes, see `make_node()`.
- Finally the IR tree will be passed to `execute_program()` in `backend_executor.py` which will generator code for the program (and execute, if appropriate).

## Backends

See directory `backends/`.

### `cpptoy`

Generates C++ code in the spirit of https://github.com/GridTools/gridtools/pull/1643. Incomplete, will be adapted to the full C++ prototype. (only code generation)

### `lisp`

Incomplete. Example for the grammar used in the model design document. (not executable)

### `roundtrip`

Generates from the IR an aquivalent Python iterator view program which is then executed in embedded mode.

### `double_roundtrip`

Generates the Python iterator view program, traces it again, generates again and executes. Ensures that the generated Python code can still be traced. While the original program might be different from the generated program (e.g. `+` will be converted to `plus()` builtin). The programs from the embedded and double roundtrip backends should be identical.

## Adding a new builtin

Currently there are 4 places where a new builtin needs to be added
- `builtin.py`: for dispatching to an actual implementation
- `embedded.py` and `tracing.py`: for the respective implementation
- `ir.py`: we check for consistent use of symbols, therefore if a `FunCall` to the new builtin is used, it needs to be available in the symbol table.
