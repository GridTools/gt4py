```python
import dataclasses
import inspect

import gt4py.next as gtx
from gt4py.next import backend

import devtools
```

<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet"><script src="https://spcl.github.io/dace/webclient2/dist/sdfv.js"></script>
<link href="https://spcl.github.io/dace/webclient2/sdfv.css" rel="stylesheet">

```python
I = gtx.Dimension("I")
Ioff = gtx.FieldOffset("Ioff", source=I, target=(I,))
OFFSET_PROVIDER = {"Ioff": I}
```

# Toolchain Overview

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)
```

# Walkthrough from Field Operator

## Starting Out

```python
@gtx.field_operator
def example_fo(a: gtx.Field[[I], gtx.float64]) -> gtx.Field[[I], gtx.float64]:
    return a + 1.0
```

```python
start = example_fo.definition_stage
```

```python
gtx.ffront.stages.FieldOperatorDefinition?
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0mffront[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mFieldOperatorDefinition[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mdefinition[0m[0;34m:[0m [0;34m'types.FunctionType'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgrid_type[0m[0;34m:[0m [0;34m'Optional[common.GridType]'[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mnode_class[0m[0;34m:[0m [0;34m'type[OperatorNodeT]'[0m [0;34m=[0m [0;34m<[0m[0;32mclass[0m [0;34m'gt4py.next.ffront.field_operator_ast.FieldOperator'[0m[0;34m>[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mattributes[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m [0;34m=[0m [0;34m<[0m[0mfactory[0m[0;34m>[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      FieldOperatorDefinition(definition: 'types.FunctionType', grid_type: 'Optional[common.GridType]' = None, node_class: 'type[OperatorNodeT]' = <class 'gt4py.next.ffront.field_operator_ast.FieldOperator'>, attributes: 'dict[str, Any]' = <factory>)
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/ffront/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## DSL -> FOAST

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style fdef fill:red
style foast fill:red
linkStyle 0 stroke:red,stroke-width:4px,color:pink
```

```python
foast = backend.DEFAULT_FIELDOP_TRANSFORMS.func_to_foast(start)
```

```python
gtx.ffront.stages.FoastOperatorDefinition?
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0mffront[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mFoastOperatorDefinition[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mfoast_node[0m[0;34m:[0m [0;34m'OperatorNodeT'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mclosure_vars[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgrid_type[0m[0;34m:[0m [0;34m'Optional[common.GridType]'[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mattributes[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m [0;34m=[0m [0;34m<[0m[0mfactory[0m[0;34m>[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      FoastOperatorDefinition(foast_node: 'OperatorNodeT', closure_vars: 'dict[str, Any]', grid_type: 'Optional[common.GridType]' = None, attributes: 'dict[str, Any]' = <factory>)
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/ffront/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## FOAST -> ITIR

This also happens inside the `decorator.FieldOperator.__gt_itir__` method during the lowering from calling Programs to ITIR

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style foast fill:red
style itir_expr fill:red
linkStyle 1 stroke:red,stroke-width:4px,color:pink
```

```python
fitir = backend.DEFAULT_FIELDOP_TRANSFORMS.foast_to_itir(foast)
```

```python
fitir.__class__
```

    gt4py.next.iterator.ir.FunctionDefinition

## FOAST -> FOAST closure

This is preparation for "directly calling" a field operator.

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style foast fill:red
style fclos fill:red
linkStyle 2 stroke:red,stroke-width:4px,color:pink
```

Here we have to dynamically generate a workflow step, because the arguments were not known before.

```python
fclos = backend.DEFAULT_FIELDOP_TRANSFORMS.foast_inject_args.__class__(
    args=(gtx.ones(domain={I: 10}, dtype=gtx.float64),),
    kwargs={
        "out": gtx.zeros(domain={I: 10}, dtype=gtx.float64)
    },
    from_fieldop=example_fo
)(foast)
```

```python
gtx.ffront.stages.FoastClosure?
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0mffront[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mFoastClosure[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mfoast_op_def[0m[0;34m:[0m [0;34m'FoastOperatorDefinition[OperatorNodeT]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0margs[0m[0;34m:[0m [0;34m'tuple[Any, ...]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkwargs[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mclosure_vars[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      FoastClosure(foast_op_def: 'FoastOperatorDefinition[OperatorNodeT]', args: 'tuple[Any, ...]', kwargs: 'dict[str, Any]', closure_vars: 'dict[str, Any]')
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/ffront/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## FOAST with args -> PAST closure

This auto-generates a program for us, directly in PAST representation and forwards the call arguments to it

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style fclos fill:red
style pclos fill:red
linkStyle 3 stroke:red,stroke-width:4px,color:pink
```

```python
pclos = backend.DEFAULT_FIELDOP_TRANSFORMS.foast_to_past_closure(fclos)
```

```python
gtx.ffront.stages.PastClosure?
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0mffront[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mPastClosure[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mclosure_vars[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpast_node[0m[0;34m:[0m [0;34m'past.Program'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgrid_type[0m[0;34m:[0m [0;34m'Optional[common.GridType]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0margs[0m[0;34m:[0m [0;34m'tuple[Any, ...]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkwargs[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      PastClosure(closure_vars: 'dict[str, Any]', past_node: 'past.Program', grid_type: 'Optional[common.GridType]', args: 'tuple[Any, ...]', kwargs: 'dict[str, Any]')
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/ffront/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## Transform PAST closure arguments

Don't ask me, seems to be necessary though

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style pclos fill:red
%%style pclos fill:red
linkStyle 4 stroke:red,stroke-width:4px,color:pink
```

```python
pclost = backend.DEFAULT_PROG_TRANSFORMS.past_transform_args(pclos)
```

## Lower PAST -> ITIR

still forwarding the call arguments

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style pclos fill:red
style pcall fill:red
linkStyle 5 stroke:red,stroke-width:4px,color:pink
```

```python
pitir = backend.DEFAULT_PROG_TRANSFORMS.past_to_itir(pclost)
```

```python
gtx.otf.stages.ProgramCall?
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0motf[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mProgramCall[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mprogram[0m[0;34m:[0m [0;34m'itir.FencilDefinition'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0margs[0m[0;34m:[0m [0;34m'tuple[Any, ...]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkwargs[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      Iterator IR representaion of a program together with arguments to be passed to it.
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/otf/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## Executing The Result

```python
gtx.program_processors.runners.roundtrip.executor(pitir.program, *pitir.args, offset_provider=OFFSET_PROVIDER, **pitir.kwargs)
```

```python
pitir.args
```

    (NumPyArrayField(_domain=Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(0, 10),)), _ndarray=array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),
     NumPyArrayField(_domain=Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(0, 10),)), _ndarray=array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])),
     10,
     10)

## Full Field Operator Toolchain

using the default step order

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style fdef fill:red
style foast fill:red
style fclos fill:red
style pclos fill:red
style pcall fill:red
linkStyle 0,2,3,4,5 stroke:red,stroke-width:4px,color:pink
```

### Starting from DSL

```python
foast_toolchain = backend.DEFAULT_FIELDOP_TRANSFORMS.replace(
    foast_inject_args=backend.FopArgsInjector(args=fclos.args, kwargs=fclos.kwargs, from_fieldop=example_fo)
)
pitir2 = foast_toolchain(start)
assert pitir2 == pitir
```

#### Pass The result to the compile workflow and execute

```python
example_compiled = gtx.program_processors.runners.roundtrip.executor.otf_workflow(
    dataclasses.replace(pitir2, kwargs=pitir2.kwargs | {"offset_provider": OFFSET_PROVIDER})
)
```

```python
example_compiled(*pitir2.args, offset_provider=OFFSET_PROVIDER)
```

```python
example_compiled(pitir2.args[1], *pitir2.args[1:], offset_provider=OFFSET_PROVIDER)
```

```python
pitir2.args[1].asnumpy()
```

    array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

### Starting from FOAST

Note that it is the exact same call but with a different input stage

```python
pitir3 = foast_toolchain(foast)
assert pitir3 == pitir
```

# Walkthrough starting from Program

## Starting Out

```python
@gtx.program
def example_prog(a: gtx.Field[[I], gtx.float64], out: gtx.Field[[I], gtx.float64]) -> None:
    example_fo(a, out=out)
```

```python
p_start = example_prog.definition_stage
```

```python
gtx.ffront.stages.ProgramDefinition?
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0mffront[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mProgramDefinition[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mdefinition[0m[0;34m:[0m [0;34m'types.FunctionType'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgrid_type[0m[0;34m:[0m [0;34m'Optional[common.GridType]'[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      ProgramDefinition(definition: 'types.FunctionType', grid_type: 'Optional[common.GridType]' = None)
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/ffront/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## DSL -> PAST

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style pdef fill:red
style past fill:red
linkStyle 6 stroke:red,stroke-width:4px,color:pink
```

```python
p_past = backend.DEFAULT_PROG_TRANSFORMS.func_to_past(p_start)
```

## PAST -> Closure

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style past fill:red
style pclos fill:red
linkStyle 7 stroke:red,stroke-width:4px,color:pink
```

```python
pclos = backend.DEFAULT_PROG_TRANSFORMS.replace(
    past_inject_args=backend.ProgArgsInjector(
        args=fclos.args,
        kwargs=fclos.kwargs
    )
)(p_past)
```

## Full Program Toolchain

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foast -->|foast_inject_args| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
past -->|past_inject_args| pclos(ProgramClosure)

style pdef fill:red
style past fill:red
style pclos fill:red
style pcall fill:red
linkStyle 4,5,6,7 stroke:red,stroke-width:4px,color:pink
```

### Starting from DSL

```python
toolchain = backend.DEFAULT_PROG_TRANSFORMS.replace(
    past_inject_args=backend.ProgArgsInjector(
        args=fclos.args,
        kwargs=fclos.kwargs
    )
)
```

```python
p_itir1 = toolchain(p_start)
```

```python
p_itir2 = toolchain(p_past)
```

```python
assert p_itir1 == p_itir2
```
