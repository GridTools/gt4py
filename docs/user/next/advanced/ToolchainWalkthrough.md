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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta
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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

style foasta fill:red
style fclos fill:red
linkStyle 2 stroke:red,stroke-width:4px,color:pink
```

Here we have to manually combine the previous result with the call arguments. When we call the toolchain as a whole later we will only have to do this once at the beginning.

```python
fclos = backend.DEFAULT_FIELDOP_TRANSFORMS.foast_to_foast_closure(
    gtx.otf.workflow.InputWithArgs(
        data=foast,
        args=(gtx.ones(domain={I: 10}, dtype=gtx.float64),),
        kwargs={
            "out": gtx.zeros(domain={I: 10}, dtype=gtx.float64),
            "from_fieldop": example_fo
        },
    )
)
```

```python
fclos.closure_vars["example_fo"].backend
```

```python
gtx.ffront.stages.FoastClosure??
```

    [0;31mInit signature:[0m
    [0mgtx[0m[0;34m.[0m[0mffront[0m[0;34m.[0m[0mstages[0m[0;34m.[0m[0mFoastClosure[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mfoast_op_def[0m[0;34m:[0m [0;34m'FoastOperatorDefinition[OperatorNodeT]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0margs[0m[0;34m:[0m [0;34m'tuple[Any, ...]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkwargs[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mclosure_vars[0m[0;34m:[0m [0;34m'dict[str, Any]'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mDocstring:[0m      FoastClosure(foast_op_def: 'FoastOperatorDefinition[OperatorNodeT]', args: 'tuple[Any, ...]', kwargs: 'dict[str, Any]', closure_vars: 'dict[str, Any]')
    [0;31mSource:[0m
    [0;34m@[0m[0mdataclasses[0m[0;34m.[0m[0mdataclass[0m[0;34m([0m[0mfrozen[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;32mclass[0m [0mFoastClosure[0m[0;34m([0m[0mGeneric[0m[0;34m[[0m[0mOperatorNodeT[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0mfoast_op_def[0m[0;34m:[0m [0mFoastOperatorDefinition[0m[0;34m[[0m[0mOperatorNodeT[0m[0;34m][0m[0;34m[0m
    [0;34m[0m    [0margs[0m[0;34m:[0m [0mtuple[0m[0;34m[[0m[0mAny[0m[0;34m,[0m [0;34m...[0m[0;34m][0m[0;34m[0m
    [0;34m[0m    [0mkwargs[0m[0;34m:[0m [0mdict[0m[0;34m[[0m[0mstr[0m[0;34m,[0m [0mAny[0m[0;34m][0m[0;34m[0m
    [0;34m[0m    [0mclosure_vars[0m[0;34m:[0m [0mdict[0m[0;34m[[0m[0mstr[0m[0;34m,[0m [0mAny[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m           ~/Code/gt4py/src/gt4py/next/ffront/stages.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m

## FOAST with args -> PAST closure

This auto-generates a program for us, directly in PAST representation and forwards the call arguments to it

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

style pclos fill:red
%%style pclos fill:red
linkStyle 4 stroke:red,stroke-width:4px,color:pink
```

```python
pclost = backend.DEFAULT_PROG_TRANSFORMS.past_transform_args(pclos)
```

```python
pclost.kwargs
```

    {}

## Lower PAST -> ITIR

still forwarding the call arguments

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

style fdefa fill:red
style fuwr fill:red
style fdef fill:red
style fargs fill:red
style foast fill:red
style fiwr fill:red
style foasta fill:red
style fclos fill:red
style pclos fill:red
style pcall fill:red
linkStyle 0,2,3,4,5,9,10,11,12,13,14 stroke:red,stroke-width:4px,color:pink
```

### Starting from DSL

```python
pitir2 = backend.DEFAULT_FIELDOP_TRANSFORMS(
    gtx.otf.workflow.InputWithArgs(data=start, args=fclos.args, kwargs=fclos.kwargs | {"from_fieldop": example_fo})
)
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

We can re-run with the output from the previous run as in- and output.

```python
example_compiled(pitir2.args[1], *pitir2.args[1:], offset_provider=OFFSET_PROVIDER)
```

```python
pitir2.args[2]
```

    10

```python
pitir.args
```

    (NumPyArrayField(_domain=Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(0, 10),)), _ndarray=array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),
     NumPyArrayField(_domain=Domain(dims=(Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(0, 10),)), _ndarray=array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])),
     10,
     10)

### Starting from FOAST

Note that it is the exact same call but with a different input stage

```python
pitir3 = backend.DEFAULT_FIELDOP_TRANSFORMS(
    gtx.otf.workflow.InputWithArgs(
        data=foast,
        args=fclos.args,
        kwargs=fclos.kwargs | {"from_fieldop": example_fo}
    )
)
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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

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
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

style pasta fill:red
style pclos fill:red
linkStyle 8 stroke:red,stroke-width:4px,color:pink
```

```python
pclos = backend.DEFAULT_PROG_TRANSFORMS(
    gtx.otf.workflow.InputWithArgs(
        data=p_past,
        args=fclos.args,
        kwargs=fclos.kwargs
    )
)
```

## Full Program Toolchain

```mermaid
graph LR

fdef(FieldOperatorDefinition) -->|func_to_foast| foast(FoastOperatorDefinition)
foast -->|foast_to_itir| itir_expr(itir.Expr)
foasta -->|foast_to_foast_closure| fclos(FoastClosure)
fclos -->|foast_to_past_closure| pclos(PastClosure)
pclos -->|past_process_args| pclos
pclos -->|past_to_itir| pcall(ProgramCall)

pdef(ProgramDefinition) -->|func_to_past| past(PastProgramDefinition)
past -->|past_lint| past
pasta -->|past_to_past_closure| pclos(ProgramClosure)

fdefa(InputWithArgs) --> fuwr{{"internal unwrapping"}} --> fdef
fuwr --> fargs(args, kwargs)

foast --> fiwr{{"internal wrapping"}} --> foasta(InputWithArgs)
fargs --> foasta

pdefa(InputWithArgs) --> puwr{{"internal unwrapping"}} --> pdef
puwr --> pargs(args, kwargs)

past --> piwr{{"internal wrapping"}} --> pasta(InputWithArgs)
pargs --> pasta

style pdefa fill:red
style puwr fill:red
style pdef fill:red
style pargs fill:red
style past fill:red
style piwr fill:red
style pasta fill:red
style pclos fill:red
style pcall fill:red
linkStyle 4,5,6,7,8,15,16,17,18,19,20 stroke:red,stroke-width:4px,color:pink
```

### Starting from DSL

```python
p_itir1 = backend.DEFAULT_PROG_TRANSFORMS(
    gtx.otf.workflow.InputWithArgs(
        data=p_start,
        args=fclos.args,
        kwargs=fclos.kwargs
    )
)
```

```python
p_itir2 = backend.DEFAULT_PROG_TRANSFORMS(
    gtx.otf.workflow.InputWithArgs(
        data=p_past,
        args=fclos.args,
        kwargs=fclos.kwargs
    )
)
```

```python
assert p_itir1 == p_itir2
```
