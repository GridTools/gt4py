```python
import dataclasses
import typing

from gt4py import next as gtx
from gt4py.next.otf import toolchain, workflow
from gt4py.next.ffront import stages as ff_stages
from gt4py import eve
```

<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet"><script src="https://spcl.github.io/dace/webclient2/dist/sdfv.js"></script>
<link href="https://spcl.github.io/dace/webclient2/sdfv.css" rel="stylesheet">

## Replace Steps

```python
cached_lowering_toolchain = gtx.backend.DEFAULT_TRANSFORMS.replace(
    past_to_itir=gtx.ffront.past_to_itir.past_to_gtir_factory(cached=False)
)
```

## Skip Steps / Change Order

```python
DUMMY_FOP = toolchain.CompilableProgram(data=ff_stages.FieldOperatorDefinition(definition=None), args=None)
```

```python
gtx.backend.DEFAULT_TRANSFORMS.step_order(DUMMY_FOP)
```

```python
@dataclasses.dataclass(frozen=True)
class SkipLinting(gtx.backend.Transforms):
    def step_order(self, inp):
        order = super().step_order(inp)
        if "past_lint" in order:
            order.remove("past_lint")  # not running "past_lint"
        return order

same_steps = dataclasses.asdict(gtx.backend.DEFAULT_TRANSFORMS)
skip_linting_transforms = SkipLinting(
    **same_steps
)
skip_linting_transforms.step_order(DUMMY_FOP)
```

## Alternative Factory

```python
class MyCodeGen:
    ...


class Cpp2BindingsGen:
    ...


class PureCpp2WorkflowFactory(gtx.program_processors.runners.gtfn.GTFNCompileWorkflowFactory):
    translation: workflow.Workflow[
        gtx.otf.stages.CompilableProgram, gtx.otf.stages.ProgramSource] = MyCodeGen()
    bindings: workflow.Workflow[
        gtx.otf.stages.ProgramSource, gtx.otf.stages.CompilableSource] = Cpp2BindingsGen()


PureCpp2WorkflowFactory(cmake_build_type=gtx.config.CMAKE_BUILD_TYPE.DEBUG)
```

## Invent new Workflow Types

```mermaid
graph LR

IN_T --> i{{split}} --> A_T --> a{{track_a}} --> B_T --> o{{combine}} --> OUT_T
i --> X_T --> x{{track_x}} --> Y_T --> o
```

```python
IN_T = typing.TypeVar("IN_T")
A_T = typing.TypeVar("A_T")
B_T = typing.TypeVar("B_T")
X_T = typing.TypeVar("X_T")
Y_T = typing.TypeVar("Y_T")
OUT_T = typing.TypeVar("OUT_T")

@dataclasses.dataclass(frozen=True)
class FullyModularDiamond(
    workflow.ChainableWorkflowMixin[IN_T, OUT_T],
    workflow.ReplaceEnabledWorkflowMixin[IN_T, OUT_T],
    typing.Protocol[IN_T, OUT_T, A_T, B_T, X_T, Y_T]
):
    split: workflow.Workflow[IN_T, tuple[A_T, X_T]]
    track_a: workflow.Workflow[A_T, B_T]
    track_x: workflow.Workflow[X_T, Y_T]
    combine: workflow.Workflow[tuple[B_T, Y_T], OUT_T]

    def __call__(self, inp: IN_T) -> OUT_T:
        a, x = self.split(inp)
        b = self.track_a(a)
        y = self.track_x(x)
        return self.combine((b, y))


@dataclasses.dataclass(frozen=True)
class PartiallyModularDiamond(
    workflow.ChainableWorkflowMixin[IN_T, OUT_T],
    workflow.ReplaceEnabledWorkflowMixin[IN_T, OUT_T],
    typing.Protocol[IN_T, OUT_T, A_T, B_T, X_T, Y_T]
):
    track_a: workflow.Workflow[A_T, B_T]
    track_x: workflow.Workflow[X_T, Y_T]

    def split(inp: IN_T) -> tuple[A_T, X_T]:
        ...

    def combine(b: B_T, y: Y_T) -> OUT_T:
        ...

    def __call__(inp: IN_T) -> OUT_T:
        a, x = self.split(inp)
        return self.combine(
            b=self.track_a(a),
            y=self.track_x(x)
        )
```
