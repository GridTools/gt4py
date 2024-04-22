import warnings
from typing import Any

from gt4py._core import definitions as core_defs
from gt4py.next import backend as next_backend, common, embedded as next_embedded, allocators as next_allocators
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.program_processors import processor_interface as ppi, modular_executor
from gt4py.next.ffront import stages as ffront_stages, past_process_args

class EmbeddedBackend(next_backend.Backend):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms_fop: next_backend.FieldopTransformWorkflow = next_backend.DEFAULT_FIELDOP_TRANSFORMS
    transforms_prog: next_backend.ProgramTransformWorkflow = next_backend.DEFAULT_PROG_TRANSFORMS

    def __call__(
            self,
            program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
            *args: tuple[Any],
            #offset_provider: dict[str, common.Dimension],
            **kwargs: dict[str, Any],
    ) -> None:
        match program:
            case ffront_stages.ProgramDefinition():
                offset_provider = kwargs.pop("offset_provider")
                return self.call_program(program, *args, offset_provider=offset_provider, **kwargs)
            case ffront_stages.FieldOperatorDefinition():
                if "from_fieldop" in kwargs:
                    kwargs.pop("from_fieldop")
                return self.call_fieldoperator(program, *args, **kwargs)

    def call_program(
            self,
            program: ffront_stages.ProgramDefinition,
            *args: tuple[Any],
            offset_provider: dict[str, common.Dimension],
            **kwargs: dict[str, Any],
    ) -> None:
        warnings.warn(
            UserWarning(
                f"Field View Program '{program.definition.__name__}': Using Python execution, consider selecting a perfomance backend."
            ),
            stacklevel=2,
        )
        with next_embedded.context.new_context(offset_provider=offset_provider) as ctx:
            # TODO(ricoh): check if rewriting still needed
            #rewritten_args, size_args, kwargs = past_process_args._process_args(
            #    self.past_stage.past_node, args, kwargs
            #)
            if "out" in kwargs:
                args = (*args, kwargs.pop("out"))
            return ctx.run(program.definition, *args, **kwargs)

    def call_fieldoperator(
            self,
            program: ffront_stages.FieldOperatorDefinition,
            *args: tuple[Any],
            **kwargs: dict[str, Any],
    ) -> None:
        attributes = program.attributes
        if attributes is not None and any(
                has_scan_op_attribute := [
                    attribute in attributes for attribute in ["init", "axis", "forward"]
                ]
        ):
            assert all(has_scan_op_attribute)
            forward = attributes["forward"]
            init = attributes["init"]
            axis = attributes["axis"]
            op = embedded_operators.ScanOperator(
                program.definition, forward, init, axis
            )
        else:
            op = embedded_operators.EmbeddedOperator(program.definition)
        return embedded_operators.field_operator_call(op, args, kwargs)


default_embedded = EmbeddedBackend(executor=modular_executor.ModularExecutor(name="new_embedded", otf_workflow=None), allocator=None)
