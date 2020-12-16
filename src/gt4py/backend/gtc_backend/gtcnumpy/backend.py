from gt4py.backend.base import BaseBackend, BaseModuleGenerator, CLIBackendMixin, register
from gt4py.gtc import gtir
from gt4py.gtc.gtir_to_oir import GTIRToOIR
from gt4py.gtc.passes.fields_metadata_pass import FieldsMetadataPass
from gt4py.gtc.passes.gtir_set_dtype import GTIRSetDtype
from gt4py.gtc.python import npir
from gt4py.gtc.python.npir_gen import NpirGen
from gt4py.gtc.python.oir_to_npir import OirToNpir


class GTCModuleGenerator(BaseModuleGenerator):



@register
class GTCNumpyBackend(BaseBackend, CLIBackendMixin):
    """NumPy backend using gtc."""

    name = "gtc:numpy"
    options: ClassVar[Dict[str, Any]] = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": debug_layout,
        "is_compatible_layout": debug_is_compatible_layout,
        "is_compatible_type": debug_is_compatible_type,
    }
    languages = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = GTCModuleGenerator

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        super().generate_bindings(language_name)
        return {"computation.py": NpirGen.apply(self.npir)}

    def make_module_source(self) -> str:
        field_info = FieldInfoGenerator.apply(self.gtir)
        args_data = ArgsData()
        args_data.field_info = field_info

    def _make_gtir(self) -> gtir.Stencil
        gtir = FieldsMetadataPass().visit(DefIRToGTIR.apply(self.builder.definition_ir))
        return GTIRSetDtype().visit(gtir)

    def _make_npir(self) -> npir.Computation
        gtir = FieldsMetadataPass().visit(DefIRToGTIR.apply(self.builder.definition_ir))
        type_deduced_gtir = GTIRSetDtype().visit(gtir)
        oir = GTIRToOIR().visit(dtype_deduced)
        return OirToNpir().visit(oir)

    @property
    def gtir(self) -> gtir.Stencil:
        key = "gtc:gtir"
        if not self.builder.backend_data[key]:
            self.builder.with_backend_data({key: self._make_gtir()})
        return self.builder.backend_data[key]

    @property
    def npir(self) -> npir.Computation:
        key = "gtcnumpy:npir"
        if not self.builder.backend_data[key]:
            self.builder.with_backend_data({key: self._make_npir()})
        return self.builder.backend_data[key]
