from eve import codegen
from mako import template as mako_tpl

class PythonNaiveCodegen(codegen.TemplatedGenerator):
    # DATA_TYPE_TO_STR: ClassVar[Mapping[common.LocationType, str]] = MappingProxyType(
    #     {
    #         common.DataType.BOOLEAN: "bool",
    #         common.DataType.INT32: "int",
    #         common.DataType.UINT32: "unsigned_int",
    #         common.DataType.FLOAT32: "float",
    #         common.DataType.FLOAT64: "double",
    #     }
    # )
    # BUILTIN_LITERAL_TO_STR: ClassVar[Mapping[common.BuiltInLiteral, str]] = MappingProxyType(
    #     {
    #         common.BuiltInLiteral.MAX_VALUE: "std::numeric_limits<TODO>::max()",
    #         common.BuiltInLiteral.MIN_VALUE: "std::numeric_limits<TODO>::min()",
    #         common.BuiltInLiteral.ZERO: "0",
    #         common.BuiltInLiteral.ONE: "1",
    #     }
    # )

    Computation_template = mako_tpl.Template("def ${ name }(${ ', '.join(params) })")

    FieldDecl_template = "{name}"

    SidCompositeEntry_template = "{name}"

    SidComposite_template = mako_tpl.Template(
        """
        auto ${ _this_node.field_name } = tu::make<gridtools::sid::composite::keys<${ ','.join([t.tag_name for t in _this_node.entries]) }>::values>(
        ${ ','.join(entries)});
        """
    )


    # def visit_KernelCall(self, node: KernelCall, **kwargs):
    #     kernel: Kernel = kwargs["symbol_tbl_kernel"][node.name]
    #     connectivities = [self.generic_visit(conn, **kwargs) for conn in kernel.connectivities]
    #     primary_connectivity: Connectivity = kernel.symbol_tbl[kernel.primary_connectivity]
    #     sids = [self.generic_visit(s, **kwargs) for s in kernel.sids if len(s.entries) > 0]

    #     # TODO I don't like that I render here and that I somehow have the same pattern for the parameters
    #     args = [c.name for c in kernel.connectivities]
    #     args += [
    #         "gridtools::sid::get_origin({0}), gridtools::sid::get_strides({0})".format(s.field_name)
    #         for s in kernel.sids
    #         if len(s.entries) > 0
    #     ]
    #     # connectivity_args = [c.name for c in kernel.connectivities]
    #     return self.generic_visit(
    #         node,
    #         connectivities=connectivities,
    #         sids=sids,
    #         primary_connectivity=primary_connectivity,
    #         args=args,
    #         **kwargs,
    #     )
