# import dace
# from dace import registry
# from dace.codegen.targets.cpu import CPUCodeGen
# from dace import dtypes
# from dace.codegen.targets.common import sym2cpp
# from dace.codegen.targets.target import DefinedType
#
#
# @registry.autoregister_params(name="cpu")
# class CPUWithPersistent(CPUCodeGen):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.allocated_symbols = set()
#
#     #
#     def allocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
#
#         name = node.data
#         nodedesc = node.desc(sdfg)
#
#         if nodedesc.transient is False:
#             return
#
#         if nodedesc.lifetime == dace.dtypes.AllocationLifetime.Persistent:
#             if nodedesc.storage == dtypes.StorageType.CPU_Heap:
#                 # Check if array is already allocated
#                 try:
#                     self._dispatcher.defined_vars.get(name)
#                     return  # Array was already allocated in this or upper scopes
#                 except KeyError:  # Array not allocated yet
#                     pass
#                 function_stream.write("%s *%s;\n" % (nodedesc.dtype.ctype, name))
#                 self._frame._initcode.write(
#                     "%s = new %s DACE_ALIGN(64)[%s];\n"
#                     % (name, nodedesc.dtype.ctype, sym2cpp(nodedesc.total_size)),
#                     sdfg,
#                     state_id,
#                     node,
#                 )
#                 self._dispatcher.defined_vars.add(
#                     name, DefinedType.Pointer, "%s *" % nodedesc.dtype.ctype
#                 )
#                 self.allocated_symbols.add(name)
#             elif nodedesc.storage == dtypes.StorageType.CPU_ThreadLocal:
#                 # Check if array is already allocated
#                 try:
#                     self._dispatcher.defined_vars.get(name)
#                     return  # Array was already allocated in this or upper scopes
#                 except KeyError:  # Array not allocated yet
#                     pass
#                 function_stream.write(
#                     "%s *%s;\n#pragma omp threadprivate(%s)\n" % (nodedesc.dtype.ctype, name, name)
#                 )
#
#                 tile_shapes = dict(
#                     I=nodedesc.tile_size[0], J=nodedesc.tile_size[1], K=dace.symbol("K")
#                 )
#                 subs = {
#                     (a if isinstance(a, dace.symbol) else a.args[1]): tile_shapes[
#                         str(a if isinstance(a, dace.symbol) else a.args[1])
#                     ]
#                     for a in nodedesc.total_size.args
#                 }
#                 total_size = nodedesc.total_size.subs(subs)
#                 self._frame._initcode.write(
#                     """#pragma omp parallel
# {
#     %s = new %s DACE_ALIGN(64)[ %s];
# }
# """
#                     % (name, nodedesc.dtype.ctype, sym2cpp(total_size)),
#                     sdfg,
#                     state_id,
#                     node,
#                 )
#                 self._dispatcher.defined_vars.add(
#                     name, DefinedType.Pointer, "%s *" % nodedesc.dtype.ctype
#                 )
#                 self.allocated_symbols.add(name)
#         else:
#             super().allocate_array(sdfg, dfg, state_id, node, function_stream, callsite_stream)
#
#     def deallocate_array(self, sdfg, dfg, state_id, node, function_stream, callsite_stream):
#
#         name = node.data
#         nodedesc = node.desc(sdfg)
#
#         if nodedesc.transient is False:
#             return
#
#         if nodedesc.storage == dtypes.StorageType.CPU_Heap:
#             if name in self.allocated_symbols:
#                 print("deleting %s" % node.data)
#                 self._frame._exitcode.write("delete[] %s;\n" % node.data, sdfg, state_id, node)
#                 self.allocated_symbols.remove(name)
#         elif nodedesc.storage == dtypes.StorageType.CPU_ThreadLocal:
#             if name in self.allocated_symbols:
#                 print("deleting %s" % node.data)
#                 self._frame._exitcode.write(
#                     """#pragma omp parallel
# {
#     delete[] %s;
# }
# """
#                     % node.data,
#                     sdfg,
#                     state_id,
#                     node,
#                 )
#                 self.allocated_symbols.remove(name)
#         else:
#             super().deallocate_array(sdfg, dfg, state_id, node, function_stream, callsite_stream)
