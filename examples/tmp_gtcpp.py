from gt4py.backend.gtc_backend.gtcppir import *
from gt4py.backend.gtc_backend.gtcpp_codegen import GTCppCodegen


inacc = GTAccessor(name="in", id=0, intent=Intent.IN, extent=GTExtent.zero())
outacc = GTAccessor(name="out", id=0, intent=Intent.INOUT, extent=GTExtent.zero())
param_list = GTParamList(accessors=[inacc, outacc])
assign = AssignStmt(
    left=AccessorRef(name="out", offset=Offset.zero()),
    right=AccessorRef(name="in", offset=Offset.zero()),
)
body = [assign]
func = GTFunctor(
    name="copy", applies=[GTApplyMethod(interval=GTInterval(), body=body)], param_list=param_list
)

stage = GTStage(functor="copy", args=[ParamArg(name="in"), ParamArg(name="out")])

ms = GTMultiStage(loop_order=common.LoopOrder.PARALLEL, stages=[stage])

gtcomp = GTComputation(
    name="gt_copy_comp", parameters=[ParamArg(name="in"), ParamArg(name="out")], multistages=[ms]
)

comp = Computation(
    name="copy_comp",
    parameters=[ParamArg(name="in"), ParamArg(name="out")],
    functors=[func],
    ctrl_flow_ast=[gtcomp],
)

print(GTCppCodegen.apply(comp))
