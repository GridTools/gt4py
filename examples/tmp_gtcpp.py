from gt4py.backend.gtc_backend.gtcppir import *
from gt4py.backend.gtc_backend.gtcpp_codegen import GTCppCodegen


inacc = GTAccessor(name="in", id=0, intent=Intent.IN, extent=GTExtent.zero())
outacc = GTAccessor(name="out", id=0, intent=Intent.INOUT, extent=GTExtent.zero())
param_list = GTParamList(accessors=[inacc, outacc])

print(GTCppCodegen.apply(param_list))
