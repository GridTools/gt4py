from gt4py.backend.gtc_backend.common import BinaryOperator, DataType
from gt4py.backend.gtc_backend.gtcppir import *
from gt4py.backend.gtc_backend.gtcpp_codegen import GTCppCodegen


# copy function

copy_func = GTFunctor(
    name="copy_function",
    param_list=GTParamList(
        accessors=[
            GTAccessor(name="out", intent=Intent.INOUT, extent=GTExtent.zero(), id=0),
            GTAccessor(name="in", intent=Intent.IN, extent=GTExtent.zero(), id=1),
        ]
    ),
    applies=[
        GTApplyMethod(
            interval=GTInterval(),
            body=[
                AssignStmt(
                    left=AccessorRef(name="out", offset=Offset.zero()),
                    right=AccessorRef(name="in", offset=Offset.zero()),
                )
            ],
        )
    ],
)

lap_func = GTFunctor(
    name="lap_function",
    param_list=GTParamList(
        accessors=[
            GTAccessor(name="out", intent=Intent.INOUT, extent=GTExtent.zero(), id=0),
            GTAccessor(
                name="in", intent=Intent.IN, extent=GTExtent(i=(-1, 1), j=(-1, 1), k=(0, 0)), id=1
            ),
        ]
    ),
    applies=[
        GTApplyMethod(
            interval=GTInterval(),
            body=[
                AssignStmt(
                    left=AccessorRef(name="out", offset=Offset.zero()),
                    right=BinaryOp(
                        op=common.BinaryOperator.SUB,
                        left=BinaryOp(
                            op=common.BinaryOperator.MUL,
                            left=Literal(value="4", vtype=common.DataType.FLOAT64),
                            right=AccessorRef(name="in", offset=Offset.zero()),
                        ),
                        right=BinaryOp(
                            op=common.BinaryOperator.ADD,
                            left=BinaryOp(
                                op=common.BinaryOperator.ADD,
                                left=AccessorRef(name="in", offset=Offset(i=-1, j=0, k=0)),
                                right=AccessorRef(name="in", offset=Offset(i=1, j=0, k=0)),
                            ),
                            right=BinaryOp(
                                op=common.BinaryOperator.ADD,
                                left=AccessorRef(name="in", offset=Offset(j=-1, i=0, k=0)),
                                right=AccessorRef(name="in", offset=Offset(j=1, i=0, k=0)),
                            ),
                        ),
                    )
                    # right=BinaryOp(op=common.BinaryOperator.SUB, left=, right=))
                )
            ],
        )
    ],
)
flx_func = GTFunctor(
    name="flx_function",
    param_list=GTParamList(
        accessors=[
            GTAccessor(name="out", intent=Intent.INOUT, extent=GTExtent.zero(), id=0),
            GTAccessor(
                name="in", intent=Intent.IN, extent=GTExtent(i=(0, 1), j=(0, 0), k=(0, 0)), id=1
            ),
            GTAccessor(
                name="lap", intent=Intent.IN, extent=GTExtent(i=(0, 1), j=(0, 0), k=(0, 0)), id=2
            ),
        ]
    ),
    applies=[
        GTApplyMethod(
            interval=GTInterval(),
            body=[
                VarDecl(
                    name="res",
                    vtype=common.DataType.FLOAT64,
                    init=BinaryOp(
                        op=common.BinaryOperator.SUB,
                        left=AccessorRef(name="lap", offset=Offset(i=1, j=0, k=0)),
                        right=AccessorRef(name="lap", offset=Offset.zero()),
                    ),
                ),
                AssignStmt(
                    left=AccessorRef(name="out", offset=Offset.zero()),
                    right=TernaryOp(
                        cond=BinaryOp(
                            op=common.BinaryOperator.GT,
                            left=BinaryOp(
                                op=common.BinaryOperator.MUL,
                                left=VarAccess(name="res"),
                                right=BinaryOp(
                                    op=common.BinaryOperator.SUB,
                                    left=AccessorRef(
                                        name="in",
                                        offset=Offset(i=1, j=0, k=0),
                                    ),
                                    right=AccessorRef(name="in", offset=Offset.zero()),
                                ),
                            ),
                            right=Literal(value="0", vtype=common.DataType.FLOAT64),
                        ),
                        true_expr=Literal(value="0", vtype=common.DataType.FLOAT64),
                        false_expr=VarAccess(name="res"),
                    ),
                ),
            ],
        )
    ],
)


fly_func = GTFunctor(
    name="fly_function",
    param_list=GTParamList(
        accessors=[
            GTAccessor(name="out", intent=Intent.INOUT, extent=GTExtent.zero(), id=0),
            GTAccessor(
                name="in", intent=Intent.IN, extent=GTExtent(i=(0, 0), j=(0, 1), k=(0, 0)), id=1
            ),
            GTAccessor(
                name="lap", intent=Intent.IN, extent=GTExtent(i=(0, 0), j=(0, 1), k=(0, 0)), id=2
            ),
        ]
    ),
    applies=[
        GTApplyMethod(
            interval=GTInterval(),
            body=[
                VarDecl(
                    name="res",
                    vtype=common.DataType.FLOAT64,
                    init=BinaryOp(
                        op=common.BinaryOperator.SUB,
                        left=AccessorRef(name="lap", offset=Offset(i=0, j=1, k=0)),
                        right=AccessorRef(name="lap", offset=Offset.zero()),
                    ),
                ),
                AssignStmt(
                    left=AccessorRef(name="out", offset=Offset.zero()),
                    right=TernaryOp(
                        cond=BinaryOp(
                            op=common.BinaryOperator.GT,
                            left=BinaryOp(
                                op=common.BinaryOperator.MUL,
                                left=VarAccess(name="res"),
                                right=BinaryOp(
                                    op=common.BinaryOperator.SUB,
                                    left=AccessorRef(
                                        name="in",
                                        offset=Offset(i=0, j=1, k=0),
                                    ),
                                    right=AccessorRef(name="in", offset=Offset.zero()),
                                ),
                            ),
                            right=Literal(value="0", vtype=common.DataType.FLOAT64),
                        ),
                        true_expr=Literal(value="0", vtype=common.DataType.FLOAT64),
                        false_expr=VarAccess(name="res"),
                    ),
                ),
            ],
        )
    ],
)
out_func = GTFunctor(
    name="out_function",
    param_list=GTParamList(
        accessors=[
            GTAccessor(name="out", intent=Intent.INOUT, extent=GTExtent.zero(), id=0),
            GTAccessor(name="in", intent=Intent.IN, extent=GTExtent.zero(), id=1),
            GTAccessor(
                name="flx", intent=Intent.IN, extent=GTExtent(i=(-1, 0), j=(0, 0), k=(0, 0)), id=2
            ),
            GTAccessor(
                name="fly", intent=Intent.IN, extent=GTExtent(i=(0, 0), j=(-1, 0), k=(0, 0)), id=3
            ),
            GTAccessor(name="coeff", intent=Intent.IN, extent=GTExtent.zero(), id=4),
        ]
    ),
    applies=[
        GTApplyMethod(
            interval=GTInterval(),
            body=[
                AssignStmt(
                    left=AccessorRef(name="out", offset=Offset.zero()),
                    right=BinaryOp(
                        op=common.BinaryOperator.SUB,
                        left=AccessorRef(name="in", offset=Offset.zero()),
                        right=BinaryOp(
                            op=common.BinaryOperator.MUL,
                            left=AccessorRef(name="coeff", offset=Offset.zero()),
                            right=BinaryOp(
                                op=common.BinaryOperator.ADD,
                                left=BinaryOp(
                                    op=common.BinaryOperator.SUB,
                                    left=AccessorRef(name="flx", offset=Offset.zero()),
                                    right=AccessorRef(name="flx", offset=Offset(i=-1, j=0, k=0)),
                                ),
                                right=BinaryOp(
                                    op=common.BinaryOperator.SUB,
                                    left=AccessorRef(name="fly", offset=Offset.zero()),
                                    right=AccessorRef(name="fly", offset=Offset(i=0, j=-1, k=0)),
                                ),
                            ),
                        ),
                    ),
                )
            ],
        )
    ],
)
# stage = GTStage(functor="copy", args=[ParamArg(name="in"), ParamArg(name="out")])

ms = GTMultiStage(
    loop_order=common.LoopOrder.PARALLEL,
    stages=[
        GTStage(functor="lap_function", args=[ParamArg(name="lap"), ParamArg(name="in")]),
        GTStage(
            functor="flx_function",
            args=[ParamArg(name="flx"), ParamArg(name="in"), ParamArg(name="lap")],
        ),
        GTStage(
            functor="fly_function",
            args=[ParamArg(name="fly"), ParamArg(name="in"), ParamArg(name="lap")],
        ),
        GTStage(
            functor="out_function",
            args=[
                ParamArg(name="out"),
                ParamArg(name="in"),
                ParamArg(name="flx"),
                ParamArg(name="fly"),
                ParamArg(name="coeff"),
            ],
        ),
    ],
    caches=[IJCache(name="lap"), IJCache(name="flx"), IJCache(name="fly")],
)

gtcomp = GTComputation(
    name="gt_hdiff_commp",
    temporaries=[
        Temporary(name="lap", vtype=common.DataType.FLOAT64),
        Temporary(name="flx", vtype=common.DataType.FLOAT64),
        Temporary(name="fly", vtype=common.DataType.FLOAT64),
    ],
    parameters=[ParamArg(name="in"), ParamArg(name="coeff"), ParamArg(name="out")],
    multistages=[ms],
)

comp = Computation(
    name="horizontal_diffusion",
    parameters=[ParamArg(name="in"), ParamArg(name="coeff"), ParamArg(name="out")],
    functors=[copy_func, lap_func, flx_func, fly_func, out_func],
    ctrl_flow_ast=[gtcomp],
)

print(GTCppCodegen.apply(comp))
