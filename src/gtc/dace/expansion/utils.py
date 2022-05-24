import dace
import dace.data
import dace.library
import dace.subsets

from gtc import common


def get_dace_debuginfo(node: common.LocNode):

    if node.loc is not None:
        return dace.dtypes.DebugInfo(
            node.loc.line,
            node.loc.column,
            node.loc.line,
            node.loc.column,
            node.loc.source,
        )
    else:
        return dace.dtypes.DebugInfo(0)
