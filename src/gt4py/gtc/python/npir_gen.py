from eve.codegen import TemplatedGenerator, FormatTemplate

__all__ = ["NpirGen"]


class NpirGen(TemplatedGenerator):

    Literal = FormatTemplate("np.{_this_node.dtype.name.lower()}({value})")

    ParallelOffset = FormatTemplate("{axis_name.lower()} {sign} {offset}:{axis_name.upper()} {sign} {offset}")

    SequentialOffset = FormatTemplate("{axis_name.lower()}_ {sign} {offset}")
