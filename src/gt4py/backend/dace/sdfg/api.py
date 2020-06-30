import dace


def apply_transformations_repeated_recursive(sdfg, *args, **kwargs):

    for st in sdfg.nodes():
        for node in st.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                apply_transformations_repeated_recursive(node.sdfg, *args, **kwargs)

    sdfg.apply_transformations_repeated(*args, **kwargs)


def replace_recursive(sdfg, symbol, new_expr):

    for st in sdfg.nodes():
        for node in st.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                if str(symbol) in node.symbol_mapping and str(symbol) == str(
                    node.symbol_mapping[str(symbol)]
                ):
                    replace_recursive(node.sdfg, symbol, new_expr)
                    del node.symbol_mapping[str(symbol)]
                    del node.sdfg.symbols[str(symbol)]
    sdfg.replace(symbol, new_expr)
