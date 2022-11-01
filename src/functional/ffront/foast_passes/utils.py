from functional.ffront import field_operator_ast as foast


def compute_assign_indices(targets: list[foast.FieldSymbol | foast.TupleSymbol | foast.ScalarSymbol | foast.Star]) -> list[int | tuple]:
    indices = list(range(len(targets)))
    for idx, elt in enumerate(targets):
        if isinstance(elt, foast.Star):
            break
        indices[idx] = idx
    for idx, elt in reversed(list(enumerate(targets))):
        rel_idx = idx - len(targets)
        if isinstance(elt, foast.Star):
            star_lower, star_upper = max(indices), min(indices)
            indices[idx] = (star_lower, star_upper)
            break
        indices[idx] = rel_idx
    return indices
