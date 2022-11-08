from functional.ffront import field_operator_ast as foast


def compute_assign_indices(
    targets: list[foast.FieldSymbol | foast.TupleSymbol | foast.ScalarSymbol | foast.Starred],
    num_elts: int
) -> list[int | tuple]:
    """Computes a list of relative indices, mapping each target to its respective value(s).

    This function is used when mapping a tuple of targets to a tuple of values, and also handles the
    case in which tuple unpacking is done using a Starred operator.

    Examples:

        The below are examples of different types of unpacking and the
        generated indices. Note that the indices in the tuple correspond
        to the lower and upper slice indices of a Starred variable.

        a, *b, c = (1, 2, 3, 4)
        (0, (1, -1), -1)

        *a, b, c = (1, 2, 3, 4)
        [(0, -2), -2, -1]

        a, b, *c = (1, 2, 3, 4)
        [0, 1, (2, 5)]
    """
    indices = list(range(len(targets)))

    for idx, elt in enumerate(targets):
        if isinstance(elt, foast.Starred):
            break
        indices[idx] = idx
    for idx, elt in reversed(list(enumerate(targets))):
        rel_idx = idx - len(targets)
        if isinstance(elt, foast.Starred):
            star_lower, star_upper = max(indices), min(indices)
            if star_upper == 0:
                star_upper = num_elts
            indices[idx] = (star_lower, star_upper)
            break
        indices[idx] = rel_idx
    return indices
