from gt4py.gtc import gtir


def prune_unused_parameters(node: gtir.Stencil):
    """
    Removes unused parameters from the gtir signature.

    (Maybe this pass should go into a later stage. If you need to touch this pass,
    e.g. when the definition_ir gets removed, consider moving it to a more appropriate
    level. Maybe to the backend IR?)
    """
    assert isinstance(node, gtir.Stencil)
    used_variables = (
        node.iter_tree()
        .if_isinstance(gtir.FieldAccess, gtir.ScalarAccess)
        .getattr("name")
        .to_list()
    )
    used_params = list(filter(lambda param: param.name in used_variables, node.params))
    return gtir.Stencil(name=node.name, params=used_params, vertical_loops=node.vertical_loops)
