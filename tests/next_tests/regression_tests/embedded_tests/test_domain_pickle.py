import pickle

from gt4py.next import common

I = common.Dimension("I")
J = common.Dimension("J")

def test_domain_pickle_after_slice():
    domain = common.domain(((I, (2, 4)), (J, (3, 5))))
    # use slice_at to populate cached property
    domain.slice_at[2:5, 5:7]

    pickle.dumps(domain)