import numpy as np
import pytest
from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next import (
    float64,
)
from gt4py.next.ffront.experimental import as_offset

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    KDim,
    cartesian_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)

def compute_expected(qc_initial: np.ndarray, scalar: float, threshold: float) -> np.ndarray:
    """Compute expected output of the scan operation.
    
    Args:
        qc_initial: Initial qc array
        scalar: Scalar value used in accumulation
        threshold: Threshold for the condition (default 5.0)
    
    Returns:
        Expected output array after simple_addition and scan
    """
    # Step 1: Apply simple_addition
    qc_tmp = qc_initial + 1.0
    
    # Step 2: Forward scan along axis 1 (KDim)
    result = np.zeros_like(qc_tmp)
    for i in range(qc_tmp.shape[0]):
        state = 0.0
        for k in range(qc_tmp.shape[1]):
            qc_in = qc_tmp[i, k]
            if qc_in > threshold:
                state = qc_in + state + scalar
            # else: state remains unchanged
            result[i, k] = state
    
    return result

class IntegrationState(NamedTuple):
    q_tmp: float

class Q(NamedTuple):
    q: cases.IKFloatField

class Q_scalar(NamedTuple):
    q: np.float64

def test_scalar_scan(cartesian_case):
    @gtx.field_operator
    def simple_addition(
        a: Q,
    ) -> Q:
        return Q(q=a.q + 1.0)
    @gtx.scan_operator(axis=KDim, forward=True, init=IntegrationState(q_tmp=0.0))
    def testee_scan(state: IntegrationState, q_in: Q_scalar, scalar: float) -> IntegrationState:
        q_tmp = q_in.q + state.q_tmp + scalar if q_in.q > 5.0 else q_in.q
        return IntegrationState(q_tmp=q_tmp)

    @gtx.field_operator
    def combined_fop(
        q_in: Q, scalar: float
    ):
        q_in = simple_addition(q_in)
        q_out = testee_scan(q_in, scalar)
        return q_out.q_tmp

    @gtx.program
    def testee(q_in: Q, q_out: cases.IKFloatField, scalar: float):
        combined_fop(q_in, scalar, out=q_out)

    q_in = cases.allocate(cartesian_case, testee, "q_in").unique()()
    q_out = cases.allocate(cartesian_case, testee, "q_out").zeros()()
    scalar = 5.0
    qc_np = q_in.q.asnumpy() if hasattr(q_in.q, 'asnumpy') else np.asarray(q_in.q)
    expected = compute_expected(qc_np, scalar, threshold=5.0)

    cases.verify(cartesian_case, testee, q_in, q_out, scalar, inout=q_out, ref=expected)
