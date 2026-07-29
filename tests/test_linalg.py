import numpy as np
import pytest

import product_holodex.core.linalg

METHODS = ["normal", "qr"]


@pytest.mark.parametrize("method", METHODS)
def test_well_conditioned(method):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((15, 3))
    b = rng.standard_normal(15)

    np_solution = np.linalg.lstsq(A, b, rcond=None)[0]
    our_solution = product_holodex.core.linalg.lstsq(A, b, method)
    r = A @ our_solution - b
    assert np.allclose(np_solution, our_solution, rtol=1e-14, atol=1e-14)
    assert np.isclose(0.0, np.abs(A.T @ r).max(), atol=1e-14)


def test_ill_conditioned():
    s = np.logspace(0, -8, 3)
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((15, 3)))
    Vt, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    A = U @ np.diag(s) @ Vt
    b = rng.standard_normal(15)

    np_solution = np.linalg.lstsq(A, b, rcond=None)[0]
    our_qr_solution = product_holodex.core.linalg.lstsq(A, b, "qr")
    our_normal_solution = product_holodex.core.linalg.lstsq(A, b, "normal")
    err_qr = np.linalg.norm(our_qr_solution - np_solution, 1)
    err_normal = np.linalg.norm(our_normal_solution - np_solution, 1)
    np_norm = np.linalg.norm(np_solution, 1)

    assert err_qr / np_norm < 1e-7
    assert err_normal / np_norm > 1e-4
    assert err_normal / err_qr > 1e3
