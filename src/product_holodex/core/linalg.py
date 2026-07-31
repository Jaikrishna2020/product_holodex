import numpy as np


def _normal(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def _qr(A, b):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)


def _svd(A, b, rcond=1e-15):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Spinv = [1 / s if (s > rcond * np.max(S)) else 0 for s in S]
    return Vt.T @ np.diag(Spinv) @ U.T @ b


_SOLVERS = {"normal": _normal, "qr": _qr, "svd": _svd}

METHODS = tuple(_SOLVERS)


def lstsq(A, b, method="qr", **kwargs):
    """
    Finds the least squares estimate of a linear system of equations

    Args:
        A (np matrix) : numpy matrix
        b (np vector) : numpy vector
        method (string) : specifies which method is used to compute least squares
        **kwargs : present to additionally intake rcond (threshold ) for svd method. rcond not needed for normal and qr

    Return:
        numpy vector : i.e the least squares estimate of the given system

    Raises:
        ValueError if the given method is not available
        LinAlgError when the matrix is rank deficient and method is QR or normal

    Methods:
        qr : computes using QR decomposition. Estimate is more precise even with ill conditioned matrices. Error scales as k. Throws linalg error for rank deficient matrices
        normal : computes by forming A.T @ A and solving the system. For ill conditioned matrices, the precision is greatly affected since error scales as k**2. Throws linalg error for rank deficient matrices
        svd : computes using SVD and applying pseudoinverse. Returns min norm solution even when singular matrix is seen. rcond is the threshold which decides below which ratio of singular value we trim the rank and drop the information
    """
    try:
        fn = _SOLVERS[method]
    except KeyError:
        raise ValueError(
            f"Method must be one of {METHODS}. Method entered {method!r}"
        ) from None
    return fn(A, b, **kwargs)
