import numpy as np


def lstsq(A, b, method="qr"):
    """
    Finds the least squares estimate of a linear system of equations

    Args:
        A (np matrix) : numpy matrix
        b (np vector) : numpy vector
        method (string) : specifies which method is used to compute least squares

    Return:
        numpy vector : i.e the least squares estimate of the given system

    Raises:
        ValueError if the given method is not available

    Methods:
        qr : computes using QR decomposition. Estimate is more precise even with ill conditioned matrices
        normal : computes by forming A.T @ A and solving the system. For ill conditioned matrices, the precision is greatly affected
    """
    if method == "normal":
        return np.linalg.solve(A.T @ A, A.T @ b)
    elif method == "qr":
        Q, R = np.linalg.qr(A)
        return np.linalg.solve(R, Q.T @ b)
    else:
        raise ValueError("method has to be normal or qr")
