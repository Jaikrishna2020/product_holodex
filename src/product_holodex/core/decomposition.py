import numpy as np


def extract_patches(img, size, stride):
    """Cut a 2D array into tiles and flatten the tiles into rows (row major flattening)

    Args:
        img : Grayscale image 2D array
        size : Dimension of the tiles
        stride : step size of the tiles

    Return:
        N x size*size array - N is total number of flatten rows we get
        Function of img dimension, size and stride

    Raises:
        ValueError if img is not 2D
        ValueError if size and stride less than 1 or greater than img dimensions

    Note : When any dimension of img is such that (dim - size)%stride != 0, the reminder pixels will be truncated at the end
    """
    if img.ndim != 2:
        raise (
            ValueError(f"Image needs to be 2D. Given input is {img.ndim!r}D instead")
        )
    H, W = img.shape
    if size < 1 or size > H or size > W:
        raise (
            ValueError(
                f"size needs to be between 1 and min(H,W). Given size is {size!r}"
            )
        )
    if stride < 1:
        raise (
            ValueError(
                f"stride needs to be between 1 and min(H,W). Given stride is {stride!r}"
            )
        )

    n_rows = (H - size) // stride + 1
    n_cols = (W - size) // stride + 1
    patches = np.empty((n_rows * n_cols, size**2), dtype=float)
    i = 0
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            patches[i] = img[r : r + size, c : c + size].flatten()
            i += 1
    return patches


def pca(X, n_components=None):
    """Principal Component Analysis on the given matrix after centering it along each feature

    Args:
        X : nparray Input matrix
        n_components : Number of components to output. None implies all components i.e min(dimensions of the matrix)

    Returns:
        components : Vectors representing principal directions on centered data
        singular_values : Singular values of the matrix representing variance
        mean : Feature wise mean of the given matrix. Mean is returned to aid in reconstructing the original data if needed

    Raises:
        ValueError if input matrix is not 2D
        ValueError if n_components is not between 1 and min(X's dimensions)

    Note: We choose full_matrices=False because we're only trying to retain the prinicipal components
    """
    if X.ndim != 2:
        raise (ValueError(f"Input matrix need to be 2D. It was {X.ndim!r}D"))
    if n_components is not None and (n_components < 1 or n_components > min(X.shape)):
        raise (
            ValueError(
                f"n_components has to be between 1 and min of the dimensions of matrix. Given input was {n_components!r}"
            )
        )
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if n_components is None:
        return Vt, S, mean
    else:
        return Vt[:n_components], S[:n_components], mean


def rank_k_approx(A, k):
    """Give the rank k approximation of the given matrix
    Note : We're not apply PCA. So the given data wasn't centered. This is purely getting the closest k rank matrix

    Args:
        A : Input matrix
        k : The rank approximation we need (Must be between 1 and min(matrix dimensions))

    Returns:
        Ak : Rank k approximation of the given matrix wrt to Frobenius Norm and spectral norm

    Raises:
        ValueError if A is not a 2D matrix
        k doesn't fall between 1 and min(dimensions of A)
    """
    if A.ndim != 2:
        raise (ValueError(f"Input matrix need to be 2D. It was {A.ndim!r}D"))
    if k < 1 or k > min(A.shape):
        raise (
            ValueError(
                f"k has to be between 1 and min of dimensions of matrix. Given input was {k!r}"
            )
        )
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = U[:, :k] * S[:k] @ Vt[:k]
    return Ak


def cumulative_energy(S):
    """Compute cumulative energy until kth element for all k values

    Args:
        S : np array singular values of a matrix

    Returns:
        cumulative energy until kth element for all k values

    Raises:
        ValueError if all singular values are 0
        ValueEror if any of the singular values are negative
    """
    if np.sum(np.abs(S)) == 0:
        raise (
            ValueError(
                "All singular values are zero. Cumulative energy is not defined in this case"
            )
        )
    if (S < 0).any():
        raise (
            ValueError(
                f"One or more of the singular values are negative which is meaningless. S = {S!r}. Please provide positive values"
            )
        )
    return np.cumsum(S**2) / np.sum(S**2)
