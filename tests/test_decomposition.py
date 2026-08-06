import numpy as np
import pytest

import product_holodex.core.decomposition as decomp

TALL = np.random.default_rng(0).standard_normal((60, 40))


def rel_fro(A, B):
    return np.linalg.norm(A - B, "fro") / np.linalg.norm(A, "fro")


def test_extract_patches_exact():
    A = np.arange(16).reshape(4, 4)
    patches = decomp.extract_patches(A, 2, 2)
    gt = np.array(
        [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], dtype=float
    )
    assert np.array_equal(patches, gt)
    assert patches.dtype == np.float64


@pytest.mark.parametrize(
    "shape, size, stride, expected", [((4, 4), 2, 1, (9, 4)), ((9, 9), 4, 2, (9, 16))]
)
def test_extract_patches_shapes(shape, size, stride, expected):
    A = np.arange(shape[0] * shape[1]).reshape(shape)
    patches = decomp.extract_patches(A, size, stride)

    assert patches.shape == expected


def test_extract_patches_uneven():
    A = np.arange(81).reshape(9, 9)
    patches = decomp.extract_patches(A, 4, 2)

    assert np.array_equal(patches[-1], A[4:8, 4:8].ravel())


def test_pca_centering():
    X = np.array([[99, 101], [100, 100], [101, 99], [100.5, 99.5], [99.5, 100.5]])
    components, S, mean = decomp.pca(X)
    _, _, Vt_uncentered = np.linalg.svd(X, full_matrices=False)
    # Centered first component matches gt
    assert abs(components[0] @ (np.array([1, -1]) / 2**0.5)) > 0.99
    # Uncentered first component points in the mean direction
    assert abs(Vt_uncentered[0] @ (mean / np.linalg.norm(mean, 2))) > 0.99
    # Centered and uncentered components are almost orthogonal in this case
    assert abs(Vt_uncentered[0] @ components[0]) < 1e-8
    # The input is collinear. So only only principal component survives
    assert np.isclose(S[1], 0.0, atol=1e-12)


@pytest.mark.parametrize("shape", [(60, 40), (40, 60)])
def test_full_rank_k_approx(shape):
    rng = np.random.default_rng(0)
    mat = rng.standard_normal(shape)
    assert rel_fro(mat, decomp.rank_k_approx(mat, min(shape))) < 1e-12


def test_rank_k_approx_exact_at_true_rank():
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((60, 60)))
    V, _ = np.linalg.qr(rng.standard_normal((40, 40)))
    sv = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    A = (U[:, : sv.size] * sv) @ V[:, : sv.size].T
    assert rel_fro(A, decomp.rank_k_approx(A, sv.size)) < 1e-12


@pytest.mark.parametrize("k", [1, 5, 20, 39])
def test_eckart_young(k):
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((60, 40))
    S = np.linalg.svd(mat, compute_uv=False)
    matk = decomp.rank_k_approx(mat, k)
    assert np.isclose(
        np.linalg.norm(mat - matk, "fro") ** 2,
        np.sum(S**2) - np.cumsum(S**2)[k - 1],
        rtol=1e-12,
    )


def test_cumulative_energy_exact():

    S_exact = np.array([3.0, 4.0])

    assert np.array_equal(decomp.cumulative_energy(S_exact), np.array([0.36, 1.0]))


@pytest.mark.parametrize("k", [1, 5, 20, 39])
def test_cumulative_energy_random(k):
    S = np.linalg.svd(TALL, compute_uv=False)
    matk = decomp.rank_k_approx(TALL, k)
    assert np.isclose(
        rel_fro(TALL, matk) ** 2, 1.0 - decomp.cumulative_energy(S)[k - 1], rtol=1e-12
    )


def test_cumulative_energy_scale_invariance():
    S = np.linalg.svd(TALL, compute_uv=False)
    assert np.allclose(
        decomp.cumulative_energy(S), decomp.cumulative_energy(S * 1e6), rtol=1e-12
    )


@pytest.mark.parametrize(
    "fn, args, msg",
    [
        (decomp.pca, (np.array([1.0, 2.0, 3.0]),), "need to be 2D"),
        (decomp.rank_k_approx, (np.array([[1.0], [2.0]]), 3), "between 1 and min"),
        (
            decomp.cumulative_energy,
            (np.array([-1.0, 3.0]),),
            "singular values are negative",
        ),
    ],
)
def test_raises(fn, args, msg):
    with pytest.raises(ValueError, match=msg):
        fn(*args)
