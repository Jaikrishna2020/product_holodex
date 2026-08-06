from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook

import product_holodex.core.decomposition as decomp

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"


def rank_k_compression(gray_img):
    m, n = gray_img.shape

    # different rank approximations
    K = np.array([1, 5, 20, 50, 75])

    img_k = np.array([decomp.rank_k_approx(gray_img, k) for k in K])
    S = np.linalg.svd(gray_img, compute_uv=False)
    c = decomp.cumulative_energy(S)

    # Generating the image
    fig, axes = plt.subplots(1, len(K) + 1, figsize=(20, 5), constrained_layout=True)

    # original image
    axes[0].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original grace_hopper grayscale")
    axes[0].axis("off")

    for ax, k, Ak in zip(axes[1:], K, img_k, strict=True):
        ax.imshow(np.clip(Ak, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"k={k}  energy {c[k - 1]:.3f}  size {k * (m + n + 1) / (m * n):.1%}"
        )
        ax.axis("off")

    fig.savefig(OUTPUT_DIR / "rank_k_compression.png", dpi=150)
    plt.close(fig)


def pca_components(comp, mean):
    # Generating the image
    fig, axes = plt.subplots(4, 6, figsize=(12, 8.5), constrained_layout=True)

    for i in range(len(axes.flat) - 1):
        v = np.abs(comp[i]).max()
        axes.flat[i].imshow(comp[i].reshape(8, 8), cmap="gray", vmin=-v, vmax=v)
        axes.flat[i].set_title(f"comp {i}", fontsize=9)
        axes.flat[i].axis("off")

    v = np.abs(mean).max()
    axes.flat[23].imshow(
        mean.reshape(8, 8), cmap="gray", vmin=mean.min(), vmax=mean.max()
    )
    axes.flat[23].set_title("mean", fontsize=9)
    axes.flat[23].axis("off")

    fig.savefig(OUTPUT_DIR / "pca_components.png", dpi=150)
    plt.close(fig)


def pca_spectrum(S):
    ks = np.arange(1, S.size + 1)
    ce = decomp.cumulative_energy(S)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(ks, ce)
    ax.axhline(0.95, linestyle="--", color="grey")
    ax.set_title("Cumulative energy vs k (0.95 line for comparison)")
    ax.set_xlabel("k (components kept)")
    ax.set_ylabel("Cumulative energy of k components")
    k95 = np.searchsorted(ce, 0.95) + 1
    ax.annotate(f"k={k95}", xy=(k95, ce[k95 - 1]))

    fig.savefig(OUTPUT_DIR / "pca_spectrum.png", dpi=150)
    plt.close(fig)


def main():
    rgb = plt.imread(cbook.get_sample_data("grace_hopper.jpg"))
    gray_weights = np.array(
        [0.2126, 0.7152, 0.0722]
    )  # weighted as per Rec.709 luma weights

    gray_img = rgb @ gray_weights
    patches = decomp.extract_patches(gray_img, 8, 4)
    comp, S, mean = decomp.pca(patches)

    OUTPUT_DIR.mkdir(exist_ok=True)

    rank_k_compression(gray_img)
    pca_components(comp, mean)
    pca_spectrum(S)


if __name__ == "__main__":
    main()
