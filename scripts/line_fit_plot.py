from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import product_holodex.core.linalg


def main():
    # y = mx + c
    m_true, c_true = [3.0, -0.5]
    rng = np.random.default_rng(0)
    x = np.linspace(0, 20, 80)
    y = m_true * x + c_true + rng.normal(0, 8, size=x.shape)

    A = np.column_stack([x, np.ones_like(x)])

    m_soln, c_soln = product_holodex.core.linalg.lstsq(A, y, method="svd")

    plt.scatter(x, y, color="black", label="Generated Data")
    plt.plot(x, m_true * x + c_true, color="blue", label="Reference line")
    plt.plot(x, m_soln * x + c_soln, color="red", label="Least Square Fit line")

    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"
    plt.savefig(OUTPUT_DIR / "line_fit_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
