from optimizer import WireBundleOptimizer
import matplotlib.pyplot as plt
import numpy as np


def plot_wire_bundle(
    positions: np.ndarray, radii: np.ndarray, outer_radius: float
) -> None:
    """
    Plot the optimized wire bundle layout.

    Parameters:
        positions (np.ndarray): Array of wire center coordinates of shape (n, 2).
        radii (np.ndarray): Array of wire radii of shape (n,).
        outer_radius (float): Radius of the enclosing circle.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Optimized Wire Bundle")

    # Draw outer circle
    outer = plt.Circle((0, 0), outer_radius, color="gray", fill=False, linestyle="--")
    ax.add_patch(outer)

    # Draw each wire
    for (x, y), r in zip(positions, radii):
        c = plt.Circle((x, y), r, color="steelblue", alpha=0.6)
        ax.add_patch(c)

    lim = outer_radius + np.max(radii)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True)
    plt.show()


def main() -> None:
    radii = 3 * [8.1026 / 2] + 2 * [3.3 / 2] + 43 * [1.17 / 2]

    optimizer = WireBundleOptimizer(radii)

    positions, radii, outer_radius = optimizer.solve_multi(
        500, max_iterations=200, n_jobs=-1
    )
    print(f"Best outer diameter: {outer_radius * 2:.4f}")

    plot_wire_bundle(positions, radii, outer_radius)


if __name__ == "__main__":
    main()
