from optimizer import WireBundleOptimizer


def main() -> None:
    radii = 3 * [8.1026 / 2] + 2 * [3.3 / 2] + 43 * [1.17 / 2]

    optimizer = WireBundleOptimizer(radii)

    _, outer_d = optimizer.solve_multi(40)
    print(f"Best outer diameter: {outer_d:.4f}")

    optimizer.plot()


if __name__ == "__main__":
    main()
