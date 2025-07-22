import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

MAX_ITER = 200
N_SPAWNS = 50


class WireBundleOptimizer:
    def __init__(self, radii: list[float]) -> None:
        """
        Initialize the optimizer with the given wire radii.

        Parameters:
            radii (list[float]): List of wire radii.
        """
        self.radii = np.array(radii, dtype=float)
        self.n = len(radii)  # number of wires

        # upper triangle indices for unique wire pairs
        self.i_idx, self.j_idx = np.triu_indices(self.n, 1)

        self.positions = np.zeros((self.n, 2))  # Final wire positions
        self.outer_radius = 0.0  # Final bundle radius

    def _unpack(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Unpack optimization vector into coordinates and outer radius.

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            tuple[np.ndarray, float]: Coordinates of wires and outer radius.
        """
        coords = x[:-1].reshape(self.n, 2)
        R = x[-1]
        return coords, R

    def _objective(self, x: np.ndarray) -> float:
        """
        Objective: minimize the outer radius.

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            float: The outer radius (last element of x).
        """
        return x[-1]

    def _grad_objective(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of the objective function.

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            np.ndarray: Gradient vector, which is zero for all but the last element.
        """
        g = np.zeros_like(x)
        g[-1] = 1.0
        return g

    def _constraint_origin(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure each wire lies entirely within the outer radius.

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            np.ndarray: Array of constraints for each wire.
        """
        coords, R = self._unpack(x)
        return R - (np.linalg.norm(coords, axis=1) + self.radii)

    def _jac_constraint_origin(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of the origin constraint.

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            np.ndarray: Jacobian matrix of the constraints.
        """
        coords, _ = self._unpack(x)
        n_vars = x.size
        J = np.zeros((self.n, n_vars))
        for i in range(self.n):
            xi = coords[i]
            norm = np.linalg.norm(xi)
            if norm > 0:
                J[i, 2 * i : 2 * i + 2] = -xi / norm
            J[i, -1] = 1.0
        return J

    def _constraint_pairs(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure wires do not overlap (pairwise).

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            np.ndarray: Array of pairwise distance constraints.
        """
        coords, _ = self._unpack(x)
        diffs = coords[self.i_idx] - coords[self.j_idx]  # Pairwise differences
        d_centers = np.linalg.norm(diffs, axis=1)  # Distances between wire centers
        min_allowed = self.radii[self.i_idx] + self.radii[self.j_idx]
        return d_centers - min_allowed

    def _jac_constraint_pairs(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of the pairwise distance constraints.

        Parameters:
            x (np.ndarray): Optimization vector containing flattened coordinates and outer radius.

        Returns:
            np.ndarray: Jacobian matrix of the pairwise distance constraints.
        """
        coords, _ = self._unpack(x)
        m = len(self.i_idx)
        n_vars = x.size
        J = np.zeros((m, n_vars))

        # Fill in the Jacobian for each pair
        for k, (i, j) in enumerate(zip(self.i_idx, self.j_idx)):
            diff = coords[i] - coords[j]  # Difference vector between wire centers
            norm = np.linalg.norm(diff)  # Normalize the difference vector
            if norm > 0:
                # Build the jacobian for the pairwise constraint
                grad = diff / norm
                J[k, 2 * i : 2 * i + 2] = grad
                J[k, 2 * j : 2 * j + 2] = -grad
        return J

    def _initial_guess_spiral(self) -> np.ndarray:
        """
        Heuristic spiral-like layout for initial guess.
        This arranges wires in a spiral pattern, ensuring they are spaced out.

        Returns:
            np.ndarray: Initial guess for wire positions and outer radius.
        """
        coords = np.zeros((self.n, 2))
        order = np.argsort(-self.radii)
        angle = 0.0
        radius = 0.0
        step = 2 * np.pi / max(self.n, 1)  # Step size for spiral

        # Arrange wires in a spiral pattern
        for idx in order:
            radius += self.radii[idx] * 1.5  # Increase radius to avoid overlap
            coords[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            angle += step
        return np.concatenate([coords.flatten(), [radius + np.max(self.radii)]])

    def solve(self, x0: np.ndarray | None = None) -> tuple[np.ndarray, float, bool]:
        """
        Solve the optimization from a given initial guess.

        Returns:
            (coords, outer_radius, success)
        """
        if x0 is None:
            x0 = self._initial_guess_spiral()

        # Constraints for the optimization
        cons = [
            {
                "type": "ineq",
                "fun": self._constraint_origin,
                "jac": self._jac_constraint_origin,
            },
            {
                "type": "ineq",
                "fun": self._constraint_pairs,
                "jac": self._jac_constraint_pairs,
            },
        ]

        # Run the optimization
        res = minimize(
            fun=self._objective,
            x0=x0,
            method="SLSQP",
            jac=self._grad_objective,
            constraints=cons,
            options={"maxiter": MAX_ITER, "ftol": 1e-12, "disp": False},
        )

        coords, R = self._unpack(res.x)
        return coords, R, res.success

    def solve_multi(
        self, N: int, n_jobs: int = -1, random_state: int | None = None
    ) -> tuple[np.ndarray, float]:
        """
        Run multiple parallel optimizations from varied initial guesses (spiral + random).

        Returns:
            Best (coords, diameter)
        """
        rng = np.random.default_rng(random_state)
        spiral_guess = self._initial_guess_spiral()
        _, R0 = self._unpack(spiral_guess)

        # Combine spiral and N-1 random initial guesses
        initial_guesses = [spiral_guess]
        for _ in range(N - 1):
            coords_rand = rng.uniform(-R0, R0, size=(self.n, 2))
            x0_rand = np.concatenate([coords_rand.flatten(), [R0]])
            initial_guesses.append(x0_rand)

        # Run parallel optimizations
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.solve)(x0) for x0 in initial_guesses
        )

        # Select best result
        best_diam = np.inf
        best_coords = None
        for coords, R, success in results:
            if success:
                diam = 2 * R
                if diam < best_diam:
                    best_diam = diam
                    best_coords = coords

        self.positions = best_coords
        self.outer_radius = best_diam / 2
        return best_coords, best_diam

    def plot(self) -> None:
        """
        Plot the optimized wire bundle layout.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_title("Optimized Wire Bundle")

        # Draw outer circle
        outer = plt.Circle(
            (0, 0), self.outer_radius, color="gray", fill=False, linestyle="--"
        )
        ax.add_patch(outer)

        # Draw each wire
        for (x, y), r in zip(self.positions, self.radii):
            c = plt.Circle((x, y), r, color="steelblue", alpha=0.6)
            ax.add_patch(c)

        lim = self.outer_radius + np.max(self.radii)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    radii = 3 * [8.1026 / 2] + 2 * [3.3 / 2] + 43 * [1.17 / 2]

    optimizer = WireBundleOptimizer(radii)

    coords, outer_d = optimizer.solve_multi(N=N_SPAWNS, n_jobs=-1, random_state=42)
    print(f"Best outer diameter: {outer_d:.4f}")

    optimizer.plot()
