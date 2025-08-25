"""
MIT License

Copyright (c) 2025 Diogo Ferreira
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed


class WireBundleOptimizer:
    def __init__(
        self, radii: list[float], margin: float, inner_exclusion_radius: float = 0.0
    ) -> None:
        """
        Initialize the optimizer with the given wire radii.

        Parameters:
            radii (list[float]): List of wire radii (mm).
            margin (float): Fractional margin added to each wire radius (e.g., 0.02 for +2%).
            inner_exclusion_radius (float): Frozen core radius (mm). New wires must lie outside
                                            this radius plus their own effective radius.
        """
        self.radii = np.array(radii, dtype=float)
        self.n = len(radii)  # number of wires
        self.margin = float(margin)
        self.inner_exclusion_radius = float(inner_exclusion_radius)

        # upper triangle indices for unique wire pairs
        self.i_idx, self.j_idx = np.triu_indices(self.n, 1)

        self.positions = np.zeros((self.n, 2))  # Final wire positions
        self.outer_radius = 0.0  # Final bundle radius

    def _unpack(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Unpack optimization vector into coordinates and outer radius."""
        coords = x[:-1].reshape(self.n, 2)
        R = x[-1]
        return coords, R

    def _objective(self, x: np.ndarray) -> float:
        """Objective: minimize the outer radius."""
        return x[-1]

    def _grad_objective(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the objective function."""
        g = np.zeros_like(x)
        g[-1] = 1.0
        return g

    def _constraint_outer(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure each wire lies entirely within the outer radius.
        g_i(x) = R - (||c_i|| + r_eff_i) >= 0
        """
        coords, R = self._unpack(x)
        r_eff = self.radii * (1.0 + self.margin)
        return R - (np.linalg.norm(coords, axis=1) + r_eff)

    def _jac_constraint_outer(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the outer boundary constraints."""
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
        g_k(x) = ||c_i - c_j|| - (r_eff_i + r_eff_j) >= 0
        """
        coords, _ = self._unpack(x)
        diffs = coords[self.i_idx] - coords[self.j_idx]
        d_centers = np.linalg.norm(diffs, axis=1)
        r_eff = self.radii * (1.0 + self.margin)
        min_allowed = r_eff[self.i_idx] + r_eff[self.j_idx]
        return d_centers - min_allowed

    def _jac_constraint_pairs(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the pairwise distance constraints."""
        coords, _ = self._unpack(x)
        m = len(self.i_idx)
        n_vars = x.size
        J = np.zeros((m, n_vars))

        for k, (i, j) in enumerate(zip(self.i_idx, self.j_idx)):
            diff = coords[i] - coords[j]
            norm = np.linalg.norm(diff)
            if norm > 0:
                grad = diff / norm
                J[k, 2 * i : 2 * i + 2] = grad
                J[k, 2 * j : 2 * j + 2] = -grad
        return J

    def _constraint_inner_hole(self, x: np.ndarray) -> np.ndarray:
        """
        Prevent wires from entering the frozen core (shielded) region.
        g_i(x) = ||c_i|| - (inner_exclusion_radius + r_eff_i) >= 0
        """
        if self.inner_exclusion_radius <= 0:
            # No constraint needed; return a trivially satisfied inequality
            coords, _ = self._unpack(x)
            return np.ones(coords.shape[0])
        coords, _ = self._unpack(x)
        r_eff = self.radii * (1.0 + self.margin)
        return np.linalg.norm(coords, axis=1) - (self.inner_exclusion_radius + r_eff)

    def _jac_constraint_inner_hole(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the inner-hole constraints."""
        coords, _ = self._unpack(x)
        n_vars = x.size
        J = np.zeros((self.n, n_vars))
        if self.inner_exclusion_radius <= 0:
            return J
        for i in range(self.n):
            xi = coords[i]
            norm = np.linalg.norm(xi)
            if norm > 0:
                J[i, 2 * i : 2 * i + 2] = xi / norm
            # no dependence on R
        return J

    def _initial_guess_spiral(self) -> np.ndarray:
        """
        Heuristic spiral-like layout for initial guess, starting outside the inner exclusion.
        """
        coords = np.zeros((self.n, 2))
        order = np.argsort(-self.radii)
        angle = 0.0
        # Start at least outside the inner hole plus the biggest wire
        base = self.inner_exclusion_radius + (
            self.radii.max() * (1.0 + self.margin) if self.n else 0.0
        )
        radius = base
        step = 2 * np.pi / max(self.n, 1)

        for idx in order:
            radius += self.radii[idx] * (1.5 + self.margin)
            coords[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            angle += step
        # outer radius seed
        R_seed = radius + self.radii.max() * (1.0 + self.margin) if self.n else base
        return np.concatenate([coords.flatten(), [R_seed]])

    def solve(
        self, x0: np.ndarray | None = None, max_iterations: int = 200
    ) -> tuple[np.ndarray, float, bool]:
        """
        Solve the optimization from a given initial guess.

        Returns:
            (coords, outer_radius, success)
        """
        if x0 is None:
            x0 = self._initial_guess_spiral()

        cons = [
            {
                "type": "ineq",
                "fun": self._constraint_outer,
                "jac": self._jac_constraint_outer,
            },
            {
                "type": "ineq",
                "fun": self._constraint_pairs,
                "jac": self._jac_constraint_pairs,
            },
        ]
        # Add inner-hole constraint only if needed
        if self.inner_exclusion_radius > 0:
            cons.append(
                {
                    "type": "ineq",
                    "fun": self._constraint_inner_hole,
                    "jac": self._jac_constraint_inner_hole,
                }
            )

        res = minimize(
            fun=self._objective,
            x0=x0,
            method="SLSQP",
            jac=self._grad_objective,
            constraints=cons,
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )

        coords, R = self._unpack(res.x)
        return coords, R, bool(res.success)

    def solve_multi(
        self, n_initializations: int, max_iterations: int = 200, n_jobs: int = -1
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run multiple parallel optimizations from varied initial guesses (spiral + random).

        Returns:
            best_coords, radii, best_R
        """
        rng = np.random.default_rng()
        spiral_guess = self._initial_guess_spiral()
        _, R0 = self._unpack(spiral_guess)

        initial_guesses = [spiral_guess]
        if n_initializations > 1:
            r_eff = self.radii * (1.0 + self.margin)
            min_ring = (
                self.inner_exclusion_radius + r_eff
            )  # per-wire minimum center distance from origin
            for _ in range(n_initializations - 1):
                coords_rand = rng.uniform(-R0, R0, size=(self.n, 2))
                # Push any point that falls inside the inner exclusion out to the rim of feasibility
                for i in range(self.n):
                    v = coords_rand[i]
                    norm = np.linalg.norm(v)
                    target = float(min_ring[i])
                    if norm < target:
                        # pick a random angle if norm is ~0
                        theta = (
                            rng.uniform(0, 2 * np.pi)
                            if norm < 1e-9
                            else np.arctan2(v[1], v[0])
                        )
                        coords_rand[i] = (
                            np.array([np.cos(theta), np.sin(theta)]) * target
                        )
                x0_rand = np.concatenate([coords_rand.flatten(), [R0]])
                initial_guesses.append(x0_rand)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.solve)(x0, max_iterations) for x0 in initial_guesses
        )

        best_radius = np.inf
        best_coords = None
        for coords, R, success in results:
            if success and R < best_radius:
                best_radius = R
                best_coords = coords

        self.positions = best_coords
        self.outer_radius = best_radius
        return best_coords, self.radii, best_radius
