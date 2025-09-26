"""
MIT License

Copyright (c) 2025 Diogo Ferreira
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from typing import Callable


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
        self.n = len(self.radii)  # number of wires
        self.margin = float(margin)
        self.inner_exclusion_radius = float(inner_exclusion_radius)

        # Precompute frequently used quantities to avoid recomputing inside callbacks.
        self.r_eff = self.radii * (1.0 + self.margin)
        self.n_vars = self.n * 2 + 1
        self.coord_idx = 2 * np.arange(self.n)
        # upper triangle indices for unique wire pairs
        self.i_idx, self.j_idx = np.triu_indices(self.n, 1)
        self.pair_r_eff = self.r_eff[self.i_idx] + self.r_eff[self.j_idx]

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
        return R - (np.linalg.norm(coords, axis=1) + self.r_eff)

    def _jac_constraint_outer(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the outer boundary constraints."""
        coords, _ = self._unpack(x)
        J = np.zeros((self.n, self.n_vars))
        if self.n == 0:
            return J
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        scaled = np.divide(
            coords,
            norms,
            out=np.zeros_like(coords),
            where=norms > 0,
        )
        idx = np.arange(self.n)
        J[idx, self.coord_idx] = -scaled[:, 0]
        J[idx, self.coord_idx + 1] = -scaled[:, 1]
        J[:, -1] = 1.0
        return J

    def _constraint_pairs(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure wires do not overlap (pairwise).
        g_k(x) = ||c_i - c_j|| - (r_eff_i + r_eff_j) >= 0
        """
        coords, _ = self._unpack(x)
        diffs = coords[self.i_idx] - coords[self.j_idx]
        d_centers = np.linalg.norm(diffs, axis=1)
        return d_centers - self.pair_r_eff

    def _jac_constraint_pairs(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the pairwise distance constraints."""
        coords, _ = self._unpack(x)
        m = self.i_idx.size
        J = np.zeros((m, self.n_vars))
        if m == 0:
            return J
        diffs = coords[self.i_idx] - coords[self.j_idx]
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        grad = np.divide(
            diffs,
            norms,
            out=np.zeros_like(diffs),
            where=norms > 0,
        )
        rows = np.arange(m)
        idx_i = self.coord_idx[self.i_idx]
        idx_j = self.coord_idx[self.j_idx]
        J[rows, idx_i] = grad[:, 0]
        J[rows, idx_i + 1] = grad[:, 1]
        J[rows, idx_j] = -grad[:, 0]
        J[rows, idx_j + 1] = -grad[:, 1]
        return J

    def _constraint_inner_hole(self, x: np.ndarray) -> np.ndarray:
        """
        Prevent wires from entering the frozen core (shielded) region.
        g_i(x) = ||c_i|| - (inner_exclusion_radius + r_eff_i) >= 0
        """
        if self.inner_exclusion_radius <= 0:
            # No constraint needed; return a trivially satisfied inequality
            return np.ones(self.n)
        coords, _ = self._unpack(x)
        return np.linalg.norm(coords, axis=1) - (
            self.inner_exclusion_radius + self.r_eff
        )

    def _jac_constraint_inner_hole(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the inner-hole constraints."""
        coords, _ = self._unpack(x)
        J = np.zeros((self.n, self.n_vars))
        if self.inner_exclusion_radius <= 0 or self.n == 0:
            return J
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        scaled = np.divide(
            coords,
            norms,
            out=np.zeros_like(coords),
            where=norms > 0,
        )
        idx = np.arange(self.n)
        J[idx, self.coord_idx] = scaled[:, 0]
        J[idx, self.coord_idx + 1] = scaled[:, 1]
        return J

    def _initial_guess_spiral(self) -> np.ndarray:
        """
        Heuristic spiral-like layout for initial guess, starting outside the inner exclusion.
        """
        coords = np.zeros((self.n, 2))
        order = np.argsort(-self.radii)
        angle = 0.0
        # Start at least outside the inner hole plus the biggest wire
        max_r_eff = self.r_eff.max() if self.n else 0.0
        base = self.inner_exclusion_radius + max_r_eff
        radius = base
        step = 2 * np.pi / max(self.n, 1)

        for idx in order:
            radius += self.radii[idx] * (1.5 + self.margin)
            coords[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            angle += step
        # outer radius seed
        R_seed = radius + max_r_eff if self.n else base
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
        self,
        n_initializations: int,
        max_iterations: int = 200,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run multiple optimizations from varied initial guesses (spiral + random).

        Returns:
            best_coords, radii, best_R
        """
        rng = np.random.default_rng()
        spiral_guess = self._initial_guess_spiral()
        _, R0 = self._unpack(spiral_guess)

        initial_guesses = [spiral_guess]
        if n_initializations > 1:
            min_ring = self.inner_exclusion_radius + self.r_eff
            for _ in range(n_initializations - 1):
                coords_rand = rng.uniform(-R0, R0, size=(self.n, 2))
                # Push any point that falls inside the inner exclusion out to the rim of feasibility
                if self.n:
                    norms = np.linalg.norm(coords_rand, axis=1)
                    mask = norms < min_ring
                    if np.any(mask):
                        dirs = coords_rand[mask].copy()
                        norm_mask = norms[mask].copy()
                        tiny = norm_mask < 1e-9
                        if np.any(tiny):
                            theta = rng.uniform(0, 2 * np.pi, size=tiny.sum())
                            dirs[tiny] = np.stack(
                                (np.cos(theta), np.sin(theta)), axis=1
                            )
                            norm_mask[tiny] = 1.0
                        dirs = dirs / norm_mask[:, None]
                        coords_rand[mask] = dirs * min_ring[mask][:, None]
                x0_rand = np.concatenate([coords_rand.flatten(), [R0]])
                initial_guesses.append(x0_rand)

        results: list[tuple[np.ndarray, float, bool]] = []
        total = len(initial_guesses)
        for idx, x0 in enumerate(initial_guesses, start=1):
            results.append(self.solve(x0, max_iterations))
            if progress_cb is not None:
                progress_cb(idx, total)

        best_radius = np.inf
        best_coords = None
        for coords, R, success in results:
            if success and R < best_radius:
                best_radius = R
                best_coords = coords

        self.positions = best_coords
        self.outer_radius = best_radius
        return best_coords, self.radii, best_radius
