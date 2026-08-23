# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Maximize the diameter-normalized minimum squared distance using deterministic multistart SLSQP."""
    sqrt3_over_2 = np.sqrt(3.0) / 2.0
    axial_points = np.array([
        [-2,  1], [-2,  2],
        [-1, -1], [-1,  0], [-1,  1], [-1,  2],
        [ 0, -2], [ 0, -1], [ 0,  0], [ 0,  1],
        [ 1, -2], [ 1, -1], [ 1,  0], [ 1,  1],
        [ 2, -1], [ 2,  0],
    ], dtype=np.float64)

    seed = np.empty((16, 2), dtype=np.float64)
    seed[:, 0] = axial_points[:, 0] + 0.5 * axial_points[:, 1]
    seed[:, 1] = sqrt3_over_2 * axial_points[:, 1]

    ii, jj = np.triu_indices(16, 1)

    def normalize(p):
        """Center and scale p to have exact unit maximum pair distance."""
        p = p - p.mean(axis=0)
        delta = p[ii] - p[jj]
        return p / np.sqrt(np.max(np.einsum("ij,ij->i", delta, delta)))

    def quality(p):
        delta = p[ii] - p[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(dsq) / np.max(dsq))

    best = normalize(seed)
    best_quality = quality(best)

    try:
        from scipy.optimize import minimize

        def constraints(z):
            p = z[:-1].reshape(16, 2)
            d = p[ii] - p[jj]
            dsq = np.einsum("ij,ij->i", d, d)
            return np.concatenate((dsq - z[-1], 1.0 - dsq))

        def constraint_jacobian(z):
            p = z[:-1].reshape(16, 2)
            d = p[ii] - p[jj]
            m = len(ii)
            jac = np.zeros((2 * m, 33), dtype=np.float64)
            rows = np.arange(m)
            jac[rows, 2 * ii] = 2.0 * d[:, 0]
            jac[rows, 2 * ii + 1] = 2.0 * d[:, 1]
            jac[rows, 2 * jj] = -2.0 * d[:, 0]
            jac[rows, 2 * jj + 1] = -2.0 * d[:, 1]
            jac[rows, -1] = -1.0
            jac[m:] = -jac[:m]
            jac[m:, -1] = 0.0
            return jac

        # Explore both lattice-adjacent and boundary-biased contact graphs.
        # Gaussian starts tend to put too many points near the center; the
        # radial starts deliberately seed configurations with an outer shell.
        rng = np.random.default_rng(91723)
        starts = [best]
        for amplitude, count in (
            (0.010, 2),
            (0.035, 4),
            (0.080, 6),
            (0.150, 8),
            (0.240, 10),
            (0.360, 10),
        ):
            for _ in range(count):
                starts.append(normalize(
                    best + amplitude * rng.standard_normal((16, 2))
                ))

        for _ in range(12):
            starts.append(normalize(rng.standard_normal((16, 2))))

        # Random angles with stratified radii supply genuinely different
        # convex-hull populations while remaining fully deterministic.
        for shell_fraction in (0.45, 0.60, 0.75, 0.90):
            for _ in range(3):
                angles = 2.0 * np.pi * rng.random(16)
                radii = np.where(
                    rng.random(16) < shell_fraction,
                    0.72 + 0.28 * rng.random(16),
                    0.05 + 0.67 * rng.random(16),
                )
                starts.append(normalize(np.column_stack((
                    radii * np.cos(angles), radii * np.sin(angles)
                ))))

        problem = {"type": "ineq", "fun": constraints, "jac": constraint_jacobian}
        bounds = [(-0.8, 0.8)] * 32 + [(0.0, 0.2)]
        objective_gradient = np.zeros(33, dtype=np.float64)
        objective_gradient[-1] = -1.0

        # When a start improves the incumbent, probe a few nearby contact-graph
        # changes around that new incumbent.  The budget is fixed, so runtime
        # and output remain deterministic.
        refinement_budget = 12
        refinement_count = 0
        for start in starts:
            delta = start[ii] - start[jj]
            initial_t = np.min(np.einsum("ij,ij->i", delta, delta))
            z0 = np.concatenate((start.ravel(), [initial_t]))
            result = minimize(
                lambda z: -z[-1],
                z0,
                method="SLSQP",
                jac=lambda z: objective_gradient,
                bounds=bounds,
                constraints=problem,
                options={"maxiter": 1600, "ftol": 5e-14, "disp": False},
            )
            if result.x.shape == (33,) and np.all(np.isfinite(result.x)):
                candidate = normalize(result.x[:-1].reshape(16, 2))
                candidate_quality = quality(candidate)
                if candidate_quality > best_quality:
                    best, best_quality = candidate, candidate_quality
                    if refinement_count < refinement_budget:
                        amplitude = (0.018, 0.055, 0.120)[refinement_count % 3]
                        starts.append(normalize(
                            best + amplitude * rng.standard_normal((16, 2))
                        ))
                        refinement_count += 1
    except Exception:
        pass

    return best


# EVOLVE-BLOCK-END
