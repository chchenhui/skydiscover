import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Numerically tighten a triangular-lattice seed under all pair-distance constraints.

    Two unit-separated points are pinned to remove translation, rotation, and
    scale freedom.  Deterministic SLSQP restarts minimize the diameter while
    requiring every remaining pair distance to be at least one; the best
    verified result is normalized before returning.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None

    s = np.sqrt(3.0)
    base = np.array([
        [0.0, 0.0], [1.0, 0.0],
        [0.5, 0.5 * s], [-0.5, 0.5 * s], [-1.0, 0.0],
        [-0.5, -0.5 * s], [0.5, -0.5 * s],
        [1.5, 0.5 * s], [0.0, s], [-1.5, 0.5 * s],
        [-1.5, -0.5 * s], [0.0, -s], [1.5, -0.5 * s],
        [2.0, 0.0], [-1.0, s], [-1.0, -s],
    ], dtype=float)

    ii, jj = np.triu_indices(16, 1)

    def ratio(points: np.ndarray) -> float:
        delta = points[ii] - points[jj]
        d2 = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(d2) / np.max(d2))

    best = base.copy()
    best_value = ratio(best)
    if minimize is None:
        return best

    def unpack(z: np.ndarray) -> np.ndarray:
        points = base.copy()
        points[2:] = z[:-1].reshape(14, 2)
        return points

    def inequalities(z: np.ndarray) -> np.ndarray:
        points = unpack(z)
        delta = points[ii] - points[jj]
        d2 = np.einsum("ij,ij->i", delta, delta)
        diameter2 = z[-1] * z[-1]
        return np.concatenate((d2 - 1.0, diameter2 - d2))

    rng = np.random.RandomState(731)
    diameter = np.sqrt(13.0)
    for restart in range(10):
        trial = base.copy()
        if restart:
            trial[2:] += rng.normal(scale=0.025, size=(14, 2))
        initial = np.concatenate((trial[2:].ravel(), [diameter]))
        result = minimize(
            lambda z: z[-1],
            initial,
            method="SLSQP",
            bounds=[(-3.0, 3.0)] * 28 + [(1.0, 4.0)],
            constraints={"type": "ineq", "fun": inequalities},
            options={"maxiter": 1800, "ftol": 1e-12, "disp": False},
        )

        if not np.all(np.isfinite(result.x)):
            continue
        candidate = unpack(result.x)
        delta = candidate[ii] - candidate[jj]
        d2 = np.einsum("ij,ij->i", delta, delta)
        candidate /= np.sqrt(np.min(d2))
        value = ratio(candidate)
        if value > best_value:
            best = candidate
            best_value = value

    return best