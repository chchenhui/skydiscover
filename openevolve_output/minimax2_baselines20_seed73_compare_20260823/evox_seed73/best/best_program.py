# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Search asymmetric unit-separated packings, then minimize their diameter.

    Points zero and one fix translation, rotation, and scale.  A deterministic
    differential-evolution penalty search escapes the rigid triangular contact
    graph; COBYLA then imposes all exact pairwise separation constraints.
    """
    h = np.sqrt(3.0) / 2.0
    base = np.array([
        [0.0, 0.0], [1.0, 0.0],
        [0.5, h], [-0.5, h], [-1.0, 0.0], [-0.5, -h], [0.5, -h],
        [1.0, 2.0 * h], [-2.0, 0.0], [1.0, -2.0 * h],
        [1.5, h], [0.0, 2.0 * h], [-1.5, h], [-1.5, -h],
        [0.0, -2.0 * h], [1.5, -h],
    ], dtype=np.float64)
    pairs = np.array([(i, j) for i in range(16) for j in range(i)],
                     dtype=np.intp)

    def points_from(z: np.ndarray) -> np.ndarray:
        p = np.empty((16, 2), dtype=np.float64)
        p[0] = (0.0, 0.0)
        p[1] = (1.0, 0.0)
        p[2:] = z[:28].reshape(14, 2)
        return p

    def distances2(p: np.ndarray) -> np.ndarray:
        d = p[pairs[:, 0]] - p[pairs[:, 1]]
        return np.einsum("ij,ij->i", d, d)

    def quality(p: np.ndarray) -> float:
        q = distances2(p)
        return float(q.min() / q.max())

    best = base.copy()
    best_value = quality(best)

    try:
        from scipy.optimize import differential_evolution, minimize

        def constrained(z: np.ndarray) -> np.ndarray:
            q = distances2(points_from(z))
            return np.concatenate((q - 1.0, z[28] - q))

        def exploratory(z: np.ndarray) -> float:
            q = distances2(points_from(z))
            return float(q.max() + 400.0 * np.maximum(0.0, 1.0 - q).sum())

        k = np.arange(28, dtype=np.float64)
        starts = []
        for phase in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
            seed = base[2:].ravel().copy()
            seed += 0.22 * np.sin(1.618 * k + phase)
            seed += 0.11 * np.cos(0.731 * k + 2.0 * phase)
            starts.append(seed)

        explored = differential_evolution(
            exploratory,
            [(-2.35, 2.35)] * 28,
            strategy="best1bin",
            maxiter=260,
            popsize=10,
            tol=2e-7,
            mutation=(0.45, 0.95),
            recombination=0.78,
            seed=1618033,
            polish=False,
            updating="immediate",
            workers=1,
            x0=base[2:].ravel(),
        )

        for raw in [base[2:].ravel(), explored.x] + starts:
            candidate = points_from(raw)
            q = distances2(candidate)
            candidate[2:] *= 1.002 / np.sqrt(q.min())
            q = distances2(candidate)
            z0 = np.concatenate((candidate[2:].ravel(),
                                 [float(q.max() + 0.03)]))

            result = minimize(
                lambda z: z[28],
                z0,
                method="COBYLA",
                constraints={"type": "ineq", "fun": constrained},
                options={
                    "maxiter": 9000,
                    "rhobeg": 0.12,
                    "tol": 3e-10,
                    "catol": 2e-9,
                },
            )
            if not np.all(np.isfinite(result.x)):
                continue

            # COBYLA identifies a feasible contact structure robustly; SLSQP
            # then performs a higher-accuracy smooth constrained polish.
            polished = minimize(
                lambda z: z[28],
                result.x,
                method="SLSQP",
                constraints={"type": "ineq", "fun": constrained},
                options={
                    "maxiter": 1800,
                    "ftol": 1e-13,
                    "disp": False,
                },
            )
            if (np.all(np.isfinite(polished.x)) and
                    constrained(polished.x).min() >= -2e-7):
                result = polished

            candidate = points_from(result.x)
            q = distances2(candidate)
            if q.min() <= 0.0:
                continue
            candidate /= np.sqrt(q.min())

            value = quality(candidate)
            if value > best_value:
                best = candidate
                best_value = value
    except Exception:
        pass

    return best


# EVOLVE-BLOCK-END
