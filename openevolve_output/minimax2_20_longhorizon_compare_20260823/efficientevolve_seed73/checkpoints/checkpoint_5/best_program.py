import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Refine a six-site triangular core and decagonal boundary by an epigraph solve."""
    import numpy as np

    def ratio(points):
        delta = points[:, None, :] - points[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        distances += np.eye(len(points)) * 1.0e9
        return float(np.min(distances) / np.max(distances))

    # Six-point triangular-lattice patch, followed by the ten boundary sites.
    h = np.sqrt(3.0) / 2.0
    core = np.array(
        [
            (-0.5, 0.0),
            (0.5, 0.0),
            (-1.0, h),
            (0.0, h),
            (1.0, h),
            (0.0, -h),
        ],
        dtype=float,
    )
    angles = 2.0 * np.pi * np.arange(10, dtype=float) / 10.0
    boundary = np.column_stack((np.cos(angles), np.sin(angles)))
    seed = np.vstack((core, boundary))

    # Center and normalize the complete seed to diameter two.
    seed -= np.mean(seed, axis=0)
    differences = seed[:, None, :] - seed[None, :, :]
    diameter = np.sqrt(np.max(np.sum(differences * differences, axis=2)))
    seed *= 2.0 / diameter

    incumbent = np.array(
        [
            [0.0, 0.0],
            *(
                (1.0 / (2.0 * np.cos(np.pi / 10.0)))
                * np.column_stack(
                    (
                        np.cos(np.pi / 10.0 + 2.0 * np.pi * np.arange(5) / 5.0),
                        np.sin(np.pi / 10.0 + 2.0 * np.pi * np.arange(5) / 5.0),
                    )
                )
            ),
            *np.column_stack(
                (
                    np.cos(2.0 * np.pi * np.arange(10) / 10.0),
                    np.sin(2.0 * np.pi * np.arange(10) / 10.0),
                )
            ),
        ],
        dtype=float,
    )

    # Maximize the common pairwise separation while constraining diameter.
    # Use several deterministic starts around the triangular-lattice seed; all
    # starts retain the same six-point core/decagonal-boundary topology.
    try:
        from scipy.optimize import minimize

        pairs = np.array(
            [(i, j) for i in range(16) for j in range(i + 1, 16)],
            dtype=int,
        )
        seed_distances = np.sqrt(
            np.sum((seed[pairs[:, 0]] - seed[pairs[:, 1]]) ** 2, axis=1)
        )
        start_t = float(np.min(seed_distances))

        def objective(x):
            return -x[-1]

        def constraints(x):
            points = x[:-1].reshape(16, 2)
            t = x[-1]
            vectors = points[pairs[:, 0]] - points[pairs[:, 1]]
            squared = np.sum(vectors * vectors, axis=1)
            return np.r_[squared - t * t, 4.0 - squared]

        starts = [seed]
        rng = np.random.RandomState(1602)
        for scale in (1.0e-3, 3.0e-3, 8.0e-3):
            trial = seed + scale * rng.standard_normal(seed.shape)
            trial -= np.mean(trial, axis=0)
            trial *= 2.0 / np.sqrt(
                np.max(
                    np.sum(
                        (trial[:, None, :] - trial[None, :, :]) ** 2,
                        axis=2,
                    )
                )
            )
            starts.append(trial)

        best = None
        best_t = -np.inf
        for points0 in starts:
            distances0 = np.sqrt(
                np.sum(
                    (points0[pairs[:, 0]] - points0[pairs[:, 1]]) ** 2,
                    axis=1,
                )
            )
            x0 = np.r_[points0.ravel(), float(np.min(distances0))]

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 600, "ftol": 3.0e-12, "disp": False},
            )
            if np.isfinite(result.fun) and result.x[-1] > best_t:
                best_t = float(result.x[-1])
                best = result.x[:-1].reshape(16, 2).copy()

        if best is None:
            candidate = seed
        else:
            candidate = best
            candidate -= np.mean(candidate, axis=0)
            candidate *= 2.0 / np.sqrt(
                np.max(
                    np.sum(
                        (candidate[:, None, :] - candidate[None, :, :]) ** 2,
                        axis=2,
                    )
                )
            )
    except Exception:
        candidate = seed

    # Preserve the incumbent unless the refined topology wins strictly.
    if ratio(candidate) > ratio(incumbent):
        return candidate
    return incumbent