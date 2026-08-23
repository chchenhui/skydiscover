import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Search reflected, rotated, and sheared free-hull triangular lattices with epigraph and exact polishing."""
    import numpy as np

    def ratio(points):
        delta = points[:, None, :] - points[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        distances += np.eye(len(points)) * 1.0e9
        return float(np.min(distances) / np.max(distances))

    # Free-hull 4-by-4 staggered triangular-lattice seed.  No point is
    # prescribed as a center and no hull radius or hull vertex count is fixed.
    h = np.sqrt(3.0) / 2.0
    seed = np.array(
        [(float(i) + 0.5 * (j & 1), h * float(j))
         for j in range(4) for i in range(4)],
        dtype=float,
    )

    def normalize(points):
        points = np.asarray(points, dtype=np.float64).copy()
        points -= np.mean(points, axis=0)
        delta = points[:, None, :] - points[None, :, :]
        diameter = np.sqrt(np.max(np.sum(delta * delta, axis=2)))
        return 2.0 * points / max(diameter, 1.0e-15)

    seed = normalize(seed)

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

        # Deterministic structural variants: reflections, rotations, and
        # shears preserve the triangular-lattice topology while exposing
        # noncircular hulls to the free epigraph solve.
        starts = []
        variants = (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[-1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        for angle in (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            for reflection in variants:
                for shear in (0.0, 0.08, -0.08):
                    transform = rotation @ reflection @ np.array(
                        [[1.0, shear], [0.0, 1.0]]
                    )
                    trial = seed @ transform.T
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

        # Small deterministic perturbations help break accidental contacts
        # without introducing stochastic or non-finite coordinates.
        rng = np.random.RandomState(1602)
        for scale in (1.0e-3, 3.0e-3):
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

            # Exact directional polishing: maximize the literal minimum
            # pair distance while retaining diameter two.  The epigraph
            # constraints above provide a feasible starting point, and this
            # final pass removes residual SLSQP tolerance loss.
            def exact_objective(x):
                points = x.reshape(16, 2)
                vectors = points[pairs[:, 0]] - points[pairs[:, 1]]
                squared = np.sum(vectors * vectors, axis=1)
                return -float(np.min(squared))

            polish = minimize(
                exact_objective,
                candidate.ravel(),
                method="SLSQP",
                constraints={
                    "type": "ineq",
                    "fun": lambda x: 4.0
                    - np.sum(
                        (
                            x.reshape(16, 2)[pairs[:, 0]]
                            - x.reshape(16, 2)[pairs[:, 1]]
                        )
                        ** 2,
                        axis=1,
                    ),
                },
                options={"maxiter": 250, "ftol": 1.0e-13, "disp": False},
            )
            if np.all(np.isfinite(polish.x)):
                polished = polish.x.reshape(16, 2)
                polished -= np.mean(polished, axis=0)
                pd = polished[:, None, :] - polished[None, :, :]
                polished *= 2.0 / np.sqrt(np.max(np.sum(pd * pd, axis=2)))
                if ratio(polished) >= ratio(candidate):
                    candidate = polished
    except Exception:
        candidate = seed

    # Preserve the incumbent unless the refined topology wins strictly.
    if ratio(candidate) > ratio(incumbent):
        return candidate
    return incumbent