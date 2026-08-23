# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Polish a 16-point alternating-corner triangular packing at unit diameter.

    The exact lattice construction is retained as a dependency-free fallback.
    When SciPy is present, deterministic symmetry-breaking COBYLA searches
    maximize a common separation bound while every pair distance is at most one.
    """
    removed = {(2, 0), (-2, 2), (0, -2)}
    axial = [
        (q, r)
        for q in range(-2, 3)
        for r in range(-2, 3)
        if max(abs(q), abs(r), abs(q + r)) <= 2 and (q, r) not in removed
    ]
    lattice = np.array(
        [(q + 0.5 * r, 0.5 * np.sqrt(3.0) * r) for q, r in axial],
        dtype=np.float64,
    )
    lattice -= lattice[0]
    ij = np.triu_indices(16, 1)

    def normalize(p):
        delta = p[:, None, :] - p[None, :, :]
        return p / np.sqrt(np.max(np.sum(delta * delta, axis=-1)))

    def value(p):
        delta = p[:, None, :] - p[None, :, :]
        d2 = np.sum(delta * delta, axis=-1)[ij]
        return float(np.min(d2) / np.max(d2))

    best = normalize(lattice)
    best_value = value(best)

    try:
        from scipy.optimize import minimize

        def unpack(v):
            return np.vstack((np.zeros(2), v[:-1].reshape(15, 2)))

        def constraints(v):
            p = unpack(v)
            delta = p[:, None, :] - p[None, :, :]
            d2 = np.sum(delta * delta, axis=-1)[ij]
            return np.concatenate((d2 - v[-1] * v[-1], 1.0 - d2))

        rng = np.random.default_rng(314159265)
        starts = [best]
        starts.extend(
            normalize(
                np.vstack((
                    np.zeros(2),
                    best[1:] + rng.normal(0.0, scale, size=(15, 2)),
                ))
            )
            for scale in (0.025, 0.050, 0.085)
            for _ in range(3)
        )

        for start in starts:
            # Starting below the actual nearest spacing makes all lower-bound
            # constraints feasible, which is substantially more reliable for
            # COBYLA than beginning on the lattice's many active contacts.
            result = minimize(
                lambda v: -v[-1],
                np.r_[start[1:].ravel(), 0.18],
                method="COBYLA",
                constraints={"type": "ineq", "fun": constraints},
                options={
                    "maxiter": 18000,
                    "rhobeg": 0.04,
                    "tol": 2e-10,
                    "catol": 2e-9,
                },
            )
            candidate = normalize(unpack(result.x))
            candidate_value = value(candidate)
            if np.isfinite(candidate_value) and candidate_value > best_value:
                best, best_value = candidate, candidate_value
    except Exception:
        pass

    return np.asarray(best, dtype=np.float64)


# EVOLVE-BLOCK-END
