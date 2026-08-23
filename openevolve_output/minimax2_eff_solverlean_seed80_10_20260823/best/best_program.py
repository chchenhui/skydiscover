import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Minimize a common diameter under all-pairs unit-separation constraints."""
    # Four staggered triangular rows, with lengths 3, 4, 5, 4.
    h = np.sqrt(3.0) / 2.0
    seed = np.array([
        [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
        [-1.5, h], [-0.5, h], [0.5, h], [1.5, h],
        [-2.0, 2.0*h], [-1.0, 2.0*h], [0.0, 2.0*h],
        [1.0, 2.0*h], [2.0, 2.0*h],
        [-1.5, 3.0*h], [-0.5, 3.0*h], [0.5, 3.0*h],
        [1.5, 3.0*h],
    ], dtype=float)

    pairs = np.array([(i, j) for i in range(16) for j in range(i + 1, 16)],
                     dtype=int)

    def normalize(p):
        p = np.asarray(p, dtype=float).copy()
        p -= p.mean(axis=0)
        delta = p[pairs[:, 0]] - p[pairs[:, 1]]
        dsq = np.einsum("ij,ij->i", delta, delta)
        p /= np.sqrt(dsq.min())
        return p

    def ratio_squared(p):
        delta = p[pairs[:, 0]] - p[pairs[:, 1]]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(dsq.min() / dsq.max())

    seed = normalize(seed)
    best = seed.copy()
    best_score = ratio_squared(best)

    # SciPy is used when available; the seed itself remains a valid,
    # deterministic fallback in minimal NumPy-only environments.
    try:
        from scipy.optimize import minimize

        m = len(pairs)

        def unpack(z):
            return z[:32].reshape(16, 2), z[32]

        def objective(z):
            return z[32]

        def objective_jac(z):
            g = np.zeros(33)
            g[32] = 1.0
            return g

        def constraints(z):
            p, D = unpack(z)
            dv = p[pairs[:, 0]] - p[pairs[:, 1]]
            dsq = np.einsum("ij,ij->i", dv, dv)
            # First block: minimum-distance constraints.
            # Second block: common-diameter constraints.
            return np.concatenate((dsq - 1.0, D * D - dsq))

        def constraints_jac(z):
            p, D = unpack(z)
            dv = p[pairs[:, 0]] - p[pairs[:, 1]]
            jac = np.zeros((2 * m, 33))
            rows = np.arange(m)
            ii = pairs[:, 0]
            jj = pairs[:, 1]

            jac[rows, 2 * ii] = 2.0 * dv[:, 0]
            jac[rows, 2 * ii + 1] = 2.0 * dv[:, 1]
            jac[rows, 2 * jj] = -2.0 * dv[:, 0]
            jac[rows, 2 * jj + 1] = -2.0 * dv[:, 1]

            rows2 = rows + m
            jac[rows2, 2 * ii] = -2.0 * dv[:, 0]
            jac[rows2, 2 * ii + 1] = -2.0 * dv[:, 1]
            jac[rows2, 2 * jj] = 2.0 * dv[:, 0]
            jac[rows2, 2 * jj + 1] = 2.0 * dv[:, 1]
            jac[rows2, 32] = 2.0 * D
            return jac

        rng = np.random.default_rng(1701)
        starts = [seed]

        # Generate centered perturbations explicitly.  This removes the
        # translational component before SLSQP sees a start, while retaining
        # deterministic symmetry-breaking perturbations of the lattice.
        amplitudes = np.linspace(0.012, 0.300, 18)
        for amplitude in amplitudes:
            perturbation = rng.normal(size=seed.shape)
            perturbation -= perturbation.mean(axis=0, keepdims=True)

            # Preserve a modest amount of the lattice's radial structure in
            # every start rather than using an unconstrained random cloud.
            radial = seed / np.maximum(
                np.linalg.norm(seed, axis=1, keepdims=True), 1.0
            )
            perturbation += 0.18 * rng.normal() * radial
            starts.append(normalize(seed + amplitude * perturbation))

        cons = {"type": "ineq", "fun": constraints, "jac": constraints_jac}
        bounds = [(-5.0, 5.0)] * 32 + [(1.0, 6.0)]

        for start in starts:
            delta = start[pairs[:, 0]] - start[pairs[:, 1]]
            diameter = np.sqrt(np.einsum("ij,ij->i", delta, delta).max())
            z0 = np.concatenate((start.ravel(), [diameter]))
            result = minimize(
                objective, z0, jac=objective_jac, constraints=cons,
                method="SLSQP", bounds=bounds,
                options={"maxiter": 1400, "ftol": 3e-13, "disp": False},
            )

            # Centering preserves every distance.  Re-solving from the
            # centered output is therefore a genuine constrained polishing
            # pass, not a change to the geometric problem.
            trials = [result]
            if np.all(np.isfinite(result.x)):
                centered = result.x[:32].reshape(16, 2).copy()
                centered -= centered.mean(axis=0, keepdims=True)
                dv = centered[pairs[:, 0]] - centered[pairs[:, 1]]
                centered_diameter = np.sqrt(
                    np.einsum("ij,ij->i", dv, dv).max()
                )
                polished = minimize(
                    objective,
                    np.concatenate((centered.ravel(), [centered_diameter])),
                    jac=objective_jac,
                    constraints=cons,
                    method="SLSQP",
                    bounds=bounds,
                    options={"maxiter": 1400, "ftol": 3e-13, "disp": False},
                )
                trials.append(polished)

            # Do not rely solely on SLSQP's status flag: accept every finite
            # output that directly satisfies both all-pairs constraint blocks.
            for trial in trials:
                if not np.all(np.isfinite(trial.x)):
                    continue
                candidate = trial.x[:32].reshape(16, 2).copy()
                candidate -= candidate.mean(axis=0, keepdims=True)
                dv = candidate[pairs[:, 0]] - candidate[pairs[:, 1]]
                dsq = np.einsum("ij,ij->i", dv, dv)
                diameter = trial.x[32]
                if dsq.min() < 1.0 - 3e-8 or dsq.max() > diameter * diameter + 3e-8:
                    continue
                candidate = normalize(candidate)
                score = ratio_squared(candidate)
                if np.isfinite(score) and score > best_score:
                    best = candidate
                    best_score = score
    except Exception:
        pass

    return best