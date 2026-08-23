import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Deterministic constrained refinement of a compact triangular-lattice patch.

    Returns
    -------
    np.ndarray
        Array of shape (16, 2).
    """
    h = np.sqrt(3.0) / 2.0
    base = np.array(
        [
            [-1.0, -1.5 * h], [0.0, -1.5 * h], [1.0, -1.5 * h],
            [-1.5, -0.5 * h], [-0.5, -0.5 * h], [0.5, -0.5 * h],
            [1.5, -0.5 * h], [2.5, -0.5 * h],
            [-2.0, 0.5 * h], [-1.0, 0.5 * h], [0.0, 0.5 * h],
            [1.0, 0.5 * h], [2.0, 0.5 * h],
            [-0.5, 1.5 * h], [0.5, 1.5 * h], [1.5, 1.5 * h],
        ],
        dtype=float,
    )

    ii, jj = np.triu_indices(16, 1)

    def ratio_squared(p):
        delta = p[ii] - p[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(dsq) / np.max(dsq))

    best = base.copy()
    best_value = ratio_squared(best)

    try:
        from scipy.optimize import minimize
    except Exception:
        return best

    def normalize_minimum_distance(p):
        q = p - np.mean(p, axis=0, keepdims=True)
        delta = q[ii] - q[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return q / np.sqrt(np.min(dsq))

    # Fixed, deterministic multistarts; no random state is used.
    starts = [base]
    k = np.arange(32, dtype=float)
    for phase in (0.37, 1.11, 2.03):
        perturb = 0.055 * np.sin(1.731 * k + phase).reshape(16, 2)
        starts.append(normalize_minimum_distance(base + perturb))

    m = len(ii)

    for start in starts:
        delta = start[ii] - start[jj]
        initial_dsq = np.einsum("ij,ij->i", delta, delta)
        x0 = np.concatenate((start.ravel(), [float(np.max(initial_dsq))]))

        def objective(x):
            return x[-1]

        def objective_jac(x):
            g = np.zeros_like(x)
            g[-1] = 1.0
            return g

        def constraints_fun(x):
            p = x[:-1].reshape(16, 2)
            z = x[-1]
            d = p[ii] - p[jj]
            dsq = np.einsum("ij,ij->i", d, d)
            return np.concatenate((dsq - 1.0, z - dsq))

        def constraints_jac(x):
            p = x[:-1].reshape(16, 2)
            d = p[ii] - p[jj]
            jac = np.zeros((2 * m, 33), dtype=float)
            rows = np.arange(m)

            grad = 2.0 * d
            jac[rows, 2 * ii] = grad[:, 0]
            jac[rows, 2 * ii + 1] = grad[:, 1]
            jac[rows, 2 * jj] = -grad[:, 0]
            jac[rows, 2 * jj + 1] = -grad[:, 1]

            jac[m + rows, 2 * ii] = -grad[:, 0]
            jac[m + rows, 2 * ii + 1] = -grad[:, 1]
            jac[m + rows, 2 * jj] = grad[:, 0]
            jac[m + rows, 2 * jj + 1] = grad[:, 1]
            jac[m + rows, -1] = 1.0
            return jac

        result = minimize(
            objective,
            x0,
            jac=objective_jac,
            constraints={"type": "ineq", "fun": constraints_fun, "jac": constraints_jac},
            method="SLSQP",
            options={"maxiter": 1200, "ftol": 1e-12, "disp": False},
        )

        if np.all(np.isfinite(result.x)):
            candidate = result.x[:-1].reshape(16, 2)
            value = ratio_squared(candidate)
            if value > best_value:
                best = candidate
                best_value = value

    # Active-contact KKT-style polishing, followed by a complete hard-constraint
    # release.  The rigid-motion gauge is fixed by anchoring point zero and
    # rotating the farthest point from it onto the x-axis.
    try:
        from scipy.optimize import least_squares

        def polish_contacts(p):
            d = p[ii] - p[jj]
            dsq = np.einsum("ij,ij->i", d, d)
            z0 = float(np.max(dsq))

            near_min = dsq <= 1.0 + 1e-7
            near_max = dsq >= z0 - 1e-7
            active_min = np.flatnonzero(near_min)
            active_max = np.flatnonzero(near_max)

            anchor = p[0].copy()
            q = p - anchor
            far = int(np.argmax(np.einsum("ij,ij->i", q, q)))
            v = q[far]
            nv = float(np.linalg.norm(v))
            if nv > 1e-14:
                c, s = v[0] / nv, v[1] / nv
                rot = np.array([[c, s], [-s, c]])
                q = q @ rot.T
            q[0] = 0.0
            q[far, 1] = 0.0

            xstart = np.concatenate((q.ravel(), [z0]))

            def contact_residual(x):
                r = x[:-1].reshape(16, 2)
                z = x[-1]
                dd = r[ii] - r[jj]
                values = np.einsum("ij,ij->i", dd, dd)
                out = []
                if active_min.size:
                    out.extend(values[active_min] - 1.0)
                if active_max.size:
                    out.extend(values[active_max] - z)
                # Explicit gauge equations remove translation and rotation
                # null directions from the equality-model least-squares solve.
                out.extend((r[0] * 10.0).tolist())
                out.append(10.0 * r[far, 1])
                return np.asarray(out, dtype=float)

            polished = least_squares(
                contact_residual,
                xstart,
                method="trf",
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=2500,
            )
            if not np.all(np.isfinite(polished.x)):
                return p

            candidate = polished.x[:-1].reshape(16, 2)
            # Release the active-set approximation back to every pairwise
            # inequality before considering the result.
            delta = candidate[ii] - candidate[jj]
            dsq2 = np.einsum("ij,ij->i", delta, delta)
            release_x0 = np.concatenate(
                (candidate.ravel(), [float(np.max(dsq2))])
            )
            released = minimize(
                objective,
                release_x0,
                jac=objective_jac,
                constraints={
                    "type": "ineq",
                    "fun": constraints_fun,
                    "jac": constraints_jac,
                },
                method="SLSQP",
                options={"maxiter": 900, "ftol": 1e-13, "disp": False},
            )
            if np.all(np.isfinite(released.x)):
                candidate = released.x[:-1].reshape(16, 2)
            else:
                return p

            value = ratio_squared(candidate)
            return candidate if value > ratio_squared(p) else p

        polished_best = polish_contacts(best)
        if ratio_squared(polished_best) > ratio_squared(best):
            best = polished_best
    except Exception:
        pass

    return best