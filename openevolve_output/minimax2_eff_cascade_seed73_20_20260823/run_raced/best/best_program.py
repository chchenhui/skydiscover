import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Anneal, then maximize an epigraph distance bound with deterministic SLSQP multistarts."""
    n = 16
    rng = np.random.RandomState(42)
    ii, jj = np.triu_indices(n, 1)

    def ratio2(p):
        d = p[ii] - p[jj]
        q = np.einsum("ij,ij->i", d, d)
        return float(q.min() / q.max())

    def normalize(p):
        p = p - p[0]
        d = p[ii] - p[jj]
        mx = np.sqrt(np.einsum("ij,ij->i", d, d).max())
        return p / mx

    # A useful incumbent is retained even if scipy is not installed.
    side = np.linspace(-1.5, 1.5, 4)
    points = np.array([(x, y) for y in side for x in side], dtype=float)
    points += rng.normal(scale=0.025, size=points.shape)

    def ratio(p):
        return np.sqrt(ratio2(p))

    current = ratio(points)
    best_points = points.copy()
    best = current
    iterations = 180000

    for step in range(iterations):
        f = step / float(iterations - 1)
        sigma = 0.11 * (1.0 - f) + 0.002
        temperature = 0.004 * (1.0 - f) + 1.0e-7
        k = rng.randint(n)
        trial = points.copy()
        trial[k] += rng.normal(scale=sigma, size=2)
        value = ratio(trial)
        change = value - current
        if change > 0.0 or rng.random_sample() < np.exp(change / temperature):
            points = trial
            current = value
            if value > best:
                best = value
                best_points = trial.copy()

    best_points = normalize(best_points)
    best_value = ratio2(best_points)

    # The optional scipy stage is an epigraph formulation:
    # maximize t subject to t <= ||pi-pj||^2 <= 1 for all pairs.
    try:
        from scipy.optimize import minimize

        def make_start(p):
            p = normalize(p)
            q = np.einsum(
                "ij,ij->i", p[ii] - p[jj], p[ii] - p[jj]
            )
            # Point zero is fixed at the origin; t is strictly feasible.
            return np.r_[p[1:].ravel(), q.min() * (1.0 - 1.0e-8)]

        def unpack(x):
            p = np.zeros((n, 2), dtype=float)
            p[1:] = x[:-1].reshape(n - 1, 2)
            return p

        def constraints(x):
            p = unpack(x)
            delta = p[ii] - p[jj]
            q = np.einsum("ij,ij->i", delta, delta)
            t = x[-1]
            return np.r_[q - t, 1.0 - q]

        def constraint_jacobian(x):
            p = unpack(x)
            delta = p[ii] - p[jj]
            m = len(ii)
            jac = np.zeros((2 * m, 2 * (n - 1) + 1), dtype=float)
            for a, (u, v) in enumerate(zip(ii, jj)):
                g = 2.0 * delta[a]
                if u != 0:
                    col = 2 * (u - 1)
                    jac[a, col:col + 2] += g
                    jac[m + a, col:col + 2] -= g
                if v != 0:
                    col = 2 * (v - 1)
                    jac[a, col:col + 2] -= g
                    jac[m + a, col:col + 2] += g
                jac[a, -1] = -1.0
            return jac

        def fun(x):
            return -x[-1]

        def fun_jac(x):
            g = np.zeros_like(x)
            g[-1] = -1.0
            return g

        # A shell start provides a topology unlike the perturbed square grid.
        shell = np.zeros((n, 2), dtype=float)
        shell[0] = (0.0, 0.0)
        for k in range(5):
            a = 2.0 * np.pi * k / 5.0 + 0.13
            shell[1 + k] = 0.43 * np.array([np.cos(a), np.sin(a)])
        for k in range(10):
            a = 2.0 * np.pi * k / 10.0
            shell[6 + k] = 0.91 * np.array([np.cos(a), np.sin(a)])

        # A staggered-row start supplies a distinct near-lattice contact graph.
        staggered = np.zeros((n, 2), dtype=float)
        h = 0.72
        for row in range(4):
            offset = 0.5 * h * (row & 1)
            for col in range(4):
                staggered[4 * row + col] = (
                    (col - 1.5) * h + offset,
                    (row - 1.5) * 0.82 * h,
                )

        starts = [
            best_points,
            normalize(shell),
            normalize(staggered),
        ]
        cons = {"type": "ineq", "fun": constraints, "jac": constraint_jacobian}
        bounds = [(-2.5, 2.5)] * (2 * (n - 1)) + [(0.0, 1.0)]

        for start in starts:
            result = minimize(
                fun, make_start(start), jac=fun_jac, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": 900, "ftol": 1.0e-12, "disp": False},
            )
            candidate = normalize(unpack(result.x))
            value = ratio2(candidate)
            # Require the documented directly recomputed quality floor.
            if (
                np.isfinite(value)
                and value > 0.0707603299953
                and value > best_value
            ):
                best_value = value
                best_points = candidate
    except Exception:
        pass

    best_points -= best_points.mean(axis=0)
    return best_points