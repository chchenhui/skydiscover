import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Deterministic multistart packing of 21 circles in a rectangle whose
    width plus height is two.  The returned coordinates are contained in
    one such rectangle, so its minimum circumscribing rectangle also has
    width + height <= 2.
    """
    from scipy.optimize import minimize

    n = 21
    ii, jj = np.tril_indices(n, -1)
    m = len(ii)
    reference_sum = 2.3658321334167627

    def objective(z):
        return -float(np.sum(z[2:3 * n:3]))

    def objective_jac(z):
        g = np.zeros_like(z)
        g[2:3 * n:3] = -1.0
        return g

    def square_constraints(z):
        """Return wall clearances and direct pairwise circle clearances."""
        q = z.reshape(n, 3)
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        d = np.hypot(dx, dy)
        rs = q[ii, 2] + q[jj, 2]
        return np.concatenate((
            q[:, 0] - q[:, 2],
            1.0 - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            1.0 - q[:, 1] - q[:, 2],
            d - rs,
        ))

    def square_constraints_jac(z):
        """Return the analytic Jacobian of direct clearance constraints."""
        q = z.reshape(n, 3)
        a = np.zeros((4 * n + m, 3 * n), dtype=float)
        k = np.arange(n)
        a[k, 3 * k] = 1.0
        a[k, 3 * k + 2] = -1.0
        a[n + k, 3 * k] = -1.0
        a[n + k, 3 * k + 2] = -1.0
        a[2 * n + k, 3 * k + 1] = 1.0
        a[2 * n + k, 3 * k + 2] = -1.0
        a[3 * n + k, 3 * k + 1] = -1.0
        a[3 * n + k, 3 * k + 2] = -1.0

        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1.0e-15)
        row = 4 * n + np.arange(m)
        ux = dx / d
        uy = dy / d
        a[row, 3 * ii] =  ux
        a[row, 3 * jj] = -ux
        a[row, 3 * ii + 1] =  uy
        a[row, 3 * jj + 1] = -uy
        a[row, 3 * ii + 2] = -1.0
        a[row, 3 * jj + 2] = -1.0
        return a

    def aspect_constraints(v):
        """Return wall clearances and direct pairwise clearances for width w."""
        q = v[:3 * n].reshape(n, 3)
        w = float(v[-1])
        h = 2.0 - w
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        d = np.hypot(dx, dy)
        rs = q[ii, 2] + q[jj, 2]
        return np.concatenate((
            q[:, 0] - q[:, 2],
            w - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            h - q[:, 1] - q[:, 2],
            d - rs,
        ))

    def aspect_constraints_jac(v):
        """Return the analytic Jacobian for variable-aspect clearances."""
        q = v[:3 * n].reshape(n, 3)
        a = np.zeros((4 * n + m, 3 * n + 1), dtype=float)
        k = np.arange(n)
        a[k, 3 * k] = 1.0
        a[k, 3 * k + 2] = -1.0

        a[n + k, 3 * k] = -1.0
        a[n + k, 3 * k + 2] = -1.0
        a[n + k, -1] = 1.0

        a[2 * n + k, 3 * k + 1] = 1.0
        a[2 * n + k, 3 * k + 2] = -1.0

        a[3 * n + k, 3 * k + 1] = -1.0
        a[3 * n + k, 3 * k + 2] = -1.0
        a[3 * n + k, -1] = -1.0

        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1.0e-15)
        ux = dx / d
        uy = dy / d
        row = 4 * n + np.arange(m)
        a[row, 3 * ii] =  ux
        a[row, 3 * jj] = -ux
        a[row, 3 * ii + 1] =  uy
        a[row, 3 * jj + 1] = -uy
        a[row, 3 * ii + 2] = -1.0
        a[row, 3 * jj + 2] = -1.0
        return a

    def safe_repair(q, w, h):
        """Return a uniformly shrunken, robustly feasible candidate or None."""
        q = np.asarray(q, dtype=float).reshape(n, 3).copy()
        if not np.all(np.isfinite(q)) or w <= 0.0 or h <= 0.0:
            return None
        q[:, 2] = np.maximum(q[:, 2], 1.0e-12)

        d = np.hypot(q[ii, 0] - q[jj, 0], q[ii, 1] - q[jj, 1])
        denom = q[ii, 2] + q[jj, 2]
        factors = np.concatenate((
            q[:, 0] / q[:, 2],
            (w - q[:, 0]) / q[:, 2],
            q[:, 1] / q[:, 2],
            (h - q[:, 1]) / q[:, 2],
            d / denom,
        ))
        f = float(np.min(factors))
        if not np.isfinite(f) or f <= 0.0:
            return None
        q[:, 2] *= min(1.0, f) * (1.0 - 2.0e-8)
        return q

    # A guaranteed feasible fallback.
    fallback = []
    for y in (1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0):
        for x in np.linspace(1.0 / 14.0, 13.0 / 14.0, 7):
            fallback.append((x, y, 0.02))
    best = np.asarray(fallback, dtype=float)

    square_bounds = [(0.0, 1.0), (0.0, 1.0), (1.0e-9, 0.5)] * n
    patterns = (
        ((6, 5, 5, 5), 0.000, 0.060),
        ((5, 6, 5, 5), 0.013, 0.060),
        ((5, 5, 6, 5), 0.027, 0.060),
        ((5, 5, 5, 6), 0.041, 0.060),
        ((5, 4, 4, 4, 4), 0.000, 0.055),
        ((4, 4, 5, 4, 4), 0.019, 0.055),
        ((4, 5, 4, 4, 4), 0.037, 0.055),
        ((4, 4, 4, 5, 4), 0.053, 0.055),
        ((4, 4, 4, 4, 5), 0.071, 0.055),
        ((7, 7, 7), 0.000, 0.060),
        ((7, 7, 7), 0.035, 0.060),
        ((8, 7, 6), 0.017, 0.055),
        ((6, 7, 8), 0.043, 0.055),
        ((6, 5, 5, 5), 0.031, 0.058),
        ((5, 5, 5, 6), 0.081, 0.058),
        ((5, 4, 4, 4, 4), 0.025, 0.060),
        ((4, 4, 4, 4, 5), 0.089, 0.060),
    )

    square_candidates = []
    square_cons = {
        "type": "ineq",
        "fun": square_constraints,
        "jac": square_constraints_jac,
    }

    for rows, phase, r0 in patterns:
        pts = []
        nr = len(rows)
        for row, count in enumerate(rows):
            margin = 1.0 / (2.0 * count)
            xs = np.linspace(margin, 1.0 - margin, count)
            shift = phase if (row & 1) else -phase
            xs = np.mod(xs + shift, 1.0)
            xs.sort()
            y = (row + 0.5) / nr
            pts.extend((x, y) for x in xs)

        z0 = np.array([[x, y, r0] for x, y in pts], dtype=float).ravel()
        try:
            res = minimize(
                objective, z0, jac=objective_jac, method="SLSQP",
                bounds=square_bounds, constraints=square_cons,
                options={"maxiter": 1500, "ftol": 1.0e-12, "disp": False},
            )
            raw = res.x if np.all(np.isfinite(res.x)) else z0
        except Exception:
            raw = z0

        cand = safe_repair(raw, 1.0, 1.0)
        if cand is not None:
            square_candidates.append(cand)
            if np.sum(cand[:, 2]) > np.sum(best[:, 2]):
                best = cand

    # Preserve several distinct contact graphs, rather than refining only
    # the single best square graph.
    square_candidates.sort(key=lambda q: float(np.sum(q[:, 2])), reverse=True)
    seeds = square_candidates[:7]
    if not seeds:
        seeds = [best.copy()]

    aspect_bounds = (
        [(0.0, 1.5), (0.0, 1.5), (1.0e-9, 0.5)] * n +
        [(0.55, 1.45)]
    )

    def aspect_objective(v):
        return -float(np.sum(v[2:3 * n:3]))

    def aspect_objective_jac(v):
        g = np.zeros_like(v)
        g[2:3 * n:3] = -1.0
        return g

    aspect_cons = {
        "type": "ineq",
        "fun": aspect_constraints,
        "jac": aspect_constraints_jac,
    }

    for q0 in seeds:
        for w0 in (0.80, 0.86, 0.93, 1.00, 1.07, 1.14, 1.20):
            h0 = 2.0 - w0
            start = q0.copy()
            start[:, 0] *= w0
            start[:, 1] *= h0
            # This radius scaling makes the transformed square packing
            # feasible before SLSQP starts.
            start[:, 2] *= min(w0, h0) * (1.0 - 1.0e-9)
            v0 = np.concatenate((start.ravel(), np.array([w0])))

            try:
                res = minimize(
                    aspect_objective, v0, jac=aspect_objective_jac,
                    method="SLSQP", bounds=aspect_bounds,
                    constraints=aspect_cons,
                    options={"maxiter": 1900, "ftol": 1.0e-12, "disp": False},
                )
                if not np.all(np.isfinite(res.x)):
                    continue
                w = float(res.x[-1])
                h = 2.0 - w
                cand = safe_repair(res.x[:3 * n], w, h)
                if cand is not None and np.sum(cand[:, 2]) > np.sum(best[:, 2]):
                    best = cand
            except Exception:
                pass

    # One final direct-clearance polish resolves the active contact graph
    # from the best incumbent after the aspect-ratio multistart phase.
    q0 = np.asarray(best, dtype=float).copy()
    w0 = 1.0
    h0 = 1.0
    v0 = np.concatenate((q0.ravel(), np.array([w0])))
    try:
        polish = minimize(
            aspect_objective, v0, jac=aspect_objective_jac,
            method="SLSQP", bounds=aspect_bounds,
            constraints=aspect_cons,
            options={"maxiter": 2600, "ftol": 3.0e-13, "disp": False},
        )
        if np.all(np.isfinite(polish.x)):
            wp = float(polish.x[-1])
            hp = 2.0 - wp
            cand = safe_repair(polish.x[:3 * n], wp, hp)
            if cand is not None and np.sum(cand[:, 2]) > np.sum(best[:, 2]):
                best = cand
    except Exception:
        pass

    # All retained candidates fit a rectangle with width plus height two.
    return np.asarray(best, dtype=float)


if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:, -1]):.15f}")