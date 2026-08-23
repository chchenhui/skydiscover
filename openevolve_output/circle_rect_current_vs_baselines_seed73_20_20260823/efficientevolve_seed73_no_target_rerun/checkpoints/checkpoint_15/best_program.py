import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Deterministic row-topology enumeration followed by variable-aspect
    constrained refinement.  Every returned candidate is explicitly repaired
    against all walls and all pairwise distance constraints.
    """
    from itertools import permutations
    from scipy.optimize import minimize

    n = 21
    ii, jj = np.tril_indices(n, -1)
    m = len(ii)

    def obj(v):
        return -float(np.sum(v[2:3 * n:3]))

    def obj_jac(v):
        g = np.zeros_like(v)
        g[2:3 * n:3] = -1.0
        return g

    def constraints_square(z):
        q = z.reshape(n, 3)
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        return np.concatenate((
            q[:, 0] - q[:, 2],
            1.0 - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            1.0 - q[:, 1] - q[:, 2],
            np.hypot(dx, dy) - q[ii, 2] - q[jj, 2],
        ))

    def jac_square(z):
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
        d = np.maximum(np.hypot(dx, dy), 1e-14)
        row = 4 * n + np.arange(m)
        a[row, 3 * ii] = dx / d
        a[row, 3 * jj] = -dx / d
        a[row, 3 * ii + 1] = dy / d
        a[row, 3 * jj + 1] = -dy / d
        a[row, 3 * ii + 2] = -1.0
        a[row, 3 * jj + 2] = -1.0
        return a

    def constraints_aspect(v):
        q = v[:-1].reshape(n, 3)
        w = float(v[-1])
        h = 2.0 - w
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        return np.concatenate((
            q[:, 0] - q[:, 2],
            w - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            h - q[:, 1] - q[:, 2],
            np.hypot(dx, dy) - q[ii, 2] - q[jj, 2],
        ))

    def jac_aspect(v):
        q = v[:-1].reshape(n, 3)
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
        d = np.maximum(np.hypot(dx, dy), 1e-14)
        row = 4 * n + np.arange(m)
        a[row, 3 * ii] = dx / d
        a[row, 3 * jj] = -dx / d
        a[row, 3 * ii + 1] = dy / d
        a[row, 3 * jj + 1] = -dy / d
        a[row, 3 * ii + 2] = -1.0
        a[row, 3 * jj + 2] = -1.0
        return a

    def repair(q, w, h):
        q = np.asarray(q, dtype=float).reshape(n, 3).copy()
        if (not np.all(np.isfinite(q))) or w <= 0.0 or h <= 0.0:
            return None
        q[:, 2] = np.maximum(q[:, 2], 1e-12)
        d = np.hypot(q[ii, 0] - q[jj, 0], q[ii, 1] - q[jj, 1])
        ratios = np.concatenate((
            q[:, 0] / q[:, 2],
            (w - q[:, 0]) / q[:, 2],
            q[:, 1] / q[:, 2],
            (h - q[:, 1]) / q[:, 2],
            d / (q[ii, 2] + q[jj, 2]),
        ))
        scale = float(np.min(ratios))
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        q[:, 2] *= min(1.0, scale) * (1.0 - 3e-8)
        return q

    # Guaranteed valid fallback.
    fallback = np.array([
        (x, y, 0.02)
        for y in (1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0)
        for x in np.linspace(1.0 / 14.0, 13.0 / 14.0, 7)
    ], dtype=float)
    best_q, best_w = fallback, 1.0
    best_value = float(np.sum(fallback[:, 2]))

    base = {
        (6, 5, 5, 5),
        (5, 6, 5, 5),
        (5, 5, 6, 5),
        (5, 5, 5, 6),
        (5, 4, 4, 4, 4),
        (4, 5, 4, 4, 4),
        (4, 4, 5, 4, 4),
        (4, 4, 4, 5, 4),
        (4, 4, 4, 4, 5),
    }
    # These are the distinct balanced row arrangements absent from the
    # original short topology list.
    base.update(set(permutations((4, 4, 4, 3, 3, 3))))
    patterns = sorted(base, key=lambda x: (len(x), x))

    square_bounds = [(0.0, 1.0), (0.0, 1.0), (1e-10, 0.5)] * n
    sqcons = {"type": "ineq", "fun": constraints_square, "jac": jac_square}
    square_seeds = []

    phases = (0.0, 0.018, 0.041)
    for rows in patterns:
        for phase in phases:
            pts = []
            nr = len(rows)
            for row, count in enumerate(rows):
                x = np.linspace(1.0 / (2 * count), 1.0 - 1.0 / (2 * count), count)
                shift = phase if (row & 1) else -phase
                x = np.clip(x + shift, 0.015, 0.985)
                y = (row + 0.5) / nr
                pts.extend((xx, y) for xx in x)
            z0 = np.array([(x, y, 0.052) for x, y in pts], dtype=float).ravel()
            try:
                res = minimize(
                    obj, z0, jac=obj_jac, method="SLSQP",
                    bounds=square_bounds, constraints=sqcons,
                    options={"maxiter": 1300, "ftol": 2e-12, "disp": False},
                )
                raw = res.x if np.all(np.isfinite(res.x)) else z0
            except Exception:
                raw = z0
            q = repair(raw, 1.0, 1.0)
            if q is not None:
                square_seeds.append(q)
                val = float(np.sum(q[:, 2]))
                if val > best_value:
                    best_q, best_w, best_value = q, 1.0, val

    square_seeds.sort(key=lambda q: float(np.sum(q[:, 2])), reverse=True)
    square_seeds = square_seeds[:12] if square_seeds else [best_q.copy()]

    abounds = [(0.0, 1.5), (0.0, 1.5), (1e-10, 0.5)] * n + [(0.55, 1.45)]
    acons = {"type": "ineq", "fun": constraints_aspect, "jac": jac_aspect}

    aspect_pool = []
    for q0 in square_seeds:
        for w0 in (0.78, 0.86, 0.94, 1.0, 1.06, 1.14, 1.22):
            h0 = 2.0 - w0
            start = q0.copy()
            start[:, 0] *= w0
            start[:, 1] *= h0
            start[:, 2] *= min(w0, h0) * (1.0 - 1e-9)
            v0 = np.r_[start.ravel(), w0]
            try:
                res = minimize(
                    obj, v0, jac=obj_jac, method="SLSQP",
                    bounds=abounds, constraints=acons,
                    options={"maxiter": 1900, "ftol": 8e-13, "disp": False},
                )
                if not np.all(np.isfinite(res.x)):
                    continue
                w = float(res.x[-1])
                q = repair(res.x[:-1], w, 2.0 - w)
                if q is None:
                    continue
                aspect_pool.append((q, w))
                val = float(np.sum(q[:, 2]))
                if val > best_value:
                    best_q, best_w, best_value = q, w, val
            except Exception:
                pass

    # Correctly polish in each candidate's own rectangle.
    aspect_pool.append((best_q.copy(), best_w))
    aspect_pool.sort(key=lambda qw: float(np.sum(qw[0][:, 2])), reverse=True)
    for q0, w0 in aspect_pool[:6]:
        v0 = np.r_[q0.ravel(), w0]
        try:
            res = minimize(
                obj, v0, jac=obj_jac, method="SLSQP",
                bounds=abounds, constraints=acons,
                options={"maxiter": 2600, "ftol": 3e-13, "disp": False},
            )
            if np.all(np.isfinite(res.x)):
                w = float(res.x[-1])
                q = repair(res.x[:-1], w, 2.0 - w)
                if q is not None and float(np.sum(q[:, 2])) > best_value:
                    best_q, best_w = q, w
                    best_value = float(np.sum(q[:, 2]))
        except Exception:
            pass

    return np.asarray(best_q, dtype=float)


if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:, 2]):.15f}")