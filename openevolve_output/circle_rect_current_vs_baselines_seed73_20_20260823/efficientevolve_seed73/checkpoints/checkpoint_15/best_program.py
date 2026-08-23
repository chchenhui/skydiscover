# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Explore deterministic six-row staggered layouts with SQP and LP radius polishing."""
    from scipy.optimize import minimize

    n = 21
    m = n * (n - 1) // 2
    rng = np.random.default_rng(20260823)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        w = z[-1]
        h = 2.0 - w
        out = np.empty(4 * n + 2 + m)
        out[:n] = x - r
        out[n:2 * n] = w - x - r
        out[2 * n:3 * n] = y - r
        out[3 * n:4 * n] = h - y - r
        out[4 * n] = w - 1e-5
        out[4 * n + 1] = 1.99999 - w
        k = 4 * n + 2
        for i, j in pairs:
            out[k] = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                      - (r[i] + r[j]) ** 2)
            k += 1
        return out

    def con_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        jac = np.zeros((4 * n + 2 + m, 3 * n + 1))
        ii = np.arange(n)
        jac[ii, ii] = 1.0
        jac[ii, 2 * n + ii] = -1.0
        jac[n + ii, ii] = -1.0
        jac[n + ii, 2 * n + ii] = -1.0
        jac[n + ii, -1] = 1.0
        jac[2 * n + ii, n + ii] = 1.0
        jac[2 * n + ii, 2 * n + ii] = -1.0
        jac[3 * n + ii, n + ii] = -1.0
        jac[3 * n + ii, 2 * n + ii] = -1.0
        jac[3 * n + ii, -1] = -1.0
        jac[4 * n, -1] = 1.0
        jac[4 * n + 1, -1] = -1.0
        k = 4 * n + 2
        for i, j in pairs:
            dx, dy, sr = x[i] - x[j], y[i] - y[j], r[i] + r[j]
            jac[k, i], jac[k, j] = 2.0 * dx, -2.0 * dx
            jac[k, n + i], jac[k, n + j] = 2.0 * dy, -2.0 * dy
            jac[k, 2 * n + i] = jac[k, 2 * n + j] = -2.0 * sr
            k += 1
        return jac

    def make_start(u, w, variation):
        """Turn normalized centers into a strictly feasible, varied initial packing."""
        h = 2.0 - w
        x = np.clip(u[:, 0] * w, 0.015, w - 0.015)
        y = np.clip(u[:, 1] * h, 0.015, h - 0.015)
        q = np.minimum.reduce((x, w - x, y, h - y))
        for i, j in pairs:
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            q[i] = min(q[i], 0.5 * d)
            q[j] = min(q[j], 0.5 * d)
        r = q * (0.82 + variation * rng.uniform(-1.0, 1.0, n))
        r = np.maximum(r, 0.004)
        z = np.r_[x, y, r, w]
        # One scalar repair makes every deliberately perturbed seed feasible.
        s = min(1.0, np.min(con(z)[:4 * n + 2 + m] /
                            np.maximum(np.r_[r, r, r, r, np.ones(2),
                                             np.ones(m) * 0.02], 1e-12)))
        if s < 1.0:
            z[2 * n:3 * n] *= max(0.05, 0.92 * s)
        return z

    # Explore structurally distinct six-row contact topologies.  The
    # deterministic Halton offsets are deliberately small enough to preserve
    # each row graph while allowing SQP to break artificial symmetries.
    row_patterns = (
        (3, 4, 4, 3, 4, 3),
        (3, 4, 3, 4, 4, 3),
        (4, 3, 4, 3, 4, 3),
    )
    y_levels = np.array((0.08, 0.24, 0.40, 0.60, 0.76, 0.92),
                        dtype=float)

    def halton(index, base):
        value = 0.0
        factor = 1.0 / base
        while index:
            index, digit = divmod(index, base)
            value += digit * factor
            factor /= base
        return value

    def six_row_layout(counts, reflected=False, displacement=0.0,
                       sequence=1):
        points = []
        for row, count in enumerate(counts):
            xs = np.linspace(0.10, 0.90, count)
            if row % 2:
                gap = 0.80 / max(count - 1, 1)
                xs += 0.5 * gap
            xs = np.clip(xs, 0.02, 0.98)
            yy = y_levels[row]
            if reflected:
                yy = 1.0 - yy
            for col, xx in enumerate(xs):
                k = sequence + 1 + row * 7 + col
                dx = displacement * (2.0 * halton(k, 2) - 1.0)
                dy = displacement * (2.0 * halton(k, 3) - 1.0)
                points.append((np.clip(xx + dx, 0.02, 0.98),
                               np.clip(yy + dy, 0.02, 0.98)))
        return np.asarray(points, dtype=float)

    starts = []
    sequence = 1
    for counts in row_patterns:
        for reflected in (False, True):
            for displacement in (0.0, 0.025, 0.055):
                base = six_row_layout(counts, reflected, displacement,
                                      sequence)
                sequence += 23
                for w in (0.88, 0.94, 1.00, 1.06, 1.12):
                    starts.append(make_start(base, w, 0.0))

    best = None
    best_score = -np.inf
    objective_jac = np.r_[np.zeros(2 * n), -np.ones(n), 0.0]
    for z0 in starts:
        result = minimize(
            lambda z: -np.sum(z[2 * n:3 * n]),
            z0, jac=lambda z: objective_jac, method="SLSQP",
            bounds=([(0.0, 2.0)] * (2 * n) +
                    [(1e-6, 0.5)] * n + [(1e-5, 1.99999)]),
            constraints={"type": "ineq", "fun": con, "jac": con_jac},
            options={"maxiter": 1100, "ftol": 3e-12, "disp": False},
        )
        if np.all(np.isfinite(result.x)):
            z = result.x
            if np.min(con(z)) >= -2e-7 and np.sum(z[2 * n:3 * n]) > best_score:
                best, best_score = z.copy(), np.sum(z[2 * n:3 * n])

    if best is None:
        best = starts[0].copy()

    # With centers and rectangle width fixed, maximizing the radii is a
    # linear program: each radius is bounded by its wall clearance and every
    # pairwise radius sum is bounded by the center distance.
    from scipy.optimize import linprog

    x0, y0, r0, w0 = (best[:n], best[n:2 * n], best[2 * n:3 * n],
                      float(best[-1]))
    h0 = 2.0 - w0
    aub = []
    bub = []
    for i in range(n):
        row = np.zeros(n)
        row[i] = -1.0
        aub.append(row)
        bub.append(-min(x0[i], w0 - x0[i], y0[i], h0 - y0[i]))
    for i, j in pairs:
        row = np.zeros(n)
        row[i] = row[j] = 1.0
        aub.append(row)
        bub.append(np.hypot(x0[i] - x0[j], y0[i] - y0[j]))
    lp = linprog(-np.ones(n), A_ub=np.asarray(aub), b_ub=np.asarray(bub),
                 bounds=[(1e-7, 0.5)] * n, method="highs")
    if lp.success:
        polished = best.copy()
        polished[2 * n:3 * n] = lp.x
        if np.min(con(polished)) >= -2e-8:
            best = polished

    x, y, r = best[:n].copy(), best[n:2 * n].copy(), best[2 * n:3 * n].copy()
    w = float(best[-1])
    h = 2.0 - w
    scale = min(np.min(x / r), np.min((w - x) / r),
                np.min(y / r), np.min((h - y) / r), 1.0)
    for i, j in pairs:
        scale = min(scale, np.hypot(x[i] - x[j], y[i] - y[j]) / (r[i] + r[j]))
    r *= max(0.0, scale) * (1.0 - 3e-8)
    return np.column_stack((x, y, r))


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
