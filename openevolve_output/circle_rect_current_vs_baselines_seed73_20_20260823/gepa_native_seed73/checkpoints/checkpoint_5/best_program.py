# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Maximize 21 circle radii in a width ``w`` by height ``2-w`` rectangle.

    Deterministic staggered five-row multistarts seed distinct contact graphs;
    SLSQP then optimizes every center, radius, and aspect ratio using exact
    constraint derivatives.  A final common-radius scaling supplies strict
    numerical feasibility.
    """
    n = 21
    # Several related row topologies are used as deterministic multistarts.
    # They seed different contact graphs while retaining a compact layout.
    counts = (4, 4, 5, 4, 4)
    rng = np.random.default_rng(21021)

    # Variable layout: [width, x_0..x_20, y_0..y_20, r_0..r_20].
    def unpack(z):
        return z[0], z[1:1 + n], z[1 + n:1 + 2 * n], z[1 + 2 * n:]

    def constraints(z):
        w, x, y, r = unpack(z)
        h = 2.0 - w
        values = [x - r, w - x - r, y - r, h - y - r]
        pairwise = []
        for i in range(n - 1):
            dx = x[i] - x[i + 1:]
            dy = y[i] - y[i + 1:]
            pairwise.append(dx * dx + dy * dy - (r[i] + r[i + 1:]) ** 2)
        values.extend(pairwise)
        return np.concatenate(values)

    def constraint_jacobian(z):
        """Exact Jacobian in the same ordering as ``constraints``."""
        _, x, y, r = unpack(z)
        jac = np.zeros((4 * n + n * (n - 1) // 2, 1 + 3 * n))
        q = np.arange(n)
        jac[q, 1 + q] = 1.0
        jac[q, 1 + 2 * n + q] = -1.0
        jac[n + q, 0] = 1.0
        jac[n + q, 1 + q] = -1.0
        jac[n + q, 1 + 2 * n + q] = -1.0
        jac[2 * n + q, 1 + n + q] = 1.0
        jac[2 * n + q, 1 + 2 * n + q] = -1.0
        jac[3 * n + q, 0] = -1.0
        jac[3 * n + q, 1 + n + q] = -1.0
        jac[3 * n + q, 1 + 2 * n + q] = -1.0

        row = 4 * n
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx, dy = x[i] - x[j], y[i] - y[j]
                sr = r[i] + r[j]
                jac[row, 1 + i] = 2.0 * dx
                jac[row, 1 + j] = -2.0 * dx
                jac[row, 1 + n + i] = 2.0 * dy
                jac[row, 1 + n + j] = -2.0 * dy
                jac[row, 1 + 2 * n + i] = -2.0 * sr
                jac[row, 1 + 2 * n + j] = -2.0 * sr
                row += 1
        return jac

    def make_start(w, row_counts, jitter):
        """Build one compact staggered seed with a prescribed row topology."""
        r0 = 0.089 + 0.004 * rng.random(n)
        h = 2.0 - w
        x = np.empty(n)
        y = np.empty(n)
        k = 0
        spacing = min(0.185, (w - 0.10) / max(row_counts))
        # Center any number of rows, not just the original five-row seeds.
        # This permits additional six-row contact graphs below.
        row_gap = min(
            np.sqrt(3.0) * spacing * 0.5,
            (h - 0.12) / max(1, len(row_counts) - 1),
        )
        for row, count in enumerate(row_counts):
            yy = h * 0.5 + (row - 0.5 * (len(row_counts) - 1)) * row_gap
            for col in range(count):
                x[k] = w * 0.5 + (col - (count - 1) * 0.5) * spacing
                y[k] = yy
                if jitter:
                    x[k] += rng.uniform(-jitter, jitter)
                    y[k] += rng.uniform(-jitter, jitter)
                k += 1
        return np.concatenate(([w], x, y, r0))

    topologies = (
        (4, 4, 5, 4, 4),
        (4, 5, 4, 4, 4),
        (4, 4, 4, 5, 4),
        (5, 4, 4, 4, 4),
        (4, 4, 5, 3, 5),
        (5, 3, 5, 4, 4),
        # These create different boundary contacts than the nearly uniform
        # 4/5-row family and are useful escapes from its local optima.
        (3, 5, 5, 5, 3),
        (3, 4, 5, 5, 4),
        (4, 5, 5, 4, 3),
        (3, 5, 5, 3, 5),
        (3, 4, 7, 4, 3),
        # Six-row arrangements expose contact graphs unavailable to all
        # five-row starts.  Alternating 3/4 rows are especially compact.
        (3, 4, 3, 4, 3, 4),
        (4, 3, 4, 3, 4, 3),
        (3, 4, 4, 3, 4, 3),
        (3, 4, 3, 4, 4, 3),
        (4, 3, 4, 4, 3, 3),
        (3, 3, 4, 4, 4, 3),
    )
    # Retain the original starts, then add independent deterministic
    # perturbations.  Circle-packing optima are quite sensitive to the
    # initial contact graph, so extra starts improve the best-of-run result
    # without changing feasibility handling.
    starts = []
    # Include slightly more elongated boxes, which are useful for the new
    # six-row seeds while retaining the original near-square samples.
    widths = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    for w in widths:
        for topology in topologies:
            starts.append(make_start(w, topology, 0.006))

    for jitter in (0.012, 0.020, 0.030):
        for w in widths:
            for topology in topologies:
                starts.append(make_start(w, topology, jitter))

    best = None
    best_sum = -np.inf

    try:
        from scipy.optimize import minimize

        bounds = [(0.75, 1.25)] + [(-0.1, 1.35)] * (2 * n) + [(1e-5, 0.3)] * n
        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[1 + 2 * n:]),
                start,
                jac=lambda z: np.r_[0.0, np.zeros(2 * n), -np.ones(n)],
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                options={"maxiter": 1800, "ftol": 2e-12, "disp": False},
            )
            candidate = result.x if np.all(np.isfinite(result.x)) else start
            w, x, y, r = unpack(candidate)
            if 0.0 < w < 2.0 and np.all(r > 0.0):
                h = 2.0 - w
                scale = 1.0
                for i in range(n):
                    scale = min(scale, x[i] / r[i], (w - x[i]) / r[i],
                                y[i] / r[i], (h - y[i]) / r[i])
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        distance = np.hypot(x[i] - x[j], y[i] - y[j])
                        scale = min(scale, distance / (r[i] + r[j]))
                score = float(np.sum(r) * min(1.0, scale))
                if score > best_sum:
                    best_sum = score
                    best = candidate
    except Exception:
        # The construction remains valid even in minimal environments without
        # SciPy, although the optimized result is normally substantially better.
        best = starts[1]

    w, x, y, r = unpack(best)
    h = 2.0 - w

    # Scale every radius together by the smallest available clearance.  This
    # preserves all center positions and guarantees strict feasibility despite
    # small SLSQP residuals or evaluator roundoff.
    scale = 1.0
    for i in range(n):
        scale = min(scale, x[i] / r[i], (w - x[i]) / r[i],
                    y[i] / r[i], (h - y[i]) / r[i])
    for i in range(n - 1):
        for j in range(i + 1, n):
            distance = np.hypot(x[i] - x[j], y[i] - y[j])
            scale = min(scale, distance / (r[i] + r[j]))
    r = r * min(1.0, 0.999999 * scale)

    return np.column_stack((x, y, r))


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
