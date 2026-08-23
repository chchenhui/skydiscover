# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Optimize 21 variable circles using deterministic affine-jittered starts.

    A 4,3 staggered hexagonal lattice provides a feasible topology.  Multiple
    aspect-ratio-scaled, safely jittered copies explore substantially different
    SLSQP basins while jointly optimizing centers, radii, and rectangle width.
    A final radius reduction and normalization provide numerical safety.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    n = 21
    sqrt3 = np.sqrt(3.0)
    base_r = 2.0 / (10.0 + 5.0 * sqrt3)

    # 4,3,4,3,4,3 staggered rows: exactly 21 circles.
    xs, ys = [], []
    for row in range(6):
        count = 4 if row % 2 == 0 else 3
        offset = 1.0 if row % 2 == 0 else 2.0
        for col in range(count):
            xs.append((offset + 2.0 * col) * base_r)
            ys.append(base_r + row * sqrt3 * base_r)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # Guaranteed fast fallback if scipy is unavailable.
    if minimize is None:
        r = base_r * (1.0 - 1e-7)
        return np.column_stack((xs, ys, np.full(n, r)))

    pair_i, pair_j = np.triu_indices(n, 1)

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:3 * n], z[-1]

    def inequalities(z):
        x, y, r, width = unpack(z)
        height = 2.0 - width
        distances = np.hypot(x[pair_i] - x[pair_j], y[pair_i] - y[pair_j])
        return np.concatenate((
            x - r, y - r,
            width - x - r, height - y - r,
            distances - r[pair_i] - r[pair_j],
        ))

    def inequalities_jacobian(z):
        """Exact Jacobian of wall and pairwise separation inequalities."""
        x, y, r, width = unpack(z)
        m = len(pair_i)
        jac = np.zeros((4 * n + m, 3 * n + 1))
        k = np.arange(n)

        # left and bottom walls
        jac[k, k] = 1.0
        jac[k, 2 * n + k] = -1.0
        jac[n + k, n + k] = 1.0
        jac[n + k, 2 * n + k] = -1.0

        # right and top walls
        jac[2 * n + k, k] = -1.0
        jac[2 * n + k, 2 * n + k] = -1.0
        jac[2 * n + k, -1] = 1.0
        jac[3 * n + k, n + k] = -1.0
        jac[3 * n + k, 2 * n + k] = -1.0
        jac[3 * n + k, -1] = -1.0

        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        d = np.hypot(dx, dy)
        d = np.maximum(d, 1e-12)
        qx, qy = dx / d, dy / d
        rows = 4 * n + np.arange(m)
        jac[rows, pair_i] = qx
        jac[rows, pair_j] = -qx
        jac[rows, n + pair_i] = qy
        jac[rows, n + pair_j] = -qy
        jac[rows, 2 * n + pair_i] = -1.0
        jac[rows, 2 * n + pair_j] = -1.0
        return jac

    def objective(z):
        return -np.sum(z[2 * n:3 * n])

    def objective_jacobian(z):
        """Exact gradient of the negative total-radius objective."""
        grad = np.zeros_like(z)
        grad[2 * n:3 * n] = -1.0
        return grad

    def initial_state(seed):
        """Create feasible starts spanning both lattice perturbation and aspect."""
        if seed == 0:
            return np.concatenate((
                xs.copy(), ys.copy(), np.full(n, 0.90 * base_r), [8.0 * base_r]
            ))

        rng = np.random.default_rng(seed)
        nominal_width = 8.0 * base_r
        nominal_height = 2.0 - nominal_width

        # Changing the box aspect ratio before optimization is important:
        # starts confined to one aspect ratio tend to converge to the same
        # contact graph.  The reduced radii leave room for affine scaling and
        # independent deterministic perturbations.
        width = rng.uniform(0.68, 1.20)
        sx = width / nominal_width
        sy = (2.0 - width) / nominal_height
        x = xs * sx + rng.uniform(-0.018, 0.018, n)
        y = ys * sy + rng.uniform(-0.018, 0.018, n)

        # Keep every generated start well inside its corresponding box.  This
        # clipping is only for initialization; SLSQP subsequently moves all
        # variables freely within the stated constraints.
        r0 = 0.52 * base_r
        x = np.clip(x, r0, width - r0)
        y = np.clip(y, r0, 2.0 - width - r0)
        return np.concatenate((x, y, np.full(n, r0), [width]))

    bounds = (
        [(0.0, 2.0)] * (2 * n)
        + [(1e-5, 0.5)] * n
        + [(0.50, 1.50)]
    )

    best = None
    best_value = -np.inf
    # Exact derivatives make broader deterministic basin coverage inexpensive.
    # The starts use distinct affine aspect ratios and perturbations, which is
    # important because good solutions have several incompatible contact graphs.
    for seed in range(180):
        result = minimize(
            objective,
            initial_state(seed),
            jac=objective_jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": inequalities,
                "jac": inequalities_jacobian,
            },
            options={"maxiter": 1800, "ftol": 3e-12, "disp": False},
        )
        if result.success or np.min(inequalities(result.x)) >= -2e-7:
            value = np.sum(result.x[2 * n:3 * n])
            if np.min(inequalities(result.x)) >= -2e-7 and value > best_value:
                best = result.x
                best_value = value

    if best is None:
        r = base_r * (1.0 - 1e-7)
        return np.column_stack((xs, ys, np.full(n, r)))

    x, y, r, _ = unpack(best)

    # Repair any tiny SLSQP constraint residual conservatively.  Reducing
    # every radius by 1e-6 repairs pair and wall residuals up to 2e-6.
    r = np.maximum(r - 1e-6, 1e-8)

    # The evaluator uses the minimum circumscribing rectangle.  Normalize to
    # ensure width + height is strictly below two despite numerical rounding.
    used_width = np.max(x + r) - np.min(x - r)
    used_height = np.max(y + r) - np.min(y - r)
    scale = min(1.0, (2.0 - 1e-8) / (used_width + used_height))
    return np.column_stack((x * scale, y * scale, r * scale))


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
