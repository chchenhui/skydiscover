# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Optimize seeded circle packings, then refine centers, radii, and rectangle aspect ratio."""
    from scipy.optimize import minimize

    n = 21
    ii, jj = np.tril_indices(n, -1)
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-8, 0.5)] * n

    def objective(z):
        return -float(np.sum(z[2::3]))

    def constraints(z):
        q = z.reshape(n, 3)
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        rs = q[ii, 2] + q[jj, 2]
        return np.concatenate((
            q[:, 0] - q[:, 2],
            1.0 - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            1.0 - q[:, 1] - q[:, 2],
            dx * dx + dy * dy - rs * rs,
        ))

    def repaired(q):
        q = np.asarray(q, dtype=float).reshape(n, 3).copy()
        q[:, :2] = np.clip(q[:, :2], 0.0, 1.0)
        q[:, 2] = np.maximum(q[:, 2], 1e-10)
        scale = min(
            1.0,
            float(np.min(q[:, 0] / q[:, 2])),
            float(np.min((1.0 - q[:, 0]) / q[:, 2])),
            float(np.min(q[:, 1] / q[:, 2])),
            float(np.min((1.0 - q[:, 1]) / q[:, 2])),
        )
        d = np.hypot(q[ii, 0] - q[jj, 0], q[ii, 1] - q[jj, 1])
        rs = q[ii, 2] + q[jj, 2]
        scale = min(scale, float(np.min(d / rs)))
        q[:, 2] *= max(0.0, min(1.0, scale * (1.0 - 8e-8)))
        return q

    best = None

    # These seeds deliberately cover several distinct contact topologies:
    # four rows, five rows, alternating row counts, and staggered centers.
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

    for rows, phase, r0 in patterns:
        nr = len(rows)
        pts = []
        for row, count in enumerate(rows):
            # Keep all starts comfortably feasible while varying the
            # horizontal phase between neighboring rows.
            margin = 1.0 / (2.0 * count)
            xs = np.linspace(margin, 1.0 - margin, count)
            shift = phase * (1.0 if row & 1 else -1.0)
            xs = np.mod(xs + shift, 1.0)
            xs.sort()
            y = (row + 0.5) / nr
            pts.extend((x, y) for x in xs)

        z0 = np.array([[x, y, r0] for x, y in pts], dtype=float).ravel()
        try:
            result = minimize(
                objective, z0, method="SLSQP", bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 1300, "ftol": 1e-12, "disp": False},
            )
            candidate = repaired(result.x if np.all(np.isfinite(result.x)) else z0)
        except Exception:
            candidate = repaired(z0)

        if best is None or np.sum(candidate[:, 2]) > np.sum(best[:, 2]):
            best = candidate

    # Final aspect-ratio refinement.  The original search uses a unit square;
    # this pass allows width=w and height=2-w while retaining circular radii.
    if best is not None:
        q0 = np.asarray(best, dtype=float).copy()
        v0 = np.concatenate((q0.ravel(), np.array([1.0])))

        def aspect_objective(v):
            return -float(np.sum(v[2:3*n:3]))

        def aspect_constraints(v):
            q = v[:3*n].reshape(n, 3)
            w = float(v[-1])
            h = 2.0 - w

            dx = q[ii, 0] - q[jj, 0]
            dy = q[ii, 1] - q[jj, 1]
            rs = q[ii, 2] + q[jj, 2]

            return np.concatenate((
                q[:, 0] - q[:, 2],
                w - q[:, 0] - q[:, 2],
                q[:, 1] - q[:, 2],
                h - q[:, 1] - q[:, 2],
                dx * dx + dy * dy - rs * rs,
            ))

        aspect_bounds = (
            [(0.0, 1.5), (0.0, 1.5), (1e-8, 0.5)] * n
            + [(0.5, 1.5)]
        )

        try:
            ar = minimize(
                aspect_objective,
                v0,
                method="SLSQP",
                bounds=aspect_bounds,
                constraints={"type": "ineq", "fun": aspect_constraints},
                options={"maxiter": 1800, "ftol": 1e-12, "disp": False},
            )
            if np.all(np.isfinite(ar.x)):
                cand = ar.x[:3*n].reshape(n, 3).copy()
                w = float(ar.x[-1])
                h = 2.0 - w

                # Uniformly shrink radii by the smallest required factor.
                cand[:, 2] = np.maximum(cand[:, 2], 1e-12)
                factors = [
                    np.min(cand[:, 0] / cand[:, 2]),
                    np.min((w - cand[:, 0]) / cand[:, 2]),
                    np.min(cand[:, 1] / cand[:, 2]),
                    np.min((h - cand[:, 1]) / cand[:, 2]),
                ]
                d = np.hypot(
                    cand[ii, 0] - cand[jj, 0],
                    cand[ii, 1] - cand[jj, 1],
                )
                factors.append(np.min(d / (cand[ii, 2] + cand[jj, 2])))
                cand[:, 2] *= max(0.0, min(1.0, float(min(factors))) * (1.0 - 2e-8))

                if np.sum(cand[:, 2]) > np.sum(best[:, 2]):
                    best = cand
        except Exception:
            pass

    return best


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
