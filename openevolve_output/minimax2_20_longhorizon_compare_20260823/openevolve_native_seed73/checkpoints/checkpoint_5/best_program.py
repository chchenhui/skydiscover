# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Build a symmetric 1+5+10 seed and run deterministic symmetry-breaking
    maximin SLSQP restarts with the diameter constrained to at most one.
    """
    # A structured non-random feasible starting point:
    # one central point, five inner points, and ten outer points.
    #
    # The radii are chosen so that the central-to-inner and nearest
    # inner-to-outer distances are both one before normalization.
    inner_radius = 1.0
    outer_radius = 2.0 * np.cos(np.pi / 10.0)

    inner_angles = np.pi / 10.0 + 2.0 * np.pi * np.arange(5) / 5.0
    outer_angles = 2.0 * np.pi * np.arange(10) / 10.0

    points = np.vstack(
        [
            np.array([[0.0, 0.0]]),
            inner_radius
            * np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
            outer_radius
            * np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
        ]
    )

    pairs = np.array(
        [(i, j) for i in range(16) for j in range(i + 1, 16)],
        dtype=int,
    )

    def pair_distances(p):
        delta = p[pairs[:, 0]] - p[pairs[:, 1]]
        return np.sqrt(np.sum(delta * delta, axis=1))

    # Normalize the initial configuration to maximum pair distance one.
    points /= np.max(pair_distances(points))
    best = points.copy()
    best_ratio = np.min(pair_distances(best)) / np.max(pair_distances(best))

    # The perfectly symmetric seed can trap a local solver in the 1+5+10
    # contact graph.  Fixed-seed perturbation restarts expose nearby
    # asymmetric contact graphs while remaining fully deterministic.
    try:
        from scipy.optimize import minimize

        def unpack(x):
            return np.vstack((np.zeros((1, 2)), x[:-1].reshape(15, 2)))

        def objective(x):
            return -x[-1]

        def constraints(x):
            p = unpack(x)
            delta = p[pairs[:, 0]] - p[pairs[:, 1]]
            dist2 = np.sum(delta * delta, axis=1)
            return np.concatenate((dist2 - x[-1] * x[-1], 1.0 - dist2))

        rng = np.random.default_rng(16031991)
        starts = [points.copy()]
        # Several amplitudes are useful: tiny changes preserve the seed's
        # graph, while larger changes permit topology changes in the packing.
        for scale in (0.008, 0.018, 0.035, 0.060):
            for _ in range(3):
                trial = points + scale * rng.standard_normal(points.shape)
                trial -= trial[0]
                trial /= np.max(pair_distances(trial))
                starts.append(trial)

        for trial in starts:
            x0 = np.concatenate(
                (trial[1:].ravel(), [np.min(pair_distances(trial))])
            )
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 900, "ftol": 1e-12, "disp": False},
            )
            candidate = unpack(result.x)
            if not np.all(np.isfinite(candidate)):
                continue

            distances = pair_distances(candidate)
            candidate_ratio = np.min(distances) / np.max(distances)
            # Ratio is scale invariant, so a small SLSQP diameter tolerance
            # is harmless; final normalization below removes it exactly.
            if candidate_ratio > best_ratio:
                best_ratio = candidate_ratio
                best = candidate
    except Exception:
        pass

    # Final normalization removes tiny optimizer constraint tolerances.
    best /= np.max(pair_distances(best))
    return best


# EVOLVE-BLOCK-END
