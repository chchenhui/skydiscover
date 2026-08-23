import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Deterministic diameter-constrained multistart packing for 16 planar points."""
    n = 16
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def pair_distances(points):
        return np.asarray(
            [np.dot(points[i] - points[j], points[i] - points[j])
             for i, j in pairs],
            dtype=float,
        )

    def normalize(points):
        points = np.asarray(points, dtype=float).copy()
        points -= points[0]
        d2 = pair_distances(points)
        diam = np.sqrt(max(float(d2.max()), 1e-30))
        return points / diam

    def score(points):
        d2 = pair_distances(points)
        return float(d2.min() / d2.max())

    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    best = normalize(grid)
    best_score = score(best)

    seeds = [grid]

    staggered = np.array(
        [[col + 0.5 * (row & 1), row * np.sqrt(3.0) / 2.0]
         for row in range(4) for col in range(4)],
        dtype=float,
    )
    seeds.append(staggered)
    seeds.append(grid @ np.array([[1.0, 0.18], [0.0, 1.0]]))
    seeds.append(staggered @ np.array([[0.92, -0.21], [0.12, 1.0]]))

    # Center, five-site inner ring, and ten-site outer ring.
    ring_1_5_10 = [[0.0, 0.0]]
    for k in range(5):
        a = 2.0 * np.pi * k / 5.0
        ring_1_5_10.append([0.36 * np.cos(a), 0.36 * np.sin(a)])
    for k in range(10):
        a = 2.0 * np.pi * (k + 0.23) / 10.0
        ring_1_5_10.append([0.78 * np.cos(a), 0.78 * np.sin(a)])
    seeds.append(np.asarray(ring_1_5_10, dtype=float))

    # Three offset rings, with no privileged center point.
    ring_3_6_7 = []
    for count, radius, phase in (
        (3, 0.23, 0.0),
        (6, 0.52, 0.5),
        (7, 0.82, 0.17),
    ):
        for k in range(count):
            a = 2.0 * np.pi * (k + phase) / count
            ring_3_6_7.append([radius * np.cos(a), radius * np.sin(a)])
    seeds.append(np.asarray(ring_3_6_7, dtype=float))

    # Center and first hexagonal shell, followed by nine second-shell positions.
    axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            shell = max(abs(q), abs(r), abs(q + r))
            if shell <= 2:
                angle = np.arctan2(np.sqrt(3.0) * r / 2.0, q + r / 2.0)
                axial.append((shell, angle, q, r))
    axial.sort(key=lambda x: (x[0], x[1]))
    hex_points = []
    for _, _, q, r in axial[:16]:
        hex_points.append([q + 0.5 * r, np.sqrt(3.0) * r / 2.0])
    seeds.append(np.asarray(hex_points, dtype=float))
    seeds.append(np.asarray(hex_points, dtype=float) @
                 np.array([[0.96, 0.16], [-0.08, 1.03]]))

    try:
        from scipy.optimize import minimize

        def constraints_value(z):
            p = z[:-1].reshape(n, 2)
            t = z[-1]
            out = np.empty(2 * len(pairs), dtype=float)
            for k, (i, j) in enumerate(pairs):
                delta = p[i] - p[j]
                d = np.dot(delta, delta)
                out[2 * k] = d - t
                out[2 * k + 1] = 1.0 - d
            return out

        def constraints_jacobian(z):
            p = z[:-1].reshape(n, 2)
            jac = np.zeros((2 * len(pairs), 2 * n + 1), dtype=float)
            for k, (i, j) in enumerate(pairs):
                delta = 2.0 * (p[i] - p[j])
                a = 2 * k
                jac[a, 2 * i:2 * i + 2] = delta
                jac[a, 2 * j:2 * j + 2] = -delta
                jac[a, -1] = -1.0
                jac[a + 1, 2 * i:2 * i + 2] = -delta
                jac[a + 1, 2 * j:2 * j + 2] = delta
            return jac

        constraint = {
            "type": "ineq",
            "fun": constraints_value,
            "jac": constraints_jacobian,
        }
        bounds = (
            [(0.0, 0.0), (0.0, 0.0)] +
            [(-2.0, 2.0)] * (2 * n - 2) +
            [(1e-10, 1.0)]
        )

        for seed in seeds:
            p0 = normalize(seed)
            d0 = pair_distances(p0)
            z0 = np.r_[p0.ravel(), 0.75 * d0.min()]
            result = minimize(
                lambda z: -z[-1],
                z0,
                jac=lambda z: np.r_[np.zeros(2 * n), -1.0],
                method="SLSQP",
                bounds=bounds,
                constraints=constraint,
                options={"ftol": 2e-11, "maxiter": 1600, "disp": False},
            )
            if np.all(np.isfinite(result.x)):
                candidate = result.x[:-1].reshape(n, 2)
                candidate = normalize(candidate)
                candidate_score = score(candidate)
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score

        # Active-contact face refinement: repeatedly expose near-active
        # minimum/diameter contacts, then warm-start SQP on that face.
        refined = best.copy()
        for threshold in (1e-4, 1e-5, 1e-6):
            d2 = pair_distances(refined)
            dmin = float(d2.min())
            dmax = float(d2.max())
            active = [
                pair for pair, d in zip(pairs, d2)
                if d - dmin <= threshold or dmax - d <= threshold
            ]
            active_set = set(active)
            ordered_pairs = active + [
                pair for pair in pairs if pair not in active_set
            ]

            # Pin one coordinate of a diameter endpoint to remove rotational
            # drift while retaining the already pinned translation.
            diameter_pair = pairs[int(np.argmax(d2))]
            endpoint = diameter_pair[1] if diameter_pair[0] == 0 else diameter_pair[0]
            coordinate = int(np.argmax(np.abs(refined[endpoint])))
            fixed_index = 2 * endpoint + coordinate
            fixed_value = float(refined[endpoint, coordinate])

            def face_values(z):
                p = z[:-1].reshape(n, 2)
                t = z[-1]
                out = np.empty(2 * len(ordered_pairs), dtype=float)
                for k, (i, j) in enumerate(ordered_pairs):
                    delta = p[i] - p[j]
                    d = np.dot(delta, delta)
                    out[2 * k] = d - t
                    out[2 * k + 1] = 1.0 - d
                return out

            def face_jacobian(z):
                p = z[:-1].reshape(n, 2)
                jac = np.zeros((2 * len(ordered_pairs), 2 * n + 1))
                for k, (i, j) in enumerate(ordered_pairs):
                    delta = 2.0 * (p[i] - p[j])
                    row = 2 * k
                    jac[row, 2 * i:2 * i + 2] = delta
                    jac[row, 2 * j:2 * j + 2] = -delta
                    jac[row, -1] = -1.0
                    jac[row + 1, 2 * i:2 * i + 2] = -delta
                    jac[row + 1, 2 * j:2 * j + 2] = delta
                return jac

            face_bounds = list(bounds)
            face_bounds[fixed_index] = (fixed_value, fixed_value)
            z0 = np.r_[refined.ravel(), 0.98 * dmin]
            face_result = minimize(
                lambda z: -z[-1],
                z0,
                jac=lambda z: np.r_[np.zeros(2 * n), -1.0],
                method="SLSQP",
                bounds=face_bounds,
                constraints={
                    "type": "ineq",
                    "fun": face_values,
                    "jac": face_jacobian,
                },
                options={"ftol": 2e-12, "maxiter": 1800, "disp": False},
            )
            if np.all(np.isfinite(face_result.x)):
                candidate = normalize(face_result.x[:-1].reshape(n, 2))
                candidate_score = score(candidate)
                if candidate_score > score(refined) and candidate_score > best_score:
                    refined = candidate
                    best = candidate.copy()
                    best_score = candidate_score

    except Exception:
        pass

    return np.asarray(best, dtype=float)