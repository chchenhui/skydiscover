# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Refine asymmetric center-pentagon-decagon seeds by active-contact smoothing."""
    outer_angles = np.arange(10, dtype=np.float64) * (np.pi / 5.0)
    inner_angles = np.pi / 10.0 + np.arange(5, dtype=np.float64) * (
        2.0 * np.pi / 5.0
    )
    radius = 1.0 / (2.0 * np.cos(np.pi / 10.0))
    parent = np.vstack((
        np.zeros((1, 2), dtype=np.float64),
        radius * np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
        np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
    ))

    iu, ju = np.triu_indices(16, 1)

    def normalize(x):
        x = x - x.mean(axis=0)
        q = x[iu] - x[ju]
        return x / np.sqrt(np.max(np.sum(q * q, axis=1)))

    def exact(x):
        q = x[iu] - x[ju]
        d = np.sum(q * q, axis=1)
        return np.min(d) / np.max(d)

    best = parent.copy()
    best_score = exact(best)
    rng = np.random.default_rng(16016016)

    for restart in range(10):
        if restart == 0:
            x = parent.copy()
        else:
            # Fixed, asymmetric perturbations provide distinct contact patterns.
            x = parent + rng.normal(0.0, 0.018, parent.shape)
        x = normalize(x)

        for beta, iterations in ((4.0, 220), (12.0, 260),
                                 (36.0, 320), (110.0, 360),
                                 (320.0, 420)):
            for it in range(iterations):
                q = x[iu] - x[ju]
                d = np.sum(q * q, axis=1)
                lo = -beta * d
                lo -= np.max(lo)
                wl = np.exp(lo)
                wl /= np.sum(wl)
                hi = beta * d
                hi -= np.max(hi)
                wh = np.exp(hi)
                wh /= np.sum(wh)

                soft_lo = (np.log(np.sum(np.exp(-beta * d -
                              np.max(-beta * d)))) +
                           np.max(-beta * d)) / (-beta)
                soft_hi = (np.log(np.sum(np.exp(beta * d -
                              np.max(beta * d)))) +
                           np.max(beta * d)) / beta
                grad = np.zeros_like(x)
                g = 2.0 * q * ((wl / max(soft_lo, 1e-12)) -
                               (wh / max(soft_hi, 1e-12)))[:, None]
                np.add.at(grad, iu, g)
                np.add.at(grad, ju, -g)

                # Remove translation and use a diminishing, scale-free step.
                grad -= grad.mean(axis=0)
                step = 0.020 * (1.0 - it / iterations) + 0.0008
                x = normalize(x + step * grad /
                              max(np.sqrt(np.sum(grad * grad)), 1e-12))

            candidate_score = exact(x)
            if candidate_score > best_score:
                best = x.copy()
                best_score = candidate_score

    # The analytic parent is retained unless the raw exact metric improves.
    if best_score > 0.0690983005625:
        return np.asarray(normalize(best), dtype=np.float64)
    return np.asarray(parent, dtype=np.float64)


# EVOLVE-BLOCK-END
