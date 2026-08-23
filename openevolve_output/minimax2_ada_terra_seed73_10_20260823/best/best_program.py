# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Anneal compact triangular starts and apply two deterministic elitist polishing phases."""
    rng = np.random.default_rng(731)

    axial = np.array([
        [0, 0],
        [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1],
        [1, -2], [0, -2], [-1, -1],
        [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2], [1, 1],
    ], dtype=float)

    base = np.empty((16, 2), dtype=float)
    base[:, 0] = axial[:, 0] + 0.5 * axial[:, 1]
    base[:, 1] = 0.5 * np.sqrt(3.0) * axial[:, 1]
    base -= base.mean(axis=0)

    pair_i, pair_j = np.triu_indices(16, 1)

    def ratio(points: np.ndarray) -> float:
        delta = points[pair_i] - points[pair_j]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(np.sqrt(dsq.min() / dsq.max()))

    best = base.copy()
    best_score = ratio(best)

    def anneal(current: np.ndarray, steps: int, batch: int,
               initial_sigma: float, initial_temperature: float) -> None:
        """Run one batched ratio-space simulated anneal."""
        nonlocal best, best_score

        for step in range(steps):
            progress = step / (steps - 1)
            sigma = initial_sigma * (1.0 - progress) + 0.00035
            temperature = initial_temperature * (1.0 - progress) ** 2 + 1.0e-7

            candidates = np.repeat(current[None, :, :], batch, axis=0)
            moved = rng.integers(0, 16, size=batch)
            candidates[np.arange(batch), moved] += rng.normal(
                scale=sigma, size=(batch, 2)
            )
            candidates -= candidates.mean(axis=1, keepdims=True)

            delta = candidates[:, pair_i] - candidates[:, pair_j]
            dsq = np.einsum("bij,bij->bi", delta, delta)
            scores = np.sqrt(dsq.min(axis=1) / dsq.max(axis=1))

            candidate_best = int(np.argmax(scores))
            if scores[candidate_best] > best_score:
                best_score = float(scores[candidate_best])
                best = candidates[candidate_best].copy()

            weights = np.exp((scores - scores.max()) / temperature)
            current = candidates[int(rng.choice(batch, p=weights / weights.sum()))]

    # Preserve the successful compact-patch trajectory before exploring alternate
    # active contact graphs from moderately perturbed copies.
    anneal(base.copy(), 40000, 12, 0.026, 0.0018)

    for _ in range(3):
        current = base + rng.normal(scale=0.055, size=base.shape)
        current -= current.mean(axis=0)
        anneal(current, 70000, 16, 0.030, 0.0022)

    # The anneals are deliberately somewhat noisy so they can switch contact
    # graphs.  Once their best incumbent has been found, use an elitist
    # small-step search: candidate zero is exactly the current packing, so this
    # phase can never lower the retained result.  It is particularly effective
    # here because the required improvement over the historical target is tiny.
    current = best.copy()
    batch = 24
    for step in range(100000):
        u = step / 99999.0
        sigma = 0.0035 * (1.0 - u) ** 2 + 2.0e-7

        candidates = np.repeat(current[None, :, :], batch, axis=0)
        moved = rng.integers(0, 16, size=batch - 1)
        candidates[1:, moved] += rng.normal(
            scale=sigma, size=(batch - 1, 2)
        )

        # A few two-point moves permit small collective adjustments of an
        # otherwise locally jammed contact graph.
        count = 7
        moved2 = rng.integers(0, 16, size=count)
        candidates[-count:, moved2] += rng.normal(
            scale=sigma, size=(count, 2)
        )
        candidates -= candidates.mean(axis=1, keepdims=True)

        delta = candidates[:, pair_i] - candidates[:, pair_j]
        dsq = np.einsum("bij,bij->bi", delta, delta)
        scores = np.sqrt(dsq.min(axis=1) / dsq.max(axis=1))
        chosen = int(np.argmax(scores))
        current = candidates[chosen]

        if scores[chosen] > best_score:
            best_score = float(scores[chosen])
            best = current.copy()

    # Preserve the original successful polish trajectory exactly, then spend
    # additional budget on a finer elite-only search.  Candidate zero remains
    # unchanged on every iteration, so this continuation cannot reduce the
    # incumbent ratio.  The smaller initial scale is appropriate after the
    # broader first polishing pass has already identified the contact graph.
    current = best.copy()
    batch = 32
    for step in range(250000):
        u = step / 249999.0
        sigma = 0.00085 * (1.0 - u) ** 2 + 2.0e-8
        candidates = np.repeat(current[None, :, :], batch, axis=0)

        # Candidate zero is the incumbent.  Most proposals move one point;
        # several tail proposals additionally move a second point, allowing
        # cooperative releases of nearly jammed contacts.
        moved = rng.integers(0, 16, size=batch - 1)
        candidates[1:, moved] += rng.normal(
            scale=sigma, size=(batch - 1, 2)
        )
        count = 10
        moved2 = rng.integers(0, 16, size=count)
        candidates[-count:, moved2] += rng.normal(
            scale=sigma, size=(count, 2)
        )
        candidates -= candidates.mean(axis=1, keepdims=True)

        delta = candidates[:, pair_i] - candidates[:, pair_j]
        dsq = np.einsum("bij,bij->bi", delta, delta)
        scores = np.sqrt(dsq.min(axis=1) / dsq.max(axis=1))
        chosen = int(np.argmax(scores))
        current = candidates[chosen]

        if scores[chosen] > best_score:
            best_score = float(scores[chosen])
            best = current.copy()

    best -= best.mean(axis=0)
    delta = best[pair_i] - best[pair_j]
    diameter = np.sqrt(np.einsum("ij,ij->i", delta, delta).max())
    return best / diameter


# EVOLVE-BLOCK-END
