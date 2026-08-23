# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Return 16 points from a triangular-lattice hexagon with alternating
    outer corners removed, giving unit nearest-neighbor spacing and
    maximum pair distance sqrt(12).
    """
    # Start with the 19 triangular-lattice points satisfying
    # max(|q|, |r|, |q+r|) <= 2 in axial lattice coordinates.
    # Removing alternating extreme vertices eliminates all diameter-4
    # pairs while retaining 16 well-separated lattice points.
    removed = {(2, 0), (-2, 2), (0, -2)}
    axial = [
        (q, r)
        for q in range(-2, 3)
        for r in range(-2, 3)
        if max(abs(q), abs(r), abs(q + r)) <= 2 and (q, r) not in removed
    ]

    # Axial triangular-lattice coordinates: neighboring points are distance 1.
    points = np.array(
        [(q + 0.5 * r, 0.5 * np.sqrt(3.0) * r) for q, r in axial],
        dtype=np.float64,
    )
    return points


# EVOLVE-BLOCK-END
