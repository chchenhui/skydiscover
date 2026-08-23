# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Construct eleven outer-ring points and a rotated inner pentagon analytically."""
    outer_count = 11
    inner_count = 5
    outer_angles = 2.0 * np.pi * np.arange(outer_count, dtype=float) / outer_count
    inner_angles = (
        np.pi / 55.0
        + 2.0 * np.pi * np.arange(inner_count, dtype=float) / inner_count
    )

    cosine_gap = np.cos(np.pi / 55.0)
    pentagon_side_factor = (2.0 * np.sin(np.pi / 5.0)) ** 2
    inner_radius = (
        -cosine_gap
        + np.sqrt(cosine_gap * cosine_gap + pentagon_side_factor - 1.0)
    ) / (pentagon_side_factor - 1.0)

    outer = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner = inner_radius * np.column_stack(
        (np.cos(inner_angles), np.sin(inner_angles))
    )
    return np.vstack((outer, inner))


# EVOLVE-BLOCK-END
