# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Construct a center, regular pentagon, and 18-degree-offset decagon.

    The pentagon has radius one.  The decagon is rotated by 18 degrees, so
    every outer point lies midway between two pentagon directions; its radius
    is chosen so these nearest cross-ring distances are one.
    """
    inner_angles = 2.0 * np.pi * np.arange(5, dtype=np.float64) / 5.0
    outer_angles = np.pi / 10.0 + 2.0 * np.pi * np.arange(10, dtype=np.float64) / 10.0

    # With a pi/10 nearest angular gap, r=2*cos(pi/10) makes the
    # pentagon-to-decagon contacts unit length.  A tiny common scale margin
    # avoids an accidental sub-unit distance from floating-point roundoff.
    scale = 1.0 + 1.0e-12
    outer_radius = scale * 2.0 * np.cos(np.pi / 10.0)

    inner = scale * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
    outer = outer_radius * np.column_stack(
        (np.cos(outer_angles), np.sin(outer_angles))
    )

    return np.vstack((np.array([[0.0, 0.0]], dtype=np.float64), inner, outer))


# EVOLVE-BLOCK-END
