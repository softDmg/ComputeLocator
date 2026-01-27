"""
Monte Carlo Pi Estimation using NumPy.

Pure Python/NumPy implementation without PySpark dependencies.
Uses vectorized operations for efficient computation on single machine.
"""
import time
import math
from typing import Optional
import numpy as np


DEFAULT_NUM_SAMPLES = 100_000_000  # 100 million samples
ACTUAL_PI = math.pi


def pi_estimation(num_samples: Optional[int] = None) -> dict:
    """
    Estimate Pi using Monte Carlo method.

    Generates random points in unit square [0,1] x [0,1] and counts
    how many fall inside the unit circle (x^2 + y^2 <= 1).
    Formula: pi/4 = (points inside) / (total points)

    Args:
        num_samples: Number of random points to generate (default: 100 million)

    Returns:
        Dictionary with results:
        - execution_time: Time in seconds
        - estimated_pi: Calculated Pi value
        - actual_pi: The actual value of Pi
        - error: Absolute error from actual Pi
        - num_samples: Number of samples used
        - success: Boolean indicating success
    """
    if num_samples is None:
        num_samples = DEFAULT_NUM_SAMPLES

    try:
        start_time = time.perf_counter()

        batch_size = 100_000
        inside_circle = 0

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            x = np.random.uniform(0, 1, current_batch_size)
            y = np.random.uniform(0, 1, current_batch_size)
            inside_circle += np.sum(x ** 2 + y ** 2 <= 1)

        estimated_pi = 4.0 * inside_circle / num_samples
        execution_time = time.perf_counter() - start_time
        return {
            "execution_time": float(execution_time),
            "estimated_pi": float(estimated_pi),
            "actual_pi": float(ACTUAL_PI),
            "error": float(abs(estimated_pi - ACTUAL_PI)),
            "num_samples": int(num_samples),
            "success": True
        }

    except Exception as e:
        return {
            "execution_time": 0.0,
            "estimated_pi": 0.0,
            "actual_pi": float(ACTUAL_PI),
            "error": 0.0,
            "num_samples": 0,
            "success": False,
            "error_message": str(e)
        }
