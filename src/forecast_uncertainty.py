"""Uncertainty bands for a point forecast, via bootstrap-resampled CV residuals.

Uniform across every model in the forecasting suite: resample the model's
own out-of-sample 1-step cross-validation residuals, scale by ``sqrt(h)``
for horizon step ``h`` (a standard random-walk-style assumption that
forecast uncertainty grows with the square root of the horizon), add to the
point forecast, and take the 10th/50th/90th percentile across resamples.
This is a simplification -- not a fitted property of any specific model --
and is documented as such rather than presented as a precise interval.
"""

from __future__ import annotations

import numpy as np


def bootstrap_forecast_bands(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(p10, p50, p90)`` arrays, one value per horizon step.

    Args:
        point_forecast: length-``horizon`` array of point forecasts.
        residuals: 1-step-ahead out-of-sample residuals from cross-validation
            (``src.forecast_evaluation.ModelCVResult.residuals``).
        n_boot: number of bootstrap resamples.
        seed: RNG seed, fixed by default for reproducibility.
    """
    point_forecast = np.asarray(point_forecast, dtype=float)
    horizon = len(point_forecast)

    if len(residuals) < 2:
        # Not enough residual history for a meaningful spread -- return the point
        # forecast itself as a degenerate (zero-width) band rather than fabricating one.
        return point_forecast.copy(), point_forecast.copy(), point_forecast.copy()

    rng = np.random.default_rng(seed)
    residuals = np.asarray(residuals, dtype=float)
    p10 = np.empty(horizon)
    p50 = np.empty(horizon)
    p90 = np.empty(horizon)

    for h in range(1, horizon + 1):
        scale = np.sqrt(h)
        resampled = rng.choice(residuals, size=n_boot, replace=True) * scale
        simulated = point_forecast[h - 1] + resampled
        p10[h - 1], p50[h - 1], p90[h - 1] = np.percentile(simulated, [10, 50, 90])

    return p10, p50, p90
