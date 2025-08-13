import numpy as np
import pandas as pd
from typing import Any, Tuple

def rolling_slope_and_r2(series: Any, window: int) -> Tuple[pd.Series, pd.Series, float, float]:
    # Accept pd.Series or array-like; keep index if present
    if isinstance(series, pd.Series):
        idx = series.index
        y   = series.astype(float).values
    else:
        y = np.asarray(series, dtype=float)
        idx = pd.RangeIndex(len(y))

    n = len(y)
    slopes = np.full(n, np.nan)
    r2     = np.full(n, np.nan)

    if window is None or window < 2 or n < window:
        return pd.Series(slopes, index=idx), pd.Series(r2, index=idx), np.nan, np.nan

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ssx = np.sum((x - x_mean) ** 2)

    for i in range(window - 1, n):
        yw = y[i - window + 1:i + 1]
        if not np.all(np.isfinite(yw)):
            continue
        y_mean = yw.mean()
        slope = np.sum((x - x_mean) * (yw - y_mean)) / ssx
        slopes[i] = slope

        y_hat = y_mean + slope * (x - x_mean)
        sse = np.sum((yw - y_hat) ** 2)
        sst = np.sum((yw - y_mean) ** 2)
        r2[i] = 1.0 - sse / sst if sst > 0 else 0.0

    slope_series = pd.Series(slopes, index=idx)
    r2_series    = pd.Series(r2, index=idx)
    return slope_series, r2_series, slope_series.iloc[-1], r2_series.iloc[-1]