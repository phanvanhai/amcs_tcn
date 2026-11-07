"""CSI denoising pipeline from the AMCS-TCN paper (Sect. "CSI processing"):
low-pass filter -> Hamming filter -> Kalman filter, applied in series along the time axis
of every (sample, antenna*subcarrier) series.

The paper does not give filter parameters; the defaults in config.PREPROCESS are our own choice
(UT-HAR: ~13% of the signal power lies above 0.2 x Nyquist, mostly high-frequency jitter).

All functions take x of shape (N, T, F) and filter along axis 1 (time). Each sample is filtered
independently, so applying the same pipeline to train / val / test does not leak information.
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfiltfilt

FILTER_ORDER = ('lowpass', 'hamming', 'kalman')


def lowpass(x, cutoff=0.2, order=4):
    """Zero-phase Butterworth low-pass. cutoff is normalized to Nyquist (0 < cutoff < 1)."""
    if not 0 < cutoff < 1:
        raise ValueError(f"lowpass cutoff must be in (0, 1) (fraction of Nyquist), got {cutoff}")
    sos = butter(order, cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, x, axis=1)


def hamming(x, window=5):
    """Smoothing FIR filter whose taps are a normalized Hamming window (centered, no phase shift)."""
    if window < 2:
        return x
    w = np.hamming(window)
    return convolve1d(x, w / w.sum(), axis=1, mode='reflect')


def kalman(x, q_ratio=1.0):
    """1-D random-walk Kalman filter run forward in time, vectorized over all series.

    Measurement noise R is estimated per series as var(first difference) / 2, process noise
    Q = q_ratio * R. Smaller q_ratio -> stronger smoothing (and more lag).
    """
    R = np.var(np.diff(x, axis=1), axis=1) / 2 + 1e-8  # (N, F)
    Q = q_ratio * R
    out = np.empty_like(x)
    est = x[:, 0].copy()
    P = R.copy()
    out[:, 0] = est
    for t in range(1, x.shape[1]):
        P = P + Q
        K = P / (P + R)
        est = est + K * (x[:, t] - est)
        P = (1 - K) * P
        out[:, t] = est
    return out


_FUNCS = {'lowpass': lowpass, 'hamming': hamming, 'kalman': kalman}


def apply_filters(x, filters, params=None):
    """Apply `filters` (any subset of FILTER_ORDER, always run in the paper's order) to x (N, T, F)."""
    params = params or {}
    unknown = set(filters) - set(FILTER_ORDER)
    if unknown:
        raise ValueError(f"Unknown filters {sorted(unknown)}, available: {list(FILTER_ORDER)}")
    x = x.astype(np.float64)
    for name in FILTER_ORDER:
        if name in filters:
            x = _FUNCS[name](x, **params.get(name, {}))
    return x.astype(np.float32)
