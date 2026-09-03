"""
VRHEED analysis core.

Pure numpy/scipy — no Qt, no OpenCV, no PySpin — so every routine here can be
unit-tested on any machine without a camera or a display.  ``main.py`` is the
only place that knows about hardware and widgets.

Contents
--------
Electron optics      wavelength_A, ewald_radius
Time series          resample_uniform, decimate_uniform, compute_spectrum,
                     compute_growth_rate, oscillation_stats, damped_sine_fit
Profiles / spots     profile_metrics, gaussian_fit, find_streaks, roi_metrics
Lattice metrology    spacing_to_lattice, calibration_constant,
                     lattice_from_calibration, strain_percent
Kinematic simulation reciprocal_basis, simulate_pattern
"""

import numpy as np

try:
    from scipy.signal import detrend, windows, find_peaks
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:                                     # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Electron optics
# ---------------------------------------------------------------------------

def wavelength_A(energy_eV):
    """Relativistic de Broglie wavelength of an electron, in angstroms.

    lambda = h / sqrt(2 m e V (1 + eV / 2 m c^2))
           = 12.2639 / sqrt(V + 0.97845e-6 V^2)   [A, V in volts]

    20 keV -> 0.0859 A, which is the number quoted on every RHEED datasheet.
    """
    v = float(energy_eV)
    if v <= 0:
        raise ValueError("energy must be positive")
    return 12.2639 / np.sqrt(v + 0.97845e-6 * v * v)


def ewald_radius(energy_eV):
    """|k| = 2*pi/lambda, in inverse angstroms."""
    return 2.0 * np.pi / wavelength_A(energy_eV)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def resample_uniform(times, values):
    """Interpolate an unevenly sampled trace onto a uniform grid.

    Frame arrival is never perfectly periodic (USB/GigE jitter, dropped
    frames, binning changes), and an FFT that assumes uniform spacing on
    jittered data smears the oscillation peak.  Resampling at the *median*
    frame interval is robust to a handful of dropped frames.

    Returns ``(t_uniform, v_uniform, fs)`` or ``None`` if there is not enough
    usable data.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.size < 8 or t.size != v.size:
        return None
    order = np.argsort(t)
    t, v = t[order], v[order]
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    span = t[-1] - t[0]
    n = int(span / dt) + 1
    if n < 8:
        return None
    # Guard against a pathological median (e.g. a long pause) blowing memory.
    n = min(n, 4_000_000)
    tu = t[0] + np.arange(n) * dt
    return tu, np.interp(tu, t, v), 1.0 / dt


def decimate_uniform(t, v, fs, fmax, max_points=4000, oversample=8.0):
    """Block-average a uniformly sampled trace down to ~``max_points``.

    A 30 min growth at 30 fps is 54,000 samples, and a six-parameter
    least-squares fit on that many points stalls the GUI thread.  Averaging
    adjacent samples in blocks keeps every oscillation slower than ``fmax``
    intact provided the decimated rate stays at least ``oversample`` times
    ``fmax`` (8 points per period is ample for a sine fit or a peak finder),
    and it lowers the white-noise level by sqrt(block) as a side effect.  If
    that rate limit cannot be met at ``max_points`` the trace is decimated
    only as far as the limit allows, so the band is never aliased.

    Block averaging is a boxcar filter whose gain at frequency f is close to
    sinc(f / fs_dec); at f = fs_dec / 8 that is 0.974.  Frequency and decay
    are unaffected, and :func:`damped_sine_fit` undoes the amplitude loss.

    Returns ``(t_dec, v_dec, fs_dec)``; the inputs unchanged when no
    decimation is needed or possible.
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n <= int(max_points) or fs <= 0:
        return t, v, fs
    block = n // int(max_points)
    fs_floor = float(oversample) * max(float(fmax), 1e-9)
    block = min(block, int(fs / fs_floor))
    if block < 2:
        return t, v, fs
    m = (n // block) * block
    td = t[:m].reshape(-1, block).mean(axis=1)
    vd = v[:m].reshape(-1, block).mean(axis=1)
    return td, vd, fs / block


def compute_spectrum(values, times=None, fps=None):
    """Hann-windowed magnitude spectrum of a detrended trace.

    Returns ``(freq, mag, fs)``; ``(None, None, None)`` when the trace is too
    short.  Either ``times`` (preferred, real timestamps in seconds) or
    ``fps`` must be supplied.
    """
    if times is not None:
        r = resample_uniform(times, values)
        if r is None:
            return None, None, None
        _, data, fs = r
    else:
        data = np.asarray(values, dtype=np.float64)
        fs = float(fps or 0.0)
        if fs <= 0:
            return None, None, None

    n = data.size
    if n < 16 or not np.all(np.isfinite(data)):
        return None, None, None

    data = detrend(data) if _HAVE_SCIPY else data - np.linspace(
        data[0], data[-1], n)
    win = windows.hann(n) if _HAVE_SCIPY else np.hanning(n)
    spec = np.abs(np.fft.rfft(data * win))
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, spec, fs


def _refine_peak(freq, mag, i):
    """Sub-bin peak location by parabolic interpolation on log magnitude.

    FFT bin spacing is 1/T; for a 60 s window that is 17 mHz, which is a 2 %
    error on a 1 Hz oscillation.  A three-point parabola through the log of
    the magnitude recovers roughly an order of magnitude of that.
    """
    if i <= 0 or i >= len(mag) - 1:
        return float(freq[i])
    y0, y1, y2 = (np.log(max(float(mag[j]), 1e-30)) for j in (i - 1, i, i + 1))
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-30:
        return float(freq[i])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(freq[i] + delta * (freq[1] - freq[0]))


def compute_growth_rate(values, times=None, fps=None, fmin=0.01, fmax=2.0):
    """Dominant oscillation frequency inside ``[fmin, fmax]``.

    Returns ``(freq_hz, mag, freq_axis)``.  ``freq_hz`` is 0.0 when nothing
    usable was found; ``mag``/``freq_axis`` are None when the trace was too
    short to transform at all.
    """
    freq, mag, _ = compute_spectrum(values, times=times, fps=fps)
    if freq is None:
        return 0.0, None, None
    band = (freq >= float(fmin)) & (freq <= float(fmax))
    if not np.any(band):
        return 0.0, mag, freq
    masked = np.where(band, mag, 0.0)
    i = int(np.argmax(masked))
    if masked[i] <= 0:
        return 0.0, mag, freq
    return _refine_peak(freq, mag, i), mag, freq


def oscillation_stats(times, values, fmin=0.01, fmax=2.0):
    """Count RHEED oscillations directly in the time domain.

    An FFT needs several periods before its peak means anything; early in a
    growth you often have two or three oscillations and want the period
    anyway.  Peak-to-peak timing also exposes *chirp* (period drifting as the
    surface roughens) that a single FFT peak averages away, and the decay of
    the peak heights is the damping the growth-mode discussion actually turns
    on.

    Returns a dict with ``n_peaks``, ``period_s``, ``period_std_s``,
    ``rate_hz``, ``damping`` (fractional amplitude loss per period, 0 = none)
    and ``peak_times``; empty dict when fewer than two peaks are found.
    """
    r = resample_uniform(times, values)
    if r is None or not _HAVE_SCIPY:
        return {}
    tu, vu, fs = r
    # Long selections are block-averaged; nothing slower than fmax is lost.
    tu, vu, fs = decimate_uniform(tu, vu, fs, fmax)

    vu = detrend(vu)
    span = float(vu.max() - vu.min())
    if span <= 0:
        return {}
    # A real oscillation cannot be faster than fmax; use that as the minimum
    # peak separation so noise spikes are not counted as periods.
    min_dist = max(1, int(fs / max(float(fmax), 1e-6) * 0.5))
    idx, props = find_peaks(vu, distance=min_dist, prominence=span * 0.10)
    if idx.size < 2:
        return {}

    # Sub-sample peak times from a parabola through each maximum; this keeps
    # the period resolution well below one (possibly decimated) sample.
    pt = tu[idx].copy()
    inner = (idx > 0) & (idx < vu.size - 1)
    if np.any(inner):
        j = idx[inner]
        y0, y1, y2 = vu[j - 1], vu[j], vu[j + 1]
        den = y0 - 2.0 * y1 + y2
        shift = np.where(np.abs(den) > 1e-12,
                         0.5 * (y0 - y2) / np.where(den == 0, 1.0, den), 0.0)
        pt[inner] += np.clip(shift, -0.5, 0.5) / fs
    periods = np.diff(pt)
    if periods.size == 0:
        return {}
    mean_p = float(np.mean(periods))
    if mean_p <= 0 or not (1.0 / max(mean_p, 1e-9) >= float(fmin)):
        return {}

    heights = props['prominences']
    damping = 0.0
    if heights.size >= 2 and heights[0] > 0:
        # Fit ln(amplitude) vs peak index -> fractional loss per period.
        k = np.arange(heights.size, dtype=float)
        slope = np.polyfit(k, np.log(np.maximum(heights, 1e-30)), 1)[0]
        damping = float(1.0 - np.exp(slope))

    return {
        'n_peaks':       int(idx.size),
        'period_s':      mean_p,
        'period_std_s':  float(np.std(periods)),
        'rate_hz':       1.0 / mean_p,
        'damping':       damping,
        'peak_times':    pt,
    }


def _damped_sine(t, amp, f, phi, tau, off, slope):
    return off + slope * t + amp * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + phi)


def damped_sine_fit(times, values, f0=None, fmin=0.01, fmax=2.0):
    """Fit a damped sine plus linear baseline to a RHEED oscillation.

    A third growth-rate method, independent of both the FFT and peak counting.
    Unlike them it returns a genuine *uncertainty* on the rate, from the
    covariance of the fit, and a decay constant that is a cleaner measure of
    roughening than regressing the peak heights.

    Returns a dict with ``rate_hz``, ``rate_err_hz``, ``amplitude``,
    ``tau_s`` (1/e decay), ``n_periods`` and ``r2``; empty dict on failure.
    """
    r = resample_uniform(times, values)
    if r is None or not _HAVE_SCIPY:
        return {}
    tu, vu, fs0 = r
    tu = tu - tu[0]
    span = float(vu.max() - vu.min())
    duration = float(tu[-1])
    if span <= 0 or duration <= 0:
        return {}

    if f0 is None or not (fmin <= f0 <= fmax):
        f0 = compute_growth_rate(vu, times=tu, fmin=fmin, fmax=fmax)[0]
    if f0 <= 0:
        return {}
    # Fewer than ~1.5 periods cannot constrain frequency and decay together.
    if f0 * duration < 1.5:
        return {}

    # The FFT has already located the oscillation, so the fit only needs to
    # resolve the neighbourhood of f0.  Decimating while preserving up to
    # 2*f0 keeps the second harmonic of a cuspy RHEED oscillation and cuts a
    # 54k-point fit to a few thousand points.  The upper frequency bound is
    # clamped to the decimated Nyquist so the fit cannot chase an alias.
    tu, vu, fs = decimate_uniform(tu, vu, fs0, min(float(fmax), 2.0 * f0))
    f_hi = min(float(fmax), 0.5 * fs)
    if not (fmin < f0 < f_hi):
        return {}

    # Phase seed from the projection of the detrended trace onto exp(-i w t):
    # for amp*exp(-t/tau)*sin(w t + phi) that sum is ~ (amp/2i) e^{i phi}
    # times a positive real, so phi = arg(sum) + pi/2.  Starting at the right
    # phase keeps the bounded solver from settling on a half-period slip.
    resid0 = detrend(vu)
    z = np.sum(resid0 * np.exp(-2j * np.pi * f0 * tu))
    phi0 = float(np.angle(z) + 0.5 * np.pi) if abs(z) > 0 else 0.0
    phi0 = float((phi0 + np.pi) % (2 * np.pi) - np.pi)

    p0 = [span / 2.0, f0, phi0, max(duration, 1e-3), float(np.mean(vu)), 0.0]
    try:
        p, cov = curve_fit(
            _damped_sine, tu, vu, p0=p0, maxfev=8000,
            bounds=([0.0, fmin, -2 * np.pi, 1e-3, -np.inf, -np.inf],
                    [np.inf, f_hi, 2 * np.pi, 1e6, np.inf, np.inf]))
    except Exception:
        return {}

    amp, f, _phi, tau, _off, _slope = (float(v) for v in p)
    # Undo the gain of averaging ``block`` point samples at the fitted
    # frequency: sin(pi f block / fs0) / (block sin(pi f / fs0)), which is
    # exactly 1 when nothing was decimated (block = 1).
    block = int(round(fs0 / fs))
    if block > 1:
        x = np.pi * f / fs0
        amp /= max(float(np.sin(block * x) / (block * np.sin(x))), 1e-3)
    err = (float(np.sqrt(abs(cov[1][1]))) if np.all(np.isfinite(cov))
           else float('nan'))
    resid = vu - _damped_sine(tu, *p)
    ss_tot = float(np.sum((vu - vu.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return {'rate_hz': f, 'rate_err_hz': err, 'amplitude': amp,
            'tau_s': tau, 'n_periods': f * duration, 'r2': r2}


# ---------------------------------------------------------------------------
# Line profiles and spots
# ---------------------------------------------------------------------------

def _gauss(x, amp, cen, sigma, off):
    return off + amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))    # 2.3548


def gaussian_fit(x, y):
    """Least-squares Gaussian-on-a-pedestal fit to a 1-D profile.

    Returns a dict with ``amp``, ``center``, ``sigma``, ``offset``, ``fwhm``
    and ``r2``; ``None`` if scipy is unavailable or the fit does not converge.
    """
    if not _HAVE_SCIPY:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 5 or x.size != y.size or not np.all(np.isfinite(y)):
        return None

    off0 = float(np.percentile(y, 10))
    amp0 = float(y.max() - off0)
    if amp0 <= 0:
        return None
    cen0 = float(x[int(np.argmax(y))])
    # Second moment of the above-background signal is a good sigma seed.
    w = np.clip(y - off0, 0, None)
    tot = float(w.sum())
    sig0 = (float(np.sqrt(np.sum(w * (x - cen0) ** 2) / tot)) if tot > 0
            else float(x.ptp()) / 6.0)
    sig0 = max(sig0, float(abs(x[1] - x[0])) if x.size > 1 else 1.0)

    try:
        p, _ = curve_fit(
            _gauss, x, y, p0=[amp0, cen0, sig0, off0], maxfev=4000,
            bounds=([0.0, float(x.min()), 1e-6, -np.inf],
                    [np.inf, float(x.max()), float(x.ptp()) or np.inf, np.inf]))
    except Exception:
        return None

    amp, cen, sigma, off = (float(v) for v in p)
    resid = y - _gauss(x, *p)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return {'amp': amp, 'center': cen, 'sigma': sigma, 'offset': off,
            'fwhm': FWHM_PER_SIGMA * sigma, 'r2': r2}


def fwhm_half_max(values, x=None):
    """FWHM by linear interpolation of the half-maximum crossings.

    Returns ``(fwhm, left, right, half_value)`` in units of ``x`` (pixel
    index when ``x`` is None), or ``None``.  Model-free fallback for profiles
    a Gaussian will not fit.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 0:
        return None
    half = lo + (hi - lo) * 0.5
    above = np.where(arr >= half)[0]
    if above.size < 2:
        return None

    left = float(above[0])
    if above[0] > 0:
        j = int(above[0]) - 1
        step = float(arr[j + 1] - arr[j])
        if abs(step) > 1e-12:
            left = j + (half - arr[j]) / step
    right = float(above[-1])
    if above[-1] < arr.size - 1:
        j = int(above[-1])
        step = float(arr[j] - arr[j + 1])
        if abs(step) > 1e-12:
            right = j + (arr[j] - half) / step

    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        scale = float(x[1] - x[0]) if x.size > 1 else 1.0
        return (right - left) * scale, x[0] + left * scale, \
               x[0] + right * scale, half
    return right - left, left, right, half


def profile_metrics(values, x=None, prefer_fit=True):
    """Position and width of the dominant feature in a line profile.

    Tries a Gaussian fit first (sub-pixel, insensitive to noise on the tails)
    and falls back to half-maximum crossings when the fit fails or is poor.
    ``method`` in the result says which one produced the numbers.
    """
    arr = np.asarray(values, dtype=np.float64)
    if x is None:
        x = np.arange(arr.size, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if prefer_fit:
        fit = gaussian_fit(x, arr)
        if fit is not None and fit['r2'] > 0.60 and fit['fwhm'] < float(x.ptp()):
            return {'center': fit['center'], 'fwhm': fit['fwhm'],
                    'amplitude': fit['amp'], 'background': fit['offset'],
                    'r2': fit['r2'], 'method': 'gaussian',
                    'half': fit['offset'] + fit['amp'] * 0.5}

    hm = fwhm_half_max(arr, x)
    if hm is None:
        return None
    fwhm, left, right, half = hm
    return {'center': 0.5 * (left + right), 'fwhm': fwhm,
            'amplitude': float(arr.max() - arr.min()),
            'background': float(arr.min()), 'r2': float('nan'),
            'method': 'half-max', 'half': half}


def find_streaks(values, x=None, prominence_frac=0.10, max_peaks=12):
    """Locate diffraction streaks in a line profile, sub-pixel refined.

    Returns a list of dicts ``{'pos', 'height', 'prominence'}`` sorted by
    position.  ``pos`` is in units of ``x``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if x is None:
        x = np.arange(arr.size, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if arr.size < 5 or not _HAVE_SCIPY:
        return []
    span = float(arr.max() - arr.min())
    if span <= 0:
        return []

    idx, props = find_peaks(arr, prominence=span * float(prominence_frac))
    if idx.size == 0:
        return []
    if idx.size > max_peaks:                       # keep the strongest few
        keep = np.argsort(props['prominences'])[-max_peaks:]
        idx, prom = idx[keep], props['prominences'][keep]
        order = np.argsort(idx)
        idx, prom = idx[order], prom[order]
    else:
        prom = props['prominences']

    step = float(x[1] - x[0]) if x.size > 1 else 1.0
    out = []
    for i, p in zip(idx, prom):
        i = int(i)
        pos = float(x[i])
        if 0 < i < arr.size - 1:                   # parabolic sub-pixel
            y0, y1, y2 = arr[i - 1], arr[i], arr[i + 1]
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-12:
                pos += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * step
        out.append({'pos': pos, 'height': float(arr[i]), 'prominence': float(p)})
    return out


def roi_metrics(sub):
    """Every scalar VRHEED can track for one ROI, from one 2-D patch.

    ``cx``/``cy`` (centroid) and ``px``/``py`` (brightest pixel) are in
    *fractional ROI coordinates* (0-1), so they stay meaningful when the ROI
    is moved or the binning changes.  The ROI minimum is removed before the
    centroid, otherwise a bright background pins it at the geometric centre.
    """
    a = np.asarray(sub, dtype=np.float64)
    if a.size == 0:
        return None
    h, wd = a.shape[:2]

    w = a - a.min()
    wsum = float(w.sum())
    if wsum > 0:
        ys = (np.arange(h) + 0.5) / h
        xs = (np.arange(wd) + 0.5) / wd
        cy = float((w.sum(axis=1) * ys).sum() / wsum)
        cx = float((w.sum(axis=0) * xs).sum() / wsum)
    else:
        cx = cy = 0.5

    pr, pc = np.unravel_index(int(np.argmax(a)), a.shape)
    return {'mean': float(a.mean()), 'max': float(a.max()),
            'min': float(a.min()), 'sum': float(a.sum()),
            'std': float(a.std()),
            'cx': cx, 'cy': cy,
            'px': (pc + 0.5) / wd, 'py': (pr + 0.5) / h}


def saturated_fraction(frame, max_value=None):
    """Fraction of pixels within 1 % of full scale.

    Saturated pixels clip the top of every RHEED oscillation, which flattens
    the peaks and biases both the FFT amplitude and any width measurement.
    """
    a = np.asarray(frame)
    if a.size == 0:
        return 0.0
    if max_value is None:
        max_value = 65535.0 if a.dtype == np.uint16 else 255.0
    return float(np.count_nonzero(a >= max_value * 0.99)) / a.size


# ---------------------------------------------------------------------------
# Lattice metrology
# ---------------------------------------------------------------------------
#
# RHEED small-angle geometry.  A streak of order n sits a distance
#   x = n * lambda * L / d
# from the specular streak, where d is the real-space in-plane periodicity,
# L the sample-to-screen distance and lambda the electron wavelength.  Hence
#   d = n * lambda * L / x.
# Everything below is that one relation, expressed two ways: absolute (you
# know L) and relative (you know a reference lattice constant).

def spacing_to_lattice(delta_px, pixel_mm, camera_distance_mm,
                       energy_keV, order=1):
    """In-plane lattice spacing (A) from a streak separation in pixels."""
    delta_px = float(delta_px)
    if delta_px <= 0 or pixel_mm <= 0 or camera_distance_mm <= 0:
        return float('nan')
    lam = wavelength_A(float(energy_keV) * 1000.0)
    x_mm = delta_px * float(pixel_mm)
    return float(order) * lam * float(camera_distance_mm) / x_mm


def calibration_constant(known_a_A, delta_px, order=1):
    """Lattice-times-separation constant K from a known surface.

    Calibrating against a substrate whose lattice constant you trust removes
    the need to measure L and the pixel pitch at all: K = a * dx / n is fixed
    by the geometry, so any later separation gives a = K * n / dx.
    """
    if delta_px <= 0:
        return float('nan')
    return float(known_a_A) * float(delta_px) / float(order)


def lattice_from_calibration(delta_px, K, order=1):
    """Invert :func:`calibration_constant`."""
    if delta_px <= 0 or not np.isfinite(K):
        return float('nan')
    return float(K) * float(order) / float(delta_px)


def strain_percent(a, a_ref):
    """In-plane strain (a - a_ref) / a_ref, in percent."""
    if not np.isfinite(a) or not np.isfinite(a_ref) or a_ref == 0:
        return float('nan')
    return (a - a_ref) / a_ref * 100.0


def coherence_length(fwhm_px, pixel_mm=None, camera_distance_mm=None,
                     energy_keV=None, K=None):
    """In-plane coherence length (A) from a streak FWHM in pixels.

    A streak of finite width means the surface is ordered only over a finite
    distance.  The width in reciprocal space is dk = (2*pi/lambda) * w / L for
    a screen width w, and the coherence length is 2*pi/dk -- which reduces to
    exactly the same lambda*L/x relation used for lattice spacing.  So the
    calibration constant K from a known substrate converts widths as well as
    separations, and no extra calibration step is needed.

    Pass either ``K`` (preferred) or the full geometry.
    """
    if K is not None:
        return lattice_from_calibration(fwhm_px, K)
    if None in (pixel_mm, camera_distance_mm, energy_keV):
        return float('nan')
    return spacing_to_lattice(fwhm_px, pixel_mm, camera_distance_mm, energy_keV)


def mean_streak_spacing(streaks):
    """Average nearest-neighbour spacing of a streak list, in the same units.

    Averaging every adjacent gap beats using the outermost pair divided by the
    order: it is insensitive to one missing streak at the edge of the ROI.
    """
    if len(streaks) < 2:
        return float('nan')
    pos = np.sort(np.array([s['pos'] for s in streaks], dtype=np.float64))
    return float(np.mean(np.diff(pos)))


# ---------------------------------------------------------------------------
# Kinematic RHEED simulation
# ---------------------------------------------------------------------------
#
# Coordinates: the surface is the x-z plane, the surface normal is +y, and the
# electron beam runs along +z at grazing angle theta below the surface:
#
#     k_i = k (0, -sin(theta), cos(theta))
#
# A 2-D surface lattice gives reciprocal *rods* at G = h b1 + k b2 lying in the
# x-z plane and extending along y.  Elastic scattering into a rod requires
#
#     k_f = k_i + G + q y_hat        with |k_f| = |k| = k
#
# so k_f,y = sqrt(k^2 - k_f,x^2 - k_f,z^2); a real root is the Bragg condition.
# A flat screen a distance L downstream maps k_f to
#
#     X = L k_f,x / k_f,z,  Y = L k_f,y / k_f,z
#
# with Y = 0 at the shadow edge, which is where the returned coordinates are
# referenced.  Rods with k_f,y < 0 are blocked by the sample and are dropped.

def reciprocal_basis(a_A, b_A, gamma_deg):
    """2-D reciprocal basis vectors (1/A, without 2*pi) for a surface mesh.

    Real-space mesh: a1 = (a, 0), a2 = b (cos gamma, sin gamma), both in the
    surface plane.  Returns ``(b1, b2)`` as length-2 arrays in the same plane.
    """
    g = np.deg2rad(float(gamma_deg))
    a1 = np.array([float(a_A), 0.0])
    a2 = np.array([float(b_A) * np.cos(g), float(b_A) * np.sin(g)])
    area = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(area) < 1e-12:
        raise ValueError("degenerate surface mesh")
    twopi = 2.0 * np.pi
    b1 = twopi / area * np.array([a2[1], -a2[0]])
    b2 = twopi / area * np.array([-a1[1], a1[0]])
    return b1, b2


def simulate_pattern(a_A, b_A, gamma_deg, azimuth_deg, energy_keV,
                     theta_deg, camera_distance_mm, hk_max=6,
                     y_max_mm=None):
    """Kinematic RHEED pattern for a 2-D surface mesh.

    Returns a list of dicts, one per allowed reflection::

        {'h', 'k', 'x_mm', 'y_mm', 'radius_mm', 'specular'}

    ``x_mm``/``y_mm`` are screen coordinates relative to the point where the
    beam axis crosses the shadow edge: +x to the right, +y up the screen.
    ``radius_mm`` is the distance from that origin, which is what makes the
    Laue arcs; ``specular`` marks the (0,0) rod.

    The caller supplies pixel scale and origin, so this stays pure geometry.
    """
    k = ewald_radius(float(energy_keV) * 1000.0)
    th = np.deg2rad(float(theta_deg))
    b1, b2 = reciprocal_basis(a_A, b_A, gamma_deg)

    phi = np.deg2rad(float(azimuth_deg))
    rot = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),  np.cos(phi)]])
    b1, b2 = rot @ b1, rot @ b2

    # k_i = k (0, -sin(theta), cos(theta)); only its in-plane components
    # shift the rods.  Its y component enters solely through |k_f| = k in the
    # Bragg condition below, so it is never needed on its own.
    kix, kiz = 0.0, k * np.cos(th)
    L = float(camera_distance_mm)
    n = int(hk_max)

    hs, ks = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1),
                         indexing='ij')
    hs, ks = hs.ravel(), ks.ravel()
    gx = hs * b1[0] + ks * b2[0]
    gz = hs * b1[1] + ks * b2[1]

    kfx = kix + gx
    kfz = kiz + gz
    rad2 = k * k - kfx ** 2 - kfz ** 2

    # Rod must be reachable (real k_f,y) and scattered downstream (k_f,z > 0)
    # or it never leaves toward the screen.
    ok = (rad2 >= 0) & (kfz > 1e-9)
    if not np.any(ok):
        return []
    hs, ks, kfx, kfz = hs[ok], ks[ok], kfx[ok], kfz[ok]
    kfy = np.sqrt(rad2[ok])

    x_mm = L * kfx / kfz
    y_mm = L * kfy / kfz
    r_mm = np.hypot(x_mm, y_mm)

    if y_max_mm is not None:
        keep = y_mm <= float(y_max_mm)
        hs, ks, x_mm, y_mm, r_mm = (v[keep] for v in (hs, ks, x_mm, y_mm, r_mm))

    out = [{'h': int(h), 'k': int(kk), 'x_mm': float(x), 'y_mm': float(y),
            'radius_mm': float(r), 'specular': (h == 0 and kk == 0)}
           for h, kk, x, y, r in zip(hs, ks, x_mm, y_mm, r_mm)]
    out.sort(key=lambda d: (abs(d['h']) + abs(d['k']), d['x_mm']))
    return out


def laue_radii_mm(pattern, tol_mm=0.5):
    """Distinct Laue-arc radii present in a simulated pattern.

    Reflections cluster onto a handful of radii; grouping them gives the arcs
    to draw as overlay circles.
    """
    if not pattern:
        return []
    r = np.sort(np.array([p['radius_mm'] for p in pattern]))
    groups, cur = [], [r[0]]
    for v in r[1:]:
        if v - cur[-1] <= float(tol_mm):
            cur.append(v)
        else:
            groups.append(float(np.mean(cur)))
            cur = [v]
    groups.append(float(np.mean(cur)))
    return [g for g in groups if g > tol_mm]
