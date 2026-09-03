"""Self-checks for vrheed_analysis.

Run with `python test_analysis.py` — no camera, no Qt, no OpenCV needed.
Every case has a known answer, so a failure means the physics changed.
Also collectable by pytest (``test_all_checks`` below); pytest is optional.
"""

import os
import sys
import time

# The analysis core works on small arrays, where multithreaded BLAS is pure
# overhead: on a loaded machine MKL/OpenBLAS thread spin-up turns a 15 ms
# bounded least-squares fit into seconds.  Pin to one thread *before* numpy
# is imported so the timing check below measures the algorithm, not the OS.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import vrheed_analysis as va

_fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


def close(a, b, rtol=1e-3):
    return np.isfinite(a) and np.isfinite(b) and abs(a - b) <= rtol * abs(b)


# --- electron optics -------------------------------------------------------

check("lambda(20 keV) = 0.0859 A", close(va.wavelength_A(20000), 0.0859, 2e-3),
      f"got {va.wavelength_A(20000):.5f}")
check("lambda(10 keV) = 0.1220 A", close(va.wavelength_A(10000), 0.12205, 2e-3),
      f"got {va.wavelength_A(10000):.5f}")
check("lambda decreases with energy",
      va.wavelength_A(30000) < va.wavelength_A(15000))

# --- spectrum / growth rate ------------------------------------------------

fs, dur, f0 = 30.0, 60.0, 0.37
t = np.arange(0, dur, 1 / fs)
y = 100 + 5 * np.sin(2 * np.pi * f0 * t) + 0.02 * t

f_hat, mag, freq = va.compute_growth_rate(y, times=t, fmin=0.05, fmax=2.0)
check("FFT recovers 0.37 Hz", close(f_hat, f0, 5e-3), f"got {f_hat:.4f}")
check("sub-bin refinement beats bin spacing",
      abs(f_hat - f0) < (freq[1] - freq[0]) * 0.5,
      f"err {abs(f_hat - f0):.5f} vs half-bin {(freq[1] - freq[0]) / 2:.5f}")

# Jittered / dropped frames must not move the answer much.
rng = np.random.default_rng(0)
t_j = np.sort(t + rng.normal(0, 0.004, t.size))
keep = rng.random(t_j.size) > 0.05
f_j, _, _ = va.compute_growth_rate(y[keep], times=t_j[keep], fmin=0.05, fmax=2.0)
check("jitter + 5% dropped frames still recovers f0", close(f_j, f0, 2e-2),
      f"got {f_j:.4f}")

# Band limits must be respected.
y2 = 100 + 5 * np.sin(2 * np.pi * 0.37 * t) + 20 * np.sin(2 * np.pi * 3.0 * t)
f_b, _, _ = va.compute_growth_rate(y2, times=t, fmin=0.05, fmax=1.0)
check("fmax excludes the out-of-band 3 Hz tone", close(f_b, f0, 1e-2),
      f"got {f_b:.4f}")

check("short trace returns no spectrum",
      va.compute_growth_rate(np.arange(5.0), fps=30.0)[1] is None)
check("flat trace does not crash",
      np.isfinite(va.compute_growth_rate(np.ones(600), fps=30.0)[0]))

# --- oscillation counting --------------------------------------------------

damped = 100 + 8 * np.exp(-t / 40.0) * np.sin(2 * np.pi * 0.25 * t)
st = va.oscillation_stats(t, damped, fmin=0.05, fmax=2.0)
check("oscillation counter finds ~15 peaks in 60 s at 0.25 Hz",
      st and 12 <= st['n_peaks'] <= 16, f"got {st.get('n_peaks')}")
check("counted period = 4.0 s", st and close(st['period_s'], 4.0, 2e-2),
      f"got {st.get('period_s', float('nan')):.3f}")
check("counted rate agrees with FFT", st and close(st['rate_hz'], 0.25, 2e-2))
check("damping detected as positive", st and st['damping'] > 0.02,
      f"got {st.get('damping', float('nan')):.3f}")
und = 100 + 8 * np.sin(2 * np.pi * 0.25 * t)
st_u = va.oscillation_stats(t, und, fmin=0.05, fmax=2.0)
check("undamped trace reports ~zero damping",
      st_u and abs(st_u['damping']) < 0.05,
      f"got {st_u.get('damping', float('nan')):.3f}")

# --- damped sine fit (third growth-rate method) -----------------------------

fit = va.damped_sine_fit(t, damped, fmin=0.05, fmax=2.0)
check("damped-sine fit recovers 0.25 Hz",
      fit and close(fit['rate_hz'], 0.25, 5e-3),
      f"got {fit.get('rate_hz', float('nan')):.5f}")
check("damped-sine fit recovers tau = 40 s",
      fit and close(fit['tau_s'], 40.0, 5e-2),
      f"got {fit.get('tau_s', float('nan')):.2f}")
check("damped-sine fit reports a small rate uncertainty",
      fit and 0 < fit['rate_err_hz'] < 0.01,
      f"got {fit.get('rate_err_hz', float('nan')):.6f}")
check("damped-sine fit is a good fit", fit and fit['r2'] > 0.99,
      f"R2={fit.get('r2', 0):.4f}")
check("damped-sine agrees with FFT and peak counting",
      fit and abs(fit['rate_hz'] - st['rate_hz']) < 0.01)
noisy_osc = damped + np.random.default_rng(3).normal(0, 1.0, t.size)
fitn = va.damped_sine_fit(t, noisy_osc, fmin=0.05, fmax=2.0)
check("damped-sine fit survives noise", fitn and close(fitn['rate_hz'], 0.25, 2e-2),
      f"got {fitn.get('rate_hz', float('nan')):.5f}")
short = t[:60]
check("too few periods -> no fit rather than a wrong one",
      va.damped_sine_fit(short, damped[:60], fmin=0.05, fmax=2.0) == {})

# --- long selection: 30 min at 30 fps = 54,000 points ------------------------
# This is the worst case the GUI hands to the fitter twice a second, so it
# must be both right and fast.  Decimation must not touch the FFT path.

t_long = np.arange(0, 1800.0, 1 / fs)
tau_long, f_long, amp_long = 400.0, 0.25, 8.0
long_tr = (100 + amp_long * np.exp(-t_long / tau_long)
           * np.sin(2 * np.pi * f_long * t_long) + 0.001 * t_long)
check("long trace has 54,000 samples", t_long.size == 54000)

t_start = time.perf_counter()
fit_l = va.damped_sine_fit(t_long, long_tr, fmin=0.05, fmax=2.0)
elapsed = time.perf_counter() - t_start
check("54k-point damped-sine fit recovers rate to 0.5%",
      fit_l and close(fit_l['rate_hz'], f_long, 5e-3),
      f"got {fit_l.get('rate_hz', float('nan')):.5f}")
check("54k-point damped-sine fit recovers tau = 400 s to 10%",
      fit_l and close(fit_l['tau_s'], tau_long, 1e-1),
      f"got {fit_l.get('tau_s', float('nan')):.1f}")
check("54k-point fit amplitude is corrected for block averaging",
      fit_l and close(fit_l['amplitude'], amp_long, 1e-2),
      f"got {fit_l.get('amplitude', float('nan')):.4f}")
check("54k-point damped-sine fit runs in under 0.5 s",
      elapsed < 0.5, f"took {elapsed * 1e3:.0f} ms")

# Peaks are counted while their prominence (2 * amplitude) exceeds 10 % of
# the trace span (~2 * amp_long), i.e. until amp has decayed by 10x:
# t_last = tau * ln(10), so n_peaks = t_last * f.
n_expected = tau_long * np.log(10.0) * f_long              # 230.3
t_start = time.perf_counter()
st_l = va.oscillation_stats(t_long, long_tr, fmin=0.05, fmax=2.0)
elapsed_st = time.perf_counter() - t_start
check("54k-point peak counter finds the analytic number of peaks (2%)",
      st_l and abs(st_l['n_peaks'] - n_expected) <= 0.02 * n_expected,
      f"got {st_l.get('n_peaks')} expected ~{n_expected:.0f}")
check("54k-point peak counter period = 4.0 s",
      st_l and close(st_l['period_s'], 1 / f_long, 5e-3),
      f"got {st_l.get('period_s', float('nan')):.4f}")
check("54k-point peak counter runs in under 0.5 s",
      elapsed_st < 0.5, f"took {elapsed_st * 1e3:.0f} ms")

# decimate_uniform itself: rate floor and no-op behaviour.
tu_l, vu_l, fs_l = va.resample_uniform(t_long, long_tr)
td, vd, fsd = va.decimate_uniform(tu_l, vu_l, fs_l, fmax=0.5)
check("decimation keeps fs >= 8*fmax", fsd >= 8 * 0.5 and vd.size < 10000,
      f"fs {fsd:.3f} Hz, {vd.size} points")
check("decimated grid is still uniform",
      np.allclose(np.diff(td), 1 / fsd))
check("decimation preserves the mean", close(vd.mean(), vu_l.mean(), 1e-6))
td2, vd2, fsd2 = va.decimate_uniform(tu_l, vu_l, fs_l, fmax=2.0)
check("fs = 30, fmax = 2 cannot be decimated (8*fmax > fs/2)",
      vd2.size == vu_l.size and fsd2 == fs_l)
tds, vds, fss = va.decimate_uniform(t, damped, fs, fmax=0.5)
check("short trace is returned untouched", vds.size == damped.size)
freq_l = va.compute_spectrum(long_tr, times=t_long)[0]
check("FFT resolution is untouched by decimation (1/T bins)",
      close(freq_l[1] - freq_l[0], 1.0 / 1800.0, 1e-3),
      f"bin {freq_l[1] - freq_l[0]:.6f} Hz")

# --- resampling edge cases ---------------------------------------------------

shuf = np.random.default_rng(2).permutation(t.size)
r_shuf = va.resample_uniform(t[shuf], damped[shuf])
check("unsorted timestamps give a monotonic grid",
      r_shuf is not None and np.all(np.diff(r_shuf[0]) > 0))
check("unsorted timestamps give the sorted trace back",
      r_shuf is not None and np.allclose(r_shuf[1], damped, atol=1e-9)
      and close(r_shuf[2], fs, 1e-9))

# --- profiles --------------------------------------------------------------

x = np.arange(200, dtype=float)
prof = 20 + 300 * np.exp(-0.5 * ((x - 87.3) / 6.0) ** 2)
fit = va.gaussian_fit(x, prof)
check("Gaussian fit centre = 87.3", fit and close(fit['center'], 87.3, 1e-3),
      f"got {fit['center']:.3f}")
check("Gaussian fit FWHM = 2.3548 sigma",
      fit and close(fit['fwhm'], 6.0 * va.FWHM_PER_SIGMA, 1e-3),
      f"got {fit['fwhm']:.3f}")

noisy = prof + np.random.default_rng(1).normal(0, 8, x.size)
m = va.profile_metrics(noisy, x)
check("noisy profile still fits a Gaussian", m and m['method'] == 'gaussian',
      f"method {m and m['method']}")
check("noisy centre within 0.5 px", m and abs(m['center'] - 87.3) < 0.5,
      f"got {m['center']:.3f}")

hm = va.fwhm_half_max(prof, x)
check("half-max FWHM agrees with the fit to 2%",
      hm and close(hm[0], fit['fwhm'], 2e-2), f"got {hm[0]:.3f}")
# A triangle of base half-width 30 crosses half maximum at +/-15, so the
# FWHM is exactly 30 and linear interpolation is exact on straight flanks.
tri = np.clip(1.0 - np.abs(x - 90.0) / 30.0, 0.0, None)
hm_t = va.fwhm_half_max(tri, x)
check("triangle profile FWHM is exactly half the base",
      hm_t and abs(hm_t[0] - 30.0) < 1e-9 and abs(hm_t[1] - 75.0) < 1e-9
      and abs(hm_t[2] - 105.0) < 1e-9,
      f"got {hm_t[0]:.6f} from {hm_t[1]:.3f} to {hm_t[2]:.3f}" if hm_t else "")
check("featureless profile yields no metrics",
      va.profile_metrics(np.full(50, 7.0)) is None)

# --- streak finding --------------------------------------------------------

streaks_x = np.arange(400, dtype=float)
sp = 61.5
prof3 = np.full(400, 10.0)
for n in (-2, -1, 0, 1, 2):
    prof3 += 200 * np.exp(-0.5 * ((streaks_x - (200 + n * sp)) / 5.0) ** 2)
found = va.find_streaks(prof3, streaks_x)
check("finds 5 streaks", len(found) == 5, f"got {len(found)}")
check("mean streak spacing = 61.5 px",
      close(va.mean_streak_spacing(found), sp, 5e-3),
      f"got {va.mean_streak_spacing(found):.3f}")
check("streak positions are sub-pixel",
      found and abs(found[2]['pos'] - 200.0) < 0.2,
      f"got {found[2]['pos']:.3f}" if found else "")
# Three streaks at a non-integer pitch: the count and the pitch to 0.05 px.
x3 = np.arange(320, dtype=float)
pitch = 60.3
prof_three = np.full(320, 10.0)
for c in (100.0, 100.0 + pitch, 100.0 + 2 * pitch):
    prof_three += 200 * np.exp(-0.5 * ((x3 - c) / 4.0) ** 2)
three = va.find_streaks(prof_three, x3)
check("three Gaussians -> three streaks", len(three) == 3, f"got {len(three)}")
check("streak positions land within 0.05 px",
      len(three) == 3 and all(
          abs(s['pos'] - c) < 0.05
          for s, c in zip(three, (100.0, 100.0 + pitch, 100.0 + 2 * pitch))),
      "  ".join(f"{s['pos']:.3f}" for s in three))
check("mean streak spacing recovers 60.3 px to 0.05 px",
      abs(va.mean_streak_spacing(three) - pitch) < 0.05,
      f"got {va.mean_streak_spacing(three):.4f}")
check("one streak has no spacing",
      not np.isfinite(va.mean_streak_spacing(three[:1])))

# --- lattice metrology -----------------------------------------------------

# 20 keV, L = 300 mm, 20 um pixels; a 3.905 A lattice (SrTiO3) should put the
# first-order streak at lambda*L/(a*p) pixels from the specular.
lam = va.wavelength_A(20000)
a_sto, L, pix = 3.905, 300.0, 0.020
dx_px = lam * L / (a_sto * pix)
a_back = va.spacing_to_lattice(dx_px, pix, L, 20.0)
check("spacing -> lattice round-trips (SrTiO3)", close(a_back, a_sto, 1e-6),
      f"got {a_back:.4f} A from {dx_px:.1f} px")
check("bigger separation means smaller lattice",
      va.spacing_to_lattice(2 * dx_px, pix, L, 20.0) < a_back)

K = va.calibration_constant(a_sto, dx_px)
check("calibration constant inverts",
      close(va.lattice_from_calibration(dx_px, K), a_sto, 1e-9))
# A film 1 % relaxed shows a 1 % smaller streak separation.
check("1% larger lattice -> +1.00% strain",
      close(va.strain_percent(
          va.lattice_from_calibration(dx_px / 1.01, K), a_sto), 1.0, 1e-6),
      f"got {va.strain_percent(va.lattice_from_calibration(dx_px / 1.01, K), a_sto):.4f}%")
check("zero strain against itself",
      abs(va.strain_percent(a_sto, a_sto)) < 1e-12)
check("bad input gives nan, not an exception",
      not np.isfinite(va.spacing_to_lattice(0, pix, L, 20.0)))

# --- coherence length -------------------------------------------------------

# Same lambda*L/x relation as the lattice spacing, so K converts widths too.
check("coherence length from calibration",
      close(va.coherence_length(dx_px, K=K), a_sto, 1e-9))
check("a narrower streak means a longer coherence length",
      va.coherence_length(dx_px / 2, K=K) > va.coherence_length(dx_px, K=K))
check("coherence length from geometry matches calibration",
      close(va.coherence_length(dx_px, pixel_mm=pix, camera_distance_mm=L,
                                energy_keV=20.0), a_sto, 1e-6))
check("coherence length without calibration or geometry is nan",
      not np.isfinite(va.coherence_length(10.0)))

# --- kinematic simulation --------------------------------------------------

b1, b2 = va.reciprocal_basis(4.0, 4.0, 90.0)
check("square mesh -> orthogonal reciprocal basis", abs(b1 @ b2) < 1e-9)
check("|b1| = 2pi/a", close(np.linalg.norm(b1), 2 * np.pi / 4.0, 1e-9))
b1h, b2h = va.reciprocal_basis(3.0, 3.0, 120.0)
check("hex mesh reciprocal angle = 60 deg",
      close(np.degrees(np.arccos(
          b1h @ b2h / (np.linalg.norm(b1h) * np.linalg.norm(b2h)))), 60.0, 1e-6))

pat = va.simulate_pattern(3.905, 3.905, 90.0, azimuth_deg=0.0,
                          energy_keV=20.0, theta_deg=2.0,
                          camera_distance_mm=300.0, hk_max=6)
check("simulation returns reflections", len(pat) > 5, f"got {len(pat)}")
spec = [p for p in pat if p['specular']]
check("specular rod is present", len(spec) == 1)
check("specular sits on the beam axis (x = 0)",
      spec and abs(spec[0]['x_mm']) < 1e-9)
check("specular height = L*tan(theta)",
      spec and close(spec[0]['y_mm'], 300.0 * np.tan(np.deg2rad(2.0)), 1e-6),
      f"got {spec[0]['y_mm']:.3f} mm")

first = sorted((p for p in pat if p['h'] == 1 and p['k'] == 0),
               key=lambda p: p['x_mm'])
check("(1,0) streak spacing matches lambda*L/a to 1%",
      first and close(abs(first[0]['x_mm']), lam * 300.0 / 3.905, 1e-2),
      f"got {abs(first[0]['x_mm']):.3f} mm, "
      f"expected {lam * 300.0 / 3.905:.3f} mm")
check("all reflections are above the shadow edge",
      all(p['y_mm'] >= -1e-9 for p in pat))
check("pattern is left-right symmetric at azimuth 0",
      close(sum(p['x_mm'] for p in pat), 0.0, 1e-6)
      or abs(sum(p['x_mm'] for p in pat)) < 1e-6)

pat90 = va.simulate_pattern(3.905, 5.5, 90.0, 90.0, 20.0, 2.0, 300.0, hk_max=6)
pat0 = va.simulate_pattern(3.905, 5.5, 90.0, 0.0, 20.0, 2.0, 300.0, hk_max=6)
w90 = max(abs(p['x_mm']) for p in pat90 if p['h'] or p['k'])
w0 = max(abs(p['x_mm']) for p in pat0 if p['h'] or p['k'])
check("rotating a rectangular mesh 90 deg changes the pattern",
      abs(w90 - w0) > 1e-6)
check("Laue arcs group into a few radii",
      1 <= len(va.laue_radii_mm(pat)) <= len(pat))

# --- ROI metrics -----------------------------------------------------------

patch = np.zeros((20, 40), dtype=np.float64)
patch[10, 30] = 100.0
rm = va.roi_metrics(patch)
check("centroid tracks a bright pixel",
      close(rm['cx'], 30.5 / 40, 1e-6) and close(rm['cy'], 10.5 / 20, 1e-6),
      f"cx={rm['cx']:.4f} cy={rm['cy']:.4f}")
check("ROI mean / max / min / sum", close(rm['mean'], 100 / 800, 1e-9)
      and rm['max'] == 100.0 and rm['min'] == 0.0 and rm['sum'] == 100.0)
check("ROI std dev", close(rm['std'], np.std(patch), 1e-12))
check("brightest-pixel position tracks the spike",
      close(rm['px'], 30.5 / 40, 1e-9) and close(rm['py'], 10.5 / 20, 1e-9),
      f"px={rm['px']:.4f} py={rm['py']:.4f}")
flat = va.roi_metrics(np.full((8, 8), 5.0))
check("flat ROI centroid falls at the centre",
      close(flat['cx'], 0.5, 1e-9) and close(flat['cy'], 0.5, 1e-9))
check("empty ROI returns None", va.roi_metrics(np.zeros((0, 5))) is None)

sat = np.full((10, 10), 255, dtype=np.uint8)
check("saturation = 100% when clipped", close(va.saturated_fraction(sat), 1.0))
sat[0, 0] = 0
check("saturation = 99% with one dark pixel",
      close(va.saturated_fraction(sat), 0.99, 1e-9))
# A 12-bit camera delivers 0-4095 in a uint16 container: without the real
# full scale every pixel looks far from saturation.
sat12 = np.full((10, 10), 1000, dtype=np.uint16)
sat12[:5, :5] = 4095
check("uint16 frame with explicit 12-bit max_value -> 25% saturated",
      close(va.saturated_fraction(sat12, max_value=4095), 0.25, 1e-9),
      f"got {va.saturated_fraction(sat12, max_value=4095):.3f}")
check("same frame against the uint16 default is 0% saturated",
      va.saturated_fraction(sat12) == 0.0)
check("empty frame is 0% saturated",
      va.saturated_fraction(np.zeros((0, 4), dtype=np.uint8)) == 0.0)

# ---------------------------------------------------------------------------


def test_all_checks():
    """pytest entry point: the checks above ran at import time."""
    assert not _fails, f"{len(_fails)} FAILED: " + ", ".join(_fails)


if __name__ == "__main__":
    print()
    if _fails:
        print(f"{len(_fails)} FAILED: " + ", ".join(_fails))
        sys.exit(1)
    print("all checks passed")
