# VRHEED — Precision Analyzer

Live RHEED acquisition and analysis for whatever camera is on your chamber,
plus offline re-analysis of recorded growths.  Version 2.2.0.

```
main.py             Qt application — display, ROIs, plots, settings, log
vrheed_cameras.py   Camera backends. One interface over FLIR, Basler, Allied
                    Vision, GenICam, scientific cameras, USB, network streams,
                    screen capture and a simulated source.
vrheed_analysis.py  All physics and signal processing. No Qt, no OpenCV,
                    no camera driver, so it runs and can be tested anywhere.
test_analysis.py    84 self-checks with known answers.  python test_analysis.py
test_app_smoke.py   End-to-end GUI test in file mode, no camera needed.
                    QT_QPA_PLATFORM=offscreen python test_app_smoke.py
test_cameras.py     Camera backends and the source chooser, no camera needed.
                    QT_QPA_PLATFORM=offscreen python test_cameras.py
requirements.txt    Runtime dependencies; every camera driver is optional
main.spec           PyInstaller build.  build.bat -> VRHEED.exe
vrheed.log          Written next to the exe at run time (see Changes in 2.1)
```

## Cameras

VRHEED 2.2 talks to more than one kind of camera.  The **Source** box at the
top of the *Camera* tab lists everything found on the machine; pick one and it
connects.  *Camera ▸ Rescan* (F5) looks again after something is plugged in,
and *Help ▸ Camera backends* shows which drivers are installed and what to
install for the rest.

| Camera | Backend | Status | Install |
|---|---|---|---|
| FLIR / Point Grey — Blackfly, Grasshopper, Chameleon | Spinnaker | **Verified on hardware** | PySpin, from the FLIR Spinnaker SDK installer (not on PyPI) |
| Simulated RHEED pattern, for demos, teaching and tests | Synthetic | **Verified** | nothing |
| USB webcams, USB microscope cameras, HDMI/composite capture dongles, analogue cameras on a USB frame grabber | USB / UVC | Partly verified | nothing — OpenCV is already a dependency |
| Basler — ace, ace 2, dart, boost | pylon | **Unverified** | `pip install pypylon` |
| Allied Vision — Manta, Alvium, Mako | Vimba X | **Unverified** | `vmbpy`, from the Vimba X SDK |
| Any other GigE Vision / USB3 Vision camera — IDS, Lucid, JAI, Baumer, Ximea, Matrix Vision, Photonfocus, Emergent | GenICam / GenTL | **Unverified** | `pip install harvesters` plus any one vendor SDK for its `.cti` producer |
| Andor iXon / Zyla, Hamamatsu ORCA, Princeton Instruments PIXIS, Photometrics, PCO, Thorlabs scientific | pylablib | **Unverified** | `pip install pylablib` plus the vendor SDK |
| NI IMAQ / IMAQdx frame grabbers — how a 1990s analogue RHEED camera gets digitised | pylablib | **Unverified** | `pip install pylablib` plus NI-IMAQ |
| IP cameras and re-streamed feeds (RTSP, HTTP-MJPEG) | Network | **Unverified** | nothing; add the URL with *Camera ▸ Add network camera* |
| A region of the desktop — the live-image pane of kSA 400, Staib or any vendor software that owns the camera exclusively | Screen capture | **Unverified** | `pip install mss` |

### What the status column means

Be careful with this. Only one backend has been used with a real camera.

**Verified on hardware** — Spinnaker, on a FLIR Blackfly S BFS-U3-27S5M:
enumerated, connected, live frames rendered, gain / exposure / binning
exercised. Its call sequence is additionally pinned by a stubbed PySpin in
`test_cameras.py`, which checks the autos go off in the right order, gain in
dB, exposure in microseconds, incomplete frames dropped, every image released,
binning resetting the sensor window, and a clean shutdown.

**Verified** — Synthetic. Nothing to attach; the test suite measures its
specular oscillation and gets the documented 0.25 Hz back.

**Partly verified** — USB / UVC. The discovery and open path runs on Windows,
Linux and macOS in CI, but always on machines with no camera attached, so the
"there is no camera here" path is well tested and the "there is one" path is
not. Expect it to work; expect gain and exposure to be approximate, because
UVC drivers disagree about the units.

**Unverified** — written against each vendor's published API and never run
against the hardware, because none of it was available. Read them as a
starting point, not a guarantee. They are guarded so a wrong assumption
degrades to *backend unavailable* or a refused control rather than a crash,
and `python vrheed_cameras.py` will tell you what it can see, but the first
person to point one of these at a real camera should expect to fix something.

If you get one working — or find what is wrong with it — please open an issue
or a pull request saying which camera and which SDK version, and this table
can be corrected.

Nothing above is a hard dependency.  With none of them installed VRHEED still
opens, still analyses recorded video and images, and still offers the USB,
network and simulated sources — which is what you want on a laptop, and what a
new user should meet before being asked to install a 1 GB vendor SDK.

**The GenICam backend is the one to reach for when your camera is not listed.**
Install any vendor's SDK for its GenTL producer and it will usually drive any
other vendor's standards-compliant camera.  Set `GENICAM_GENTL64_PATH` to the
folder holding the `.cti` file if the installer did not.

`python vrheed_cameras.py` prints the backends and the cameras it can see —
the first thing to run when a camera does not appear in the app.

### What each backend can and cannot do

**VRHEED starts at 4x4 binning.** A RHEED pattern is a handful of broad
streaks, so the resolution costs nothing you were using, while each output
pixel collects 16x the photons and the camera delivers frames faster - a
cleaner intensity trace, which is the measurement everything else rests on.
Drop to 1x from the Camera tab when you want the sharpest streak profile.

Two consequences worth knowing. The pixel-pitch box on the Lattice tab means
the pitch *after* binning, so at 4x it is your sensor pitch times four (times
lens magnification); the Lattice tab now says so with the binning in force.
And a calibration made with *Calibrate from pattern* is rescaled automatically
when you change binning, so you do not have to redo it.

Gain, exposure, frame rate and binning are enabled per camera; a control the
camera cannot honour is greyed out rather than silently ignored.  Binning that
the hardware cannot do is done in software (mean pooling) so the 1× / 2× / 4×
buttons work on every source — the status line says which kind you got, since
only hardware binning also reduces read noise and raises the frame rate.

Two sources deserve their caveats stated plainly, and the app states them in
the Source box when you connect:

- **USB / UVC.** Gain and exposure are driver-dependent and approximate; the
  units differ between DirectShow, V4L2 and AVFoundation and many webcams
  ignore the request entirely.  On Windows, *Driver…* opens the camera's own
  property sheet, which is the reliable way to turn auto-exposure off — and
  auto-exposure must be off, because every auto step puts an edge in the
  intensity trace that looks exactly like a growth transient.
- **Screen capture and network streams.** You are measuring a display or a
  compressed video, so relative changes and oscillation timing survive but
  absolute intensities do not.  For screen capture the frame rate is the
  capture rate, not the camera's, so the FFT frequency axis is only as good as
  what the vendor software is drawing.

On macOS, the USB and screen-capture backends need camera and screen-recording
permission in *System Settings ▸ Privacy & Security*.

### Auto-connect

VRHEED reconnects to the camera it used last session when it is still there.
Otherwise it takes the first real camera it finds — never the simulated source
and never a screen region, both of which would happily produce frames and look
exactly like a working camera.

## Changes in 2.2

**More than one kind of camera.** The camera layer moved out of `main.py` into
`vrheed_cameras.py` behind a single `CameraBackend` interface, and nine
backends were written against it — see [Cameras](#cameras) above. `main.py` no
longer contains the word `PySpin`: it asks the backend for a frame and the
backend deals with buffer release, incomplete frames, colour sensors and
binning. Adding a tenth camera is a subclass and one line in `BACKENDS`.

Alongside it: a source chooser in the *Camera* tab with rescan and connect, a
*Camera* menu, *Help ▸ Camera backends* listing what is installed and what to
install, capability-driven controls (a camera that has no gain node no longer
offers a gain slider that does nothing), control ranges read from the camera,
a resolution chooser for USB cameras, software binning so 1×/2×/4× works on
every source, per-camera metadata in exported CSV headers, and reconnection to
the camera used last session.

## Changes in 2.1

**Crash logging.** PyQt5 aborts the whole process when a Python exception
escapes a slot or timer callback and `sys.excepthook` is still the default —
one malformed frame used to end a growth run with no trace. The hook is now
replaced: unhandled exceptions are written with a full traceback to a rotating
`vrheed.log` (1 MB × 3) next to the executable and shown briefly in the status
bar, and the app carries on. The frame pipeline additionally catches its own
errors and logs each distinct one once, so a fault that recurs at 60 Hz cannot
fill the log in seconds. *Settings ▸ Open log file location* tells you where
it is. The same hook serves the camera grab thread; from there it only logs,
since touching a widget off the GUI thread is itself a crash.

**Stale frames.** Switching source (open a video, open a still, return to the
camera, change binning) now empties the frame queue first. Previously a live
frame still queued when a file opened was measured and drawn as if it were the
file's first frame, and a still image could be silently dropped because the
queue was full.

**Every video frame is measured.** The UI tick used to take one frame from the
queue and draw it; at 10× playback that meant one sample in ten reached the
histories, and the FFT saw an aliased, irregularly sampled trace. Measurement
and rendering are now separate steps: every queued frame is measured, only the
last is drawn. Playback speed is therefore limited by how fast the machine can
*process* frames, never by dropping them — a 20× request that the machine
cannot meet simply runs slower than 20×, with every sample intact. Playback
uses a fixed 15 ms tick with a fractional frame budget per tick, because the
old `1000/(fps·speed)` interval hit Qt's ~1 ms floor at high speed and ran
slower than asked. The smoke test replays a 1200-frame video at 10× and checks
it produced exactly 1200 samples.

**Seek.** Seeking truncates every history at the target time, so seeking
backwards no longer draws a zigzag back over the existing trace and the FFT
window sees one monotonic timeline. When playback is stopped or paused the
frame at the seek position is decoded and shown, so you can see where you
landed. While paused that frame is shown but not recorded, and the file is
rewound so it is measured once when playback resumes.

**Persistent settings and the Settings menu.** Lattice geometry, analysis
band and history length, display (colormap, contrast, levels, gamma,
averaging), the whole Simulate tab including the shadow-edge origin, the
window geometry and splitter positions, and the last folder used in file
dialogs are saved on close (`QSettings`, `UCF/VRHEED`) and restored on start.
Camera gain, exposure and frame rate are deliberately *not* persisted: they
belong to the camera and are set per growth from the pattern. *Settings ▸
Reset settings to defaults* forgets everything and restores the built-in
values without touching the ROIs. Restore happens before any ROI exists, so
nothing a spin box triggers can clear history.

**Session files are v3.** They now carry the Simulate tab (a, b, γ, azimuth,
θ, max order, px/mm, overlay on/off, origin) alongside the ROIs, since both
describe the same geometry of the same sample. v1 and v2 files still load;
loading also recomputes the lattice readout, which caches its last numbers.

**FFT refresh cadence.** The spectra, peak markers and metric panel refresh at
most four times a second; the time traces still redraw for every displayed
frame. The FFT of a 30 min trace for every ROI costs more than the whole frame
pipeline and nobody reads the numbers faster than that. Moving the selection,
clicking another ROI, changing an ROI's metric or seeking forces an immediate
refresh, and the switch also bypasses the separate 0.5 s throttle on the
peak-count and damped-sine lines so the panel changes ROI at once.

**Frame averaging is a running sum.** The N-frame rolling mean adds the
incoming frame and subtracts the one leaving the window instead of re-summing
up to 32 full frames every tick; the sum is rebuilt from scratch every 1000
frames so float32 rounding cannot accumulate into a visible offset over an
hour. The window and its sum are always reset together on any change of frame
size, binning, rotation or source.

**Decimated damped-sine fit and peak counting.** A 30 min growth at 30 fps is
54 000 samples, and a six-parameter least-squares fit on all of them stalled
the GUI. Long selections are now block-averaged to ~4000 points before the fit
and before peak finding, subject to the rule that the decimated rate stays at
least **8 × fmax** (eight points per period), so nothing inside the analysis
band can alias; if that limit cannot be met at 4000 points the trace is
decimated only as far as the limit allows. Block averaging is a boxcar whose
gain at fs_dec/8 is 0.974; frequency and decay are unaffected and the fit
corrects the amplitude. The fit is also seeded with the FFT phase, which is
what stops it converging to a half-period-shifted local minimum. At 30 fps with
fmax = 2 Hz the rule gives a block of 1.875 → 1, so the peak counter is
effectively undecimated; the sine fit uses `min(fmax, 2·f0)` and does decimate
for slow oscillations.

**Single-thread BLAS.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` are set to 1 (via `setdefault`, so an environment value
wins) before numpy is imported. Every array the app hands to BLAS is small,
and for those the thread start-up and hand-off costs more than the arithmetic;
worse, under machine load the multi-threaded SVD inside `curve_fit`
oversubscribed the cores and stalled for whole seconds. Single-thread is faster
here and never stalls the UI.

**Environment variables** used by the tests so they never touch the
operator's real configuration: `VRHEED_SETTINGS_FILE` redirects `QSettings`
to an INI file; `VRHEED_LOG_FILE` redirects the log. Both are optional.

Also: file dialogs open in the last folder used; the About box and the CSV
metadata header record the app version, library versions and log path; the
history length in file mode is sized from the file's own frame rate.

**Known limitation.** `oscillation_stats` (peak counting) over-counts on very
noisy traces, because its 10 % prominence threshold is relative to the span of
the trace and a noise excursion of that size becomes a "peak". The FFT and
damped-sine numbers are unaffected; when the counted period disagrees with
them on a noisy trace, trust those two.

## Bugs fixed

**Thickness rates were 10× too low.** `nm/min` used `Å/s × 0.6` and `µm/hr`
used `Å/s × 0.036`. One Å/s is 0.1 nm × 60 s = **6 nm/min** and
1e-4 µm × 3600 s = **0.36 µm/hr**. The `Hz`, `ML/s` and `Å/s` readings were
always correct, so anything reported in those units is unaffected.

**The time axis was frame-index/fps, not real time.** Three consequences, all
gone now that every sample carries its own timestamp:

- An ROI drawn ten minutes into a growth was replotted from t = 0, so two ROIs
  that were oscillating in phase appeared shifted by minutes.
- Once a history deque filled, old samples dropped off the front while the axis
  still started at zero, so the whole trace slid backwards in time.
- Frame jitter and dropped frames smeared the FFT peak. Traces are now
  resampled onto a uniform grid at the median frame interval before transform.

**An ROI dragged inside-out or off the frame stopped recording silently.** The
empty array produced no samples while the other ROIs kept going. Boxes are now
ordered, clamped to the frame and given a minimum size.

**Restarting acquisition could run two grab threads at once.** Changing binning
signalled the old thread to stop and started a new one 150 ms later without
waiting; the old one could still be inside `GetNextImage`. It is now joined.

**Closing while recording left an unplayable file.** The `VideoWriter` is
released on close. Changing binning or rotation mid-recording also used to
silently drop every later frame (frame size changed); recording now stops with
a message instead.

**Spinnaker buffers leaked on a grab error.** An image whose `GetNDArray`
raised was never released; Spinnaker hands out a fixed pool, and after a
handful the stream stalled. Release now happens in a `finally`.

Also: the time axis no longer stays stretched to the previous run's length
after Clear; loading an ROI file no longer leaves the axis unanchored; the
capture thread no longer spins at 100 % CPU on a persistent camera error;
the FFT and metrics refresh while paused, so dragging the selection works.

## Layout

Controls moved into tabs (Camera / Display / Analysis / Lattice / Simulate) —
stacked vertically they were taller than the window and pushed the plots off
the bottom. Splitter panes can no longer be collapsed to nothing. The window
never opens larger than the available screen. High-DPI scaling is enabled
before the `QApplication` is built. The status bar is created up front rather
than appearing on first use and shoving every widget up a few pixels.

## New analysis

**Per-ROI tracked quantity.** Each ROI records one of: mean / peak / min /
integrated intensity, intensity std dev, centroid X or Y, brightest-pixel X or
Y, profile FWHM, coherence length, profile position, or streak spacing.
Centroid tracks spot drift; FWHM tracks streak sharpening and surface order;
streak spacing tracks the in-plane lattice parameter in real time. Switching
clears that ROI's history, since the old samples measured something else.

**ROI window tracking.** An ROI can re-centre itself every frame on its
brightest pixel or its centroid, so the box follows a streak that drifts during
growth instead of sliding off it. It corrects a fraction of the error per frame
rather than snapping, so it settles quickly without chasing shot noise. Turn it
off when you are measuring spot *position* — a tracking box holds the centroid
at its own centre by construction. Tracking is applied per measured frame, so
it keeps up at any playback speed.

**In-plane coherence length.** A streak of finite width means order over a
finite distance. The width in reciprocal space obeys the same `λL/x` relation
as the streak separation, so the lattice calibration converts widths for free
and no second calibration is needed. Reported only when the ROI contains a
single resolved streak: across several streaks the half-maximum crossings span
the whole group, which would turn a 300 px "width" into a sub-ångström
coherence length.

**Gaussian profile fitting.** Position and width come from a least-squares
Gaussian-on-a-pedestal fit — sub-pixel and far less noise-sensitive than
half-maximum crossings, which remain the fallback when the fit fails.

**Sub-bin FFT peak.** Parabolic interpolation on the log magnitude. Bin spacing
is 1/T, which is 17 mHz for a 60 s window — a 2 % error on a 1 Hz oscillation.

**Three independent growth-rate methods**, as on the kSA 400, all shown at once
so they can be compared:

1. *Windowed FFT* with sub-bin peak refinement — the headline Hz / ML/s number.
2. *Peak counting* — peak count, mean period ± scatter, and fractional damping
   per period. Useful two oscillations into a growth, long before the FFT peak
   means anything, and it exposes period drift that one FFT peak averages away.
   See the known limitation above for noisy traces.
3. *Damped-sine least-squares fit* — the only one that carries a real
   **uncertainty** on the rate, from the fit covariance, plus the 1/e decay
   time and the thickness deposited over the selection. This is the number to
   quote. It declines to fit fewer than ~1.5 periods rather than returning a
   confident wrong answer, and says "no decay resolved" instead of reporting a
   decay constant that has simply run to its bound.

**Lattice constant and strain.** Streaks are located sub-pixel in the ROI
profile and their mean separation Δx converted with `d = n·λ·L/x`. Either
supply the geometry (energy, screen distance, pixel pitch) or click *Calibrate
from pattern* against a known substrate, which fixes `K = a·Δx` and makes the
geometry irrelevant. *Set strain reference* then reports % strain against it.
Note the sign: **narrower** streaks mean a **larger** lattice constant.

**Kinematic pattern simulation.** Full Ewald-sphere construction for a 2-D
surface mesh (a, b, γ, azimuth, incidence angle, beam energy) overlaid on the
live image with the specular spot, Laue arcs and (h,k) labels. Use it to
identify an azimuth or confirm which order you are measuring. *Scale from
streaks* sets px/mm from the measured spacing in one click.

**Saturation indicator** in the status bar. Clipped pixels flatten the tops of
the oscillations and bias both FFT amplitude and any width measurement.

## Display

Colormap selection, fixed-vs-auto contrast with a *Grab* button, and N-frame
rolling averaging. Auto contrast rescales to each frame's own min/max, which
makes recorded video flicker and hides slow drifts — prefer fixed levels when
recording. Frame averaging is display-only unless you tick *Also average
measurements*, since a rolling mean low-pass filters the oscillation.

Zoom, gamma, colormap, contrast and the reference frame are all display-only.
**Every ROI value is measured on the raw frame.**

## Offline analysis

`File ▸ Open Video` replays a recording through the identical pipeline, with
play/pause, seek and speed. Sample timestamps come from the file's own frame
rate, so **growth rates are independent of playback speed**, and since 2.1
every frame is measured whatever the speed. `File ▸ Open Image` loads a still
for profile, FWHM and lattice measurement.

PySpin is an optional import — VRHEED opens and does file analysis on a
machine with no Spinnaker SDK and no camera.

## File formats

CSV keeps the old `Frame, Time_s, ROI_0…` columns and appends a real
`t_ROI_n_s` column per ROI, because ROIs created later genuinely start later.
The `#` metadata header records the app version and log path, beam energy,
electron wavelength, screen geometry, lattice calibration, colormap and
contrast settings.

Saved ROI files (v3) store each ROI's tracked metric and tracking mode, the
lattice calibration, the analysis band and history length, and the Simulate
tab. v1 and v2 files still load.

## Verifying

```bash
python test_analysis.py
QT_QPA_PLATFORM=offscreen python test_app_smoke.py
QT_QPA_PLATFORM=offscreen python test_cameras.py
```

`test_analysis.py` (also collectable by `pytest`) checks the electron
wavelength against the standard 20 keV value, FFT recovery under jitter and
dropped frames, agreement between all three growth-rate methods, the
decimation rule and its amplitude correction, Gaussian and half-max widths
against each other, streak spacing, the SrTiO₃ lattice round trip, strain sign
and magnitude, coherence length, reciprocal-basis orthogonality, and that the
simulated specular spot lands at L·tan θ with (1,0) at λL/a.

`test_app_smoke.py` builds a synthetic video whose spot oscillates at exactly
0.30 Hz, replays it through the real GUI at 10× with no camera and no display,
and checks one sample per frame, the recovered rate, CSV export, session
save/load in all three versions, seek truncation, that a bad frame is logged
once and does not stop the loop, that the excepthook is safe from a worker
thread, and that settings round-trip through a sandboxed `QSettings` file.
`test_cameras.py` covers the camera layer without hardware. A stubbed PySpin
stands in for the Blackfly S on the MBE and checks the whole FLIR sequence
against what 2.1 did: lookup by serial, every auto control off with the SDK's
own integer constants and in the right order, gain in dB and exposure in
microseconds, control ranges rounded inward so a 47.99 dB maximum never comes
back as 48.0, incomplete frames dropped, every image released, binning
resetting the sensor window, and shutdown unwinding EndAcquisition → DeInit →
ReleaseInstance. Beyond that it covers: software binning
against a hand-computed mean, mono conversion of BGR/BGRA/16-bit frames, the
synthetic source actually oscillating at its documented 0.25 Hz, a fake camera
that maxes out at 2× binning still delivering 4×, a control setter that raises
being contained, stream-URL storage with the password kept out of the label,
and — through the real GUI — auto-connect preferring hardware over the
simulator, panel values being pushed to a newly connected camera, switching
cameras releasing the old one, and a camera that refuses to open leaving the
app usable and saying why.

All three exit non-zero on failure, so they can gate a build.

Note the strain sign convention: **narrower streaks mean a larger in-plane
lattice constant**, since `a = K/Δx`.

## Compared with the kSA 400

Not covered by the kSA 400 at all: it drives its own camera and nothing else.
VRHEED works with whatever is on the chamber — see [Cameras](#cameras).

Covered: growth rate (all three of their methods), lattice spacing, strain
evolution, FWHM / in-plane coherence length, multiple simultaneous window
monitoring, window tracking on peak or centroid, per-window statistics, movie
mode re-analysis, real-time palettes and zoom, background subtraction, frame
summation, and CSV/Excel-readable export.

Not implemented: line profiles at arbitrary angle (VRHEED slices are
horizontal), elliptical windows, 3-D surface and contour plots, 2-D FFT and
edge/median image filters, external hardware triggering and rotation
synchronisation, and the LEED/PLE/Auger/electron-gun plug-ins.
