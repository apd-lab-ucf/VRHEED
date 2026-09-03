"""End-to-end smoke test for the VRHEED GUI in file-analysis mode.

Run with ``QT_QPA_PLATFORM=offscreen python test_app_smoke.py`` -- no camera
needed.  It writes a synthetic RHEED video whose spot oscillates at exactly
0.30 Hz, replays it through the real app at 10x, and checks that every frame
became one sample, that the FFT recovers the rate, and that CSV export,
session files, seeking and persistent settings all behave.  Settings and the
log are redirected to a temp dir so the operator's real configuration is
never touched.

Exit status is non-zero on any failure, so it can gate a build.
"""

import os
import sys
import csv
import json
import time
import tempfile
import threading

# Headless Qt and sandboxed settings/log MUST be set before main is imported.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_TMP = tempfile.mkdtemp(prefix="vrheed_smoke_")
os.environ['VRHEED_SETTINGS_FILE'] = os.path.join(_TMP, "settings.ini")
os.environ['VRHEED_LOG_FILE'] = os.path.join(_TMP, "vrheed.log")

try:
    import numpy as np
    import cv2
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QEventLoop
except ImportError as e:
    print(f"SKIP  {e.name or e} not available")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


# --- synthetic RHEED video --------------------------------------------------

FPS, DUR, F0 = 30.0, 40.0, 0.30
W, H = 640, 480
N_FRAMES = int(FPS * DUR)


def make_video(path):
    """Gaussian spot, intensity oscillating at F0 with a slow decay plus noise.

    MJPG in an .avi is the codec OpenCV can both write and read back with a
    reliable frame count on every platform; mp4v often lies about the count.
    """
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W]
    spot = np.exp(-(((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * 18.0 ** 2)))
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    if not wr.isOpened():
        return False
    for k in range(N_FRAMES):
        t = k / FPS
        amp = 150.0 * (0.6 + 0.4 * np.sin(2 * np.pi * F0 * t)) * np.exp(-t / 90.0)
        img = 20.0 + amp * spot + rng.normal(0, 3.0, (H, W))
        img = np.clip(img, 0, 255).astype(np.uint8)
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    wr.release()
    return True


def pump(app, seconds):
    """Run the Qt event loop for ``seconds`` of wall-clock time."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents(QEventLoop.AllEvents, 20)
        time.sleep(0.001)


def pump_until(app, cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents(QEventLoop.AllEvents, 20)
        if cond():
            return True
        time.sleep(0.001)
    return cond()


# ---------------------------------------------------------------------------

video_path = os.path.join(_TMP, "osc.avi")
check("synthetic video written", make_video(video_path))
cap = cv2.VideoCapture(video_path)
n_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
check("cv2 reads back the frame count", n_read == N_FRAMES, f"got {n_read}")

app = QApplication.instance() or QApplication([sys.argv[0]])
import main  # noqa: E402  (after the environment is prepared)
import vrheed_analysis as va  # noqa: E402

main._setup_logging()
main._install_excepthook()

win = main.VRHEED_App()
win.resize(1400, 900)
win.show()
pump(app, 0.2)
check("opens in file-analysis mode without PySpin",
      win.source_mode == 'camera' and win.cam is None)
check("version string present", main.__version__ == "2.1.0", main.__version__)

# --- replay every frame at 10x ----------------------------------------------

win._open_video(video_path)
check("video opened", win.video_cap is not None and win.source_mode == 'file')
check("history length uses the file fps",
      win._history_maxlen() == int(win.hist_spin.value() * 60 * FPS),
      f"got {win._history_maxlen()}")
roi_idx = win._add_roi([0.40, 0.35, 0.60, 0.65])          # around the spot
win.speed_spin.setValue(10.0)
check("play timer uses the fixed tick",
      win.play_timer.interval() == main.PLAY_TICK_MS)

t0 = time.monotonic()
finished = pump_until(app, lambda: not win.play_timer.isActive(), timeout=180)
pump(app, 0.3)                       # let the UI loop drain the tail
elapsed = time.monotonic() - t0
check("playback reached the end of the video", finished, f"{elapsed:.1f} s")

hist = np.array(win.histories[roi_idx], dtype=float)
tim = np.array(win.times[roi_idx], dtype=float)
check("one sample per video frame (no aliasing at 10x)",
      hist.size == N_FRAMES, f"{hist.size} samples for {N_FRAMES} frames")
check("timestamps strictly increasing",
      tim.size > 1 and np.all(np.diff(tim) > 0))
check("timestamps follow the file frame rate",
      tim.size > 1 and abs(np.median(np.diff(tim)) - 1 / FPS) < 1e-6)
check("replay at 10x was actually faster than real time",
      elapsed < DUR, f"{elapsed:.1f} s for a {DUR:.0f} s video")

win._update_plots(force=True)
cached = win._fft_cache.get(roi_idx)
rate = cached[0] if cached else float('nan')
check("FFT growth rate within 3% of 0.30 Hz",
      np.isfinite(rate) and abs(rate - F0) <= 0.03 * F0, f"got {rate:.4f} Hz")
rate_direct, _, _ = va.compute_growth_rate(hist, times=tim, fmin=0.05, fmax=2.0)
check("cached rate matches a direct compute",
      abs(rate - rate_direct) < 1e-9, f"{rate:.5f} vs {rate_direct:.5f}")
check("metric panel shows the rate",
      win.metric_labels['freq'].text() not in ("---", ""),
      win.metric_labels['freq'].text())
# Switching the active ROI must also bypass the peak-count / damped-sine
# throttle, not just the FFT cache, or those two lines lag by 0.5 s.
win._last_stat_time = time.monotonic()
win._sync_metric_combo()
check("selecting an ROI forces the peak-count and fit lines to refresh",
      win._last_stat_time == 0.0 and win._fft_dirty)

# --- CSV export ---------------------------------------------------------------

csv_path = os.path.join(_TMP, "trace.csv")
win._save_csv(csv_path)
rows = []
with open(csv_path, newline='') as f:
    for r in csv.reader(l for l in f if not l.startswith('#')):
        rows.append(r)
check("CSV has header + one row per sample",
      len(rows) == hist.size + 1, f"{len(rows) - 1} data rows")
check("CSV header names the ROI",
      rows and rows[0][:3] == ["Frame", "Time_s", f"ROI_{roi_idx}"])
with open(csv_path) as f:
    head = f.read(4000)
check("CSV metadata carries version and log path",
      f"VRHEED version: {main.__version__}" in head and "Log file:" in head)

# --- seek backwards truncates ----------------------------------------------

seek_to = 600
win._seek_video(seek_to)
pump(app, 0.3)
hist2 = np.array(win.histories[roi_idx], dtype=float)
tim2 = np.array(win.times[roi_idx], dtype=float)
# Playback is stopped (end of video) but not paused, so the frame AT the seek
# position is shown and recorded once: 600 earlier samples + that one.
check("seek backwards truncates the history",
      hist2.size in (seek_to, seek_to + 1), f"{hist2.size} samples left")
check("no sample later than the seek position",
      tim2.size and tim2.max() <= seek_to / FPS + 1e-6, f"max t {tim2.max():.3f}")
check("timestamps still strictly increasing after seek",
      np.all(np.diff(tim2) > 0))
check("frame at the seek position was displayed",
      win.last_raw_frame is not None and win.frame_queue.empty())
check("position readout follows the seek",
      win._video_index in (seek_to, seek_to + 1), str(win._video_index))

# --- session JSON round trip -------------------------------------------------

win.roi_boxes[roi_idx]['metric'] = 'max'
win.sim_a.setValue(4.123)
win._set_sim_origin(0.42, 0.81)
box_before = list(win.roi_boxes[roi_idx]['box'])
sess = os.path.join(_TMP, "rois.json")
win._save_session(sess)
with open(sess) as f:
    payload = json.load(f)
check("session file is version 3", payload.get('version') == 3)
check("session stores the Simulate tab",
      abs(payload['sim']['a'] - 4.123) < 1e-9 and
      payload['sim']['origin'] == [0.42, 0.81])

win.sim_a.setValue(3.0)
win._set_sim_origin(0.5, 0.5)
win._load_session(sess)
check("session round-trips the ROI box and metric",
      len(win.roi_boxes) == 1 and
      np.allclose(win.roi_boxes[0]['box'], box_before) and
      win.roi_boxes[0]['metric'] == 'max')
check("session round-trips the Simulate tab",
      abs(win.sim_a.value() - 4.123) < 1e-9 and
      np.allclose(win._sim_origin, (0.42, 0.81)))

# v1 (bare lists) and v2 (no sim block) files must still load.
for ver, rois in ((1, [[0.1, 0.1, 0.3, 0.3]]),
                  (2, [{'box': [0.1, 0.1, 0.3, 0.3], 'metric': 'fwhm'}])):
    p = os.path.join(_TMP, f"v{ver}.json")
    with open(p, 'w') as f:
        json.dump({'version': ver, 'rois': rois}, f)
    win._load_session(p)
    check(f"loads a v{ver} session", len(win.roi_boxes) == 1)

# --- crash-proofing --------------------------------------------------------

win._clear_rois()
win.frame_queue.put(("not a frame", 0.0))
win._ui_loop()                       # must not raise
win.frame_queue.put(("not a frame", 0.0))
win._ui_loop()
check("bad frame does not kill the UI loop", True)
check("bad frame logged once", len(win._logged_errors) == 1,
      f"{len(win._logged_errors)} distinct")
with open(os.environ['VRHEED_LOG_FILE']) as f:
    log_text = f.read()
check("traceback written to the log",
      "frame pipeline" in log_text and "Traceback" in log_text)
check("excepthook installed", sys.excepthook is main._excepthook)
try:
    main._excepthook(RuntimeError, RuntimeError("synthetic"), None)
    check("excepthook does not abort", True)
except BaseException as e:            # pragma: no cover
    check("excepthook does not abort", False, repr(e))
# The same hook serves threading.excepthook; from a worker thread it must log
# and return without touching any widget.
win.statusBar().showMessage("sentinel", 0)
th = threading.Thread(target=main._excepthook,
                      args=(RuntimeError, RuntimeError("from a worker thread"), None))
th.start(); th.join(5.0)
pump(app, 0.05)
check("excepthook from a worker thread leaves the GUI alone",
      not th.is_alive() and win.statusBar().currentMessage() == "sentinel",
      win.statusBar().currentMessage())
with open(os.environ['VRHEED_LOG_FILE']) as f:
    check("worker-thread exception still reaches the log",
          "from a worker thread" in f.read())

# --- persistent settings -----------------------------------------------------

win.energy_spin.setValue(15.0)
win.cmap_combo.setCurrentIndex(3)
win.close()
pump(app, 0.1)
win2 = main.VRHEED_App()
check("QSettings round-trips a changed energy",
      abs(win2.energy_spin.value() - 15.0) < 1e-9, f"got {win2.energy_spin.value()}")
check("QSettings round-trips the colormap", win2.cmap_combo.currentIndex() == 3)
check("QSettings round-trips the shadow-edge origin",
      np.allclose(win2._sim_origin, (0.42, 0.81)), str(win2._sim_origin))
win2._reset_settings()
check("reset restores the default energy", abs(win2.energy_spin.value() - 20.0) < 1e-9)
check("reset restores the default origin", win2._sim_origin == (0.5, 0.75))
win2.close()
check("settings live in the sandbox, not the user's profile",
      os.path.exists(os.environ['VRHEED_SETTINGS_FILE']))

# ---------------------------------------------------------------------------

print()
if _fails:
    print(f"{len(_fails)} FAILED: " + ", ".join(_fails))
    sys.exit(1)
print(f"all checks passed  ({_TMP})")
