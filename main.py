import sys
import os

# Pin the BLAS/LAPACK thread pools to one thread *before* numpy is imported
# (the pools are sized once, at load).  Every array this app hands to BLAS is
# small -- a few hundred samples for the damped-sine fit, a few thousand for
# the FFT -- and for those MKL/OpenBLAS thread start-up and hand-off costs
# more than the arithmetic.  Worse, under machine load the multi-threaded SVD
# inside curve_fit oversubscribes the cores and stalls for whole seconds,
# which freezes the UI.  Single-thread BLAS is faster here and never stalls.
# setdefault() so an operator who really wants more threads can still set
# the variable in their environment.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
del _v

# Also before cv2 is imported.  Scanning for USB cameras makes OpenCV log a
# warning per index that is not one; they are normal and they frighten people.
# ERROR keeps everything that matters.  See vrheed_cameras for the detail.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import csv
import json
import logging
import logging.handlers
import platform
import threading
import queue
import time
import traceback
from datetime import datetime
from collections import deque

__version__ = "2.2.0"


def _installed_backends():
    """Comma-separated names of the camera drivers present, for the About box."""
    names = [name for name, ok, _ in vcam.backend_status() if ok]
    return ", ".join(names) if names else "none"


def _resource(filename):
    """Return path to a bundled resource, works both frozen (PyInstaller) and from source."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)


def _app_dir():
    """Directory the user can actually find: next to the .exe when frozen,
    next to this script from source.  sys._MEIPASS is a throwaway temp dir
    that PyInstaller deletes on exit, so a log written there would vanish
    with the crash it was meant to explain."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


LOG_PATH = os.environ.get('VRHEED_LOG_FILE') or os.path.join(_app_dir(), "vrheed.log")
logger = logging.getLogger("vrheed")


def _setup_logging():
    """Rotating file log (~1 MB x 3) next to the executable.

    Idempotent, so the smoke test and a second construction of the window do
    not stack handlers and duplicate every line.
    """
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    try:
        h = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    except OSError:
        # Read-only install directory: stderr is better than nothing, and it
        # still keeps the excepthook from being a silent no-op.
        h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)


# The live window, so the excepthook can put the error where the operator is
# looking.  Set by VRHEED_App.__init__.
_MAIN_WINDOW = None


def _excepthook(exc_type, exc, tb):
    """Log an unhandled exception and carry on.

    PyQt5 >= 5.5 calls qFatal() -- i.e. aborts the whole process -- when a
    Python exception escapes a slot or timer callback and sys.excepthook is
    still the default.  One malformed frame would kill the app mid-growth
    with no trace.  Replacing the hook is what turns that into a log line.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    logger.error("Unhandled exception:\n%s", text)
    msg = str(exc).splitlines()
    first = f"{exc_type.__name__}: {msg[0]}" if msg else exc_type.__name__
    # Only the GUI thread may touch widgets: this hook also serves as
    # threading.excepthook for the camera grab thread, and a showMessage()
    # from there would be exactly the kind of crash it exists to prevent.
    win = _MAIN_WINDOW
    if win is not None and threading.current_thread() is threading.main_thread():
        try:
            win.statusBar().showMessage(
                f"Error (details in {os.path.basename(LOG_PATH)}): {first[:160]}",
                8000)
        except Exception:
            pass


def _install_excepthook():
    sys.excepthook = _excepthook
    # Background threads (the camera grab loop) report through their own hook.
    if hasattr(threading, 'excepthook'):
        threading.excepthook = lambda a: _excepthook(
            a.exc_type, a.exc_value, a.exc_traceback)


def _missing_dependency(exc):
    """Turn a bare ImportError into something the operator can act on.

    A raw "ModuleNotFoundError: No module named 'pyqtgraph'" tells someone who
    just wants to look at a RHEED pattern nothing at all.  Name the package as
    pip spells it -- cv2 is opencv-python, which nobody guesses -- and give the
    one command that fixes every case.
    """
    name = getattr(exc, 'name', None) or str(exc)
    pip_name = {'cv2': 'opencv-python', 'np': 'numpy'}.get(name, name)
    return (
        f"\nVRHEED cannot start: the Python package '{pip_name}' is not installed.\n\n"
        f"  Python in use: {sys.executable}\n\n"
        "Install everything VRHEED needs with:\n\n"
        "    pip install -r requirements.txt\n\n"
        "If you made a virtual environment, activate it FIRST, or pip installs\n"
        "into a different Python than the one running this file:\n\n"
        "    PowerShell     .\\.venv\\Scripts\\Activate.ps1\n"
        "    Command Prompt .venv\\Scripts\\activate.bat\n")


try:
    import cv2
    import numpy as np
except ImportError as _e:
    raise SystemExit(_missing_dependency(_e))

# Camera support lives in its own module: FLIR/Spinnaker, Basler pylon,
# Allied Vision Vimba, any GenICam camera through a GenTL producer, scientific
# cameras and frame grabbers through pylablib, USB/UVC devices, network
# streams, screen capture, and a simulated source for demos and tests.  Every
# vendor driver is imported lazily inside its own backend, so VRHEED starts
# with none of them installed -- which is how you work on a laptop away from
# the MBE, analysing recorded video and images.
import vrheed_cameras as vcam
from vrheed_cameras import CameraError

import vrheed_analysis as va

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QSplitter,
        QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
        QSlider, QPushButton, QRadioButton, QButtonGroup,
        QFileDialog, QMessageBox, QSizePolicy, QDoubleSpinBox,
        QCheckBox, QComboBox, QTabWidget, QSpinBox, QStatusBar, QAction,
        QDialog, QDialogButtonBox, QInputDialog,
    )
    from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QSettings, QByteArray
    from PyQt5.QtGui import QImage, QPixmap, QIcon, QGuiApplication

    import pyqtgraph as pg
except ImportError as _e:
    raise SystemExit(_missing_dependency(_e))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Smallest ROI edge, as a fraction of the frame.  Anything smaller measures
# noise rather than a diffraction feature and makes the profile fit unstable.
MIN_ROI_FRAC = 0.005

# Quantities a single ROI can track over time.  The key is stored in the ROI
# dict and in saved sessions, so do not rename them casually.
ROI_METRICS = [
    ('mean',    'Mean intensity'),
    ('max',     'Peak intensity'),
    ('min',     'Min intensity'),
    ('sum',     'Integrated intensity'),
    ('std',     'Intensity std dev'),
    ('cx',      'Centroid X (ROI fraction)'),
    ('cy',      'Centroid Y (ROI fraction)'),
    ('px',      'Peak pixel X (ROI fraction)'),
    ('py',      'Peak pixel Y (ROI fraction)'),
    ('fwhm',    'Profile FWHM (px)'),
    ('coh',     'Coherence length (Å)'),
    ('pos',     'Profile position (px)'),
    ('spacing', 'Streak spacing (px)'),
]
ROI_METRIC_LABEL = dict(ROI_METRICS)

# Window tracking modes, after the kSA "Tracking" feature: keep the ROI
# centred on the feature as it moves during growth.
TRACK_MODES = [('off', 'Off'), ('peak', 'Brightest pixel'), ('centroid', 'Centroid')]
TRACK_LABEL = dict(TRACK_MODES)

# Fraction of the position error corrected per frame.  Correcting fully makes
# the box chase shot noise; this settles in a few frames without jitter.
TRACK_GAIN = 0.35

# Binning the app starts at.  4x is the sensible default for RHEED: the
# pattern is a handful of broad streaks, not fine detail, so the resolution
# costs nothing you were using, while each output pixel collects 16x the
# photons and the camera delivers frames faster.  That buys a cleaner
# intensity trace, which is the measurement everything else rests on.
#
# It is applied through CameraBackend.set_binning, so a sensor that caps out
# below 4 gets the rest in software and still ends up at 4x.
DEFAULT_BINNING = 4

# Frames waiting between the source and the UI loop.  The UI loop measures
# EVERY queued frame each tick and draws only the last, so a deeper queue does
# not add display latency; it is what lets 20x video playback keep every
# sample instead of aliasing.  The camera thread still drops the oldest frame
# when it is full, so a stalled UI never builds a backlog of live frames.
FRAME_QUEUE_SIZE = 16

# Video playback timer period.  A fixed tick with a variable number of frames
# decoded per tick works at any speed; the old 1000/(fps*speed) interval hit
# Qt's ~1 ms floor at 20x and simply ran slower than asked.
PLAY_TICK_MS = 15

# Spectrum / metric panel refresh period.  The FFT of a 30 min trace for every
# ROI costs more than the whole frame pipeline; four times a second is faster
# than anyone reads the numbers.
FFT_REFRESH_S = 0.25

COLORMAPS = [
    ('Plasma',  cv2.COLORMAP_PLASMA),
    ('Inferno', cv2.COLORMAP_INFERNO),
    ('Viridis', cv2.COLORMAP_VIRIDIS),
    ('Magma',   cv2.COLORMAP_MAGMA),
    ('Hot',     cv2.COLORMAP_HOT),
    ('Jet',     cv2.COLORMAP_JET),
    ('Bone',    cv2.COLORMAP_BONE),
    ('Grayscale', None),
]

# subcontrol-origin/left/padding keep the title from sitting on top of the
# border and clipping its own first and last letters.
GROUP_QSS = ("QGroupBox{color:white;border:1px solid #7f8c8d;border-radius:4px;"
             "margin-top:10px;padding-top:6px;}"
             "QGroupBox::title{color:white;subcontrol-origin:margin;"
             "left:8px;padding:0 4px;}")


# ---------------------------------------------------------------------------
# Camera display widget
# ---------------------------------------------------------------------------

class CameraLabel(QLabel):
    """QLabel that forwards mouse events as signals for ROI drawing."""
    sig_press    = pyqtSignal(int, int, int)   # x, y, Qt.MouseButton
    sig_move     = pyqtSignal(int, int)
    sig_release  = pyqtSignal(int, int)
    sig_wheel    = pyqtSignal(int, int, int)   # x, y, angleDelta
    sig_dblclick = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background: black;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Return a fixed size hint so Qt's layout never tries to grow the label
    # to match the pixmap — prevents the infinite-resize feedback loop.
    def sizeHint(self):        return QSize(800, 600)
    def minimumSizeHint(self): return QSize(320, 240)

    def mousePressEvent(self, e):
        self.sig_press.emit(e.x(), e.y(), int(e.button()))

    def mouseMoveEvent(self, e):
        self.sig_move.emit(e.x(), e.y())

    def mouseReleaseEvent(self, e):
        self.sig_release.emit(e.x(), e.y())

    def mouseDoubleClickEvent(self, e):
        self.sig_dblclick.emit(e.x(), e.y())

    def wheelEvent(self, e):
        self.sig_wheel.emit(e.x(), e.y(), e.angleDelta().y())


class ScreenRegionDialog(QDialog):
    """Pick the rectangle of the desktop the screen-capture source reads.

    Coordinates are relative to the chosen monitor's top-left corner, which is
    what a user reads off a screenshot; the backend adds the monitor's own
    offset.  "Whole screen" is there because the first thing anyone does is
    try it, see the toolbars, and then want to crop.
    """

    def __init__(self, parent, monitor_index, region=None):
        super().__init__(parent)
        self.setWindowTitle(f"Screen {monitor_index} capture region")
        mon_w, mon_h = self._monitor_size(monitor_index)
        x, y, w, h = region or (0, 0, mon_w, mon_h)

        grid = QGridLayout()
        self._spins = {}
        for row, (key, label, val, hi) in enumerate((
                ('x', "Left (px):",   x, mon_w), ('y', "Top (px):",    y, mon_h),
                ('w', "Width (px):",  w, mon_w), ('h', "Height (px):", h, mon_h))):
            grid.addWidget(QLabel(label), row, 0, alignment=Qt.AlignRight)
            sp = QSpinBox()
            sp.setRange(1 if key in ('w', 'h') else 0, max(hi, 1))
            sp.setValue(int(val))
            grid.addWidget(sp, row, 1)
            self._spins[key] = sp

        btn_full = QPushButton("Whole screen")
        btn_full.clicked.connect(lambda: self._set(0, 0, mon_w, mon_h))
        grid.addWidget(btn_full, 4, 0, 1, 2)

        hint = QLabel(
            "Frame the live-image pane of whatever program owns the camera.\n"
            "You are measuring its display, so relative changes are valid and\n"
            "absolute intensities are not.")
        hint.setStyleSheet("color:#7f8c8d;")
        grid.addWidget(hint, 5, 0, 1, 2)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        grid.addWidget(buttons, 6, 0, 1, 2)
        self.setLayout(grid)

    @staticmethod
    def _monitor_size(index):
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[index]
                return int(mon['width']), int(mon['height'])
        except Exception:
            return 3840, 2160

    def _set(self, x, y, w, h):
        for key, val in zip(('x', 'y', 'w', 'h'), (x, y, w, h)):
            self._spins[key].setValue(int(val))

    def region(self):
        return tuple(self._spins[k].value() for k in ('x', 'y', 'w', 'h'))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class VRHEED_App(QMainWindow):

    # ROI colors as (R, G, B) tuples — used for both OpenCV overlay and pg pens
    ROI_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    COLOR_NAMES = ['yellow', 'cyan', 'magenta', 'green']

    def __init__(self):
        super().__init__()
        global _MAIN_WINDOW
        _MAIN_WINDOW = self
        logger.info("VRHEED %s starting - Python %s, numpy %s, OpenCV %s, "
                    "frozen=%s, log=%s",
                    __version__, platform.python_version(), np.__version__,
                    cv2.__version__,
                    bool(getattr(sys, 'frozen', False)), LOG_PATH)
        logger.info("Camera backends: %s",
                    ", ".join(f"{n}={'yes' if ok else 'no'}"
                              for n, ok, _ in vcam.backend_status()))
        self.setWindowTitle("VRHEED - Precision Analyzer")
        self.setWindowIcon(QIcon(_resource("turtle.ico")))

        # Open at 1600x950 but never larger than the screen actually available,
        # otherwise the window opens with its controls off the bottom edge.
        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        w = min(1600, avail.width() - 40) if avail else 1600
        h = min(950, avail.height() - 60) if avail else 950
        self.resize(max(w, 1000), max(h, 650))
        self.setMinimumSize(1000, 650)

        # Camera.  self.cam is a vrheed_cameras.CameraBackend once connected,
        # None otherwise; nothing outside _connect_camera / _disconnect_camera
        # assigns it, so "is there a camera?" is one check everywhere else.
        self.cam = None
        self.cam_info = None            # CameraInfo of the connected camera
        self._cameras = []              # everything the last scan found
        self._last_cam_key = ""         # key to reconnect to next session
        self.source_mode = 'camera'     # 'camera' | 'file'

        # ROI state — boxes stored as NORMALIZED coords (0-1 fraction of frame).
        # This makes them independent of resolution/binning changes.
        self.roi_boxes    = []    # list of {'box','slice_y','metric'}
        self.histories    = {}    # idx → deque(float)   measured value
        self.times        = {}    # idx → deque(float)   seconds since t-origin
        self.active_roi   = -1
        self.drag_mode    = None
        self.start_mouse  = (0, 0)
        self.temp_box     = None  # [x1,y1,x2,y2] in display coords while drawing
        self._roi_cache   = {}    # idx → last computed profile/metric bundle

        # Display transform — updated every frame, used to convert mouse↔norm
        self._off_x  = 0;   self._off_y  = 0    # letterbox offset (display px)
        self._new_w  = 1;   self._new_h  = 1    # scaled frame size in display px
        self._frame_w = 1;  self._frame_h = 1   # raw frame dimensions

        # Time-plot x-range state.  Sample times are REAL seconds measured from
        # _t_origin, not frame_index/fps — see _append_sample.
        self._t_origin = None     # source timestamp when the first ROI appeared
        self._x_right  = 60.0     # current right edge of x axis (seconds)

        # Zoom state
        self._zoom_level = 1.0      # 1.0 = fit-to-window, >1 = zoomed in
        self._zoom_cx    = 0.5      # viewport center in normalized full-frame coords
        self._zoom_cy    = 0.5
        # Viewport in normalized full-frame coords (updated every frame)
        self._vp_x1 = 0.0;  self._vp_y1 = 0.0
        self._vp_w  = 1.0;  self._vp_h  = 1.0

        # Rotation
        self._rotation_deg = 0          # base rotation: 0 / 90 / 180 / 270

        # Plot curves (created lazily per ROI)
        self.time_curves  = {}
        self.fft_curves   = {}
        self.peak_markers = {}   # idx → ScatterPlotItem at the detected peak
        self.selected_range = (0.0, 0.0)   # absolute seconds

        # Slice plot items (created lazily per ROI)
        self.slice_curves = {}   # idx → PlotDataItem
        self.fwhm_items   = {}   # idx → {'half_line': InfiniteLine, 'text': TextItem}
        self.streak_lines = []   # InfiniteLines marking detected streaks

        # Threading / frame pipeline
        self.frame_queue      = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._stop_event      = threading.Event()
        self._cap_thread      = None
        self.last_raw_frame   = None
        self.reference_frame  = None
        self._last_frame_time = None
        self.fps_tracker      = deque(maxlen=30)
        self.actual_fps       = 30.0
        self.frame_count      = 0
        self.paused           = False
        self._gamma_lut       = None   # cached LUT, rebuilt only when gamma changes
        self._gamma_cached    = None
        self._raw_max         = 255.0  # full-scale ADU of the current pixel format
        self._avg_frames      = deque(maxlen=1)
        # Running sum of the frames in _avg_frames (float32), so the rolling
        # mean costs one add and one subtract per frame instead of re-summing
        # up to 32 full frames every tick.  See _averaged_frame.
        self._avg_sum         = None
        self._avg_count       = 0
        # Errors already written to the log by _log_once, keyed by message.
        self._logged_errors   = set()

        # Video recording / playback
        self.video_writer        = None
        self.is_recording        = False
        self._rec_size           = None   # (w, h) the writer was opened with
        self._last_display_frame = None   # camera-res BGR frame for snap/record
        self.video_cap           = None
        self.video_fps           = 30.0
        self.video_nframes       = 0
        self._video_index        = 0
        self._play_budget        = 0.0    # fractional frames owed to playback
        self._last_dir           = ""     # starting folder for file dialogs

        # Lattice metrology
        # Throttled analysis results for the metrics panel
        self._last_stat_time = 0.0
        self._stat_cache  = {}
        self._fit_cache   = {}
        # Spectrum cache: idx -> (rate, mag, freq, sub_t, sub_v), refreshed at
        # most every FFT_REFRESH_S or immediately when _fft_dirty is set.
        self._fft_cache      = {}
        self._last_fft_time  = 0.0
        self._fft_dirty      = True

        self._lattice_K   = None    # calibration constant (A * px)
        self._a_ref       = None    # reference lattice constant for strain
        self._last_spacing_px = float('nan')
        self._last_a      = float('nan')

        # Simulation overlay
        self._sim_pattern = []
        self._sim_dirty   = True
        self._pick_origin = False

        self._setup_ui()
        # Snapshot the built-in defaults before user settings overwrite them,
        # so "Reset settings" has something to go back to.  Restore happens
        # while there are no ROIs, so nothing a spin box triggers can clear
        # history.
        self._defaults = self._settings_snapshot()
        self._restore_settings()
        self._init_camera()

        # UI refresh at ~60 fps — reads from frame_queue, updates display + plots
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._ui_loop)
        self.ui_timer.start(16)

        # Playback timer for file mode (started only when a video is open)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._feed_video_frame)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _setup_ui(self):
        pg.setConfigOption('background', '#2c3e50')
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOption('antialias', True)

        # Create the status bar up front.  Creating it lazily inside a handler
        # makes every widget jump a few pixels the first time a message fires.
        self.setStatusBar(QStatusBar(self))
        self.fps_display = QLabel("FPS: ---")
        self.fps_display.setStyleSheet("color:#1abc9c;")
        self.sat_display = QLabel("Sat: ---")
        self.sat_display.setToolTip(
            "Fraction of pixels within 1% of full scale.\n"
            "Saturated pixels clip the tops of the oscillations and bias\n"
            "both the FFT amplitude and any width measurement.")
        self.statusBar().addPermanentWidget(self.sat_display)
        self.statusBar().addPermanentWidget(self.fps_display)

        self._build_menus()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        splitter.setHandleWidth(8)
        # Collapsible panes let a stray double-click on the handle hide the
        # camera or the dashboard entirely, with no obvious way back.
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #7f8c8d; border-radius: 3px; }"
            "QSplitter::handle:hover { background: #1abc9c; }")
        root_layout.addWidget(splitter)

        # ── Left: camera view (top) + slice plot (bottom) ─────────────────
        left_split = QSplitter(Qt.Vertical)
        self.left_splitter = left_split
        left_split.setHandleWidth(6)
        left_split.setChildrenCollapsible(False)
        left_split.setStyleSheet(
            "QSplitter::handle { background: #7f8c8d; border-radius: 3px; }"
            "QSplitter::handle:hover { background: #1abc9c; }")

        self.cam_label = CameraLabel()
        self.cam_label.sig_press.connect(self._on_mouse_down)
        self.cam_label.sig_move.connect(self._on_mouse_move)
        self.cam_label.sig_release.connect(self._on_mouse_up)
        self.cam_label.sig_wheel.connect(self._on_wheel)
        self.cam_label.sig_dblclick.connect(lambda *_: self._reset_zoom())
        left_split.addWidget(self.cam_label)

        slice_container = QWidget()
        slice_container.setStyleSheet("background:#2c3e50;")
        slice_container.setMinimumHeight(120)
        slice_vlay = QVBoxLayout(slice_container)
        slice_vlay.setContentsMargins(0, 0, 0, 0)
        slice_vlay.setSpacing(0)

        slice_ctrl_lay = QHBoxLayout()
        slice_ctrl_lay.setContentsMargins(4, 2, 4, 0)
        self.slice_logy_cb = QCheckBox("Log Y")
        self.slice_fit_cb = QCheckBox("Gaussian fit")
        self.slice_fit_cb.setChecked(True)
        self.slice_fit_cb.setToolTip(
            "Fit a Gaussian-on-a-pedestal to each ROI profile.\n"
            "Sub-pixel and far less noise-sensitive than half-maximum\n"
            "crossings; falls back to half-max when the fit fails.")
        self.slice_streak_cb = QCheckBox("Mark streaks")
        self.slice_streak_cb.setToolTip(
            "Detect every streak in the active ROI profile and report their\n"
            "mean separation, which feeds the Lattice tab.")
        slice_ctrl_lay.addWidget(self.slice_logy_cb)
        slice_ctrl_lay.addWidget(self.slice_fit_cb)
        slice_ctrl_lay.addWidget(self.slice_streak_cb)
        slice_ctrl_lay.addStretch()
        self.slice_info = QLabel("")
        self.slice_info.setStyleSheet("color:#bdc3c7;")
        slice_ctrl_lay.addWidget(self.slice_info)
        slice_vlay.addLayout(slice_ctrl_lay)

        self.plot_slice = pg.PlotWidget(title="Horizontal Slice  (drag ─── line in image)")
        self.plot_slice.setStyleSheet("background:#2c3e50;")
        self.plot_slice.showGrid(x=True, y=True, alpha=0.3)
        self.plot_slice.setLabel('bottom', 'Pixel (horizontal)')
        self.plot_slice.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.slice_logy_cb.toggled.connect(
            lambda v: self.plot_slice.setLogMode(y=v))
        slice_vlay.addWidget(self.plot_slice)

        left_split.addWidget(slice_container)
        left_split.setSizes([750, 200])

        splitter.addWidget(left_split)

        # ── Right: dashboard ───────────────────────────────────────────────
        dash = QWidget()
        dash.setStyleSheet("background:#2c3e50; color:white;")
        dash.setMinimumWidth(400)
        dash_layout = QVBoxLayout(dash)
        dash_layout.setContentsMargins(8, 8, 8, 8)
        dash_layout.setSpacing(5)
        splitter.addWidget(dash)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1140, 460])

        # Controls live in tabs.  Stacking every group vertically made the
        # dashboard taller than the window as soon as the display and lattice
        # controls were added, which pushed the plots off the bottom.
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #7f8c8d;border-radius:4px;}"
            "QTabBar::tab{background:#34495e;color:white;padding:4px 8px;}"
            "QTabBar::tab:selected{background:#1abc9c;color:#2c3e50;font-weight:bold;}")
        self.tabs.addTab(self._build_camera_tab(),   "Camera")
        self.tabs.addTab(self._build_display_tab(),  "Display")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_lattice_tab(),  "Lattice")
        self.tabs.addTab(self._build_sim_tab(),      "Simulate")
        self.tabs.setMinimumHeight(230)
        # 2.2: raised from 300 to fit the Source chooser on the Camera tab.
        # The cap is what stops the control dashboard growing until the plots
        # are pushed off the bottom of the window, so it is deliberately only
        # as large as the tallest tab now needs.
        self.tabs.setMaximumHeight(340)
        dash_layout.addWidget(self.tabs)

        dash_layout.addWidget(self._build_metrics_group())

        # Intensity vs Time plot ───────────────────────────────────────────
        time_ctrl = QHBoxLayout()
        self.time_logy_cb = QCheckBox("Log Y")
        self.time_norm_cb = QCheckBox("Normalize")
        self.time_norm_cb.setToolTip(
            "Divide each ROI by its maximum value so the brightest point is 1.0\n"
            "and all values show intensity relative to the peak")
        self.time_bg_cb = QCheckBox("Subtract BG")
        self.time_bg_cb.setToolTip(
            "Subtract the minimum value of each ROI trace to remove the\n"
            "baseline offset (applied before normalization if both are checked)")
        btn_fitx = QPushButton("Fit X")
        btn_fitx.setFixedWidth(48)
        btn_fitx.setToolTip("Set the time axis to span all recorded data")
        btn_fitx.clicked.connect(self._fit_x_range)
        time_ctrl.addWidget(self.time_logy_cb)
        time_ctrl.addWidget(self.time_norm_cb)
        time_ctrl.addWidget(self.time_bg_cb)
        time_ctrl.addStretch()
        time_ctrl.addWidget(btn_fitx)
        dash_layout.addLayout(time_ctrl)

        self.plot_time = pg.PlotWidget(title="Intensity vs Time")
        self.plot_time.showGrid(x=True, y=True, alpha=0.3)
        self.plot_time.setLabel('bottom', 'Time', units='s')
        self.plot_time.setLabel('left', 'Mean intensity')
        self.plot_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_time.enableAutoRange(axis='x', enable=False)
        # A 30 min history at 30 fps is 54k points per ROI; peak-mode
        # downsampling keeps the redraw cheap without hiding the oscillations.
        self.plot_time.setDownsampling(auto=True, mode='peak')
        self.plot_time.setClipToView(True)
        self.time_logy_cb.toggled.connect(
            lambda v: self.plot_time.setLogMode(y=v))
        # Draggable region selector (replaces matplotlib SpanSelector)
        self.region = pg.LinearRegionItem(
            brush=pg.mkBrush(26, 188, 156, 40),
            pen=pg.mkPen('#1abc9c', width=1))
        self.region.setZValue(10)
        self.plot_time.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        dash_layout.addWidget(self.plot_time, stretch=2)

        # FFT Spectrum plot ────────────────────────────────────────────────
        fft_ctrl = QHBoxLayout()
        self.fft_logy_cb = QCheckBox("Log Y")
        fft_ctrl.addWidget(self.fft_logy_cb)
        fft_ctrl.addStretch()
        dash_layout.addLayout(fft_ctrl)

        self.plot_fft = pg.PlotWidget(title="FFT Spectrum")
        self.plot_fft.showGrid(x=True, y=True, alpha=0.3)
        self.plot_fft.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_fft.setLabel('bottom', 'Frequency', units='Hz')
        # Lock x-range to [0, fmax] so the view doesn't change with fps
        self.plot_fft.setXRange(0, self.fmax_spin.value(), padding=0)
        self.plot_fft.enableAutoRange(axis='x', enable=False)
        self.fft_logy_cb.toggled.connect(
            lambda v: self.plot_fft.setLogMode(y=v))
        self.fmax_spin.valueChanged.connect(
            lambda v: self.plot_fft.setXRange(0, v, padding=0))
        dash_layout.addWidget(self.plot_fft, stretch=2)

        # Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setStyleSheet("background:#f39c12;color:white;font-weight:bold;")
        self.btn_pause.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.btn_pause)
        for label, color, slot, tip in [
                ("CLEAR ALL", "#c0392b", self._clear_rois,
                 "Delete all ROIs and their history"),
                ("SAVE CSV",  "#2980b9", self._save_csv,
                 "Save the tracked value for ALL ROIs.\n"
                 "Columns: Frame, Time_s, ROI_0…, then a real timestamp\n"
                 "column per ROI.  ROIs created later start later in time."),
                ("Save ROIs", "#16a085", self._save_session,
                 "Save ROI positions, tracked metrics and lattice calibration"),
                ("Load ROIs", "#16a085", self._load_session,
                 "Restore ROI positions and settings from a JSON file")]:
            b = QPushButton(label)
            b.setStyleSheet(f"background:{color};color:white;")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        dash_layout.addLayout(btn_row)

        # Capture buttons ──────────────────────────────────────────────────
        cap_row = QHBoxLayout()
        btn_snap = QPushButton("SNAP IMAGE")
        btn_snap.setStyleSheet("background:#8e44ad;color:white;font-weight:bold;")
        btn_snap.clicked.connect(self._snap_image)
        cap_row.addWidget(btn_snap)
        self.btn_record = QPushButton("REC VIDEO")
        self.btn_record.setStyleSheet("background:#c0392b;color:white;font-weight:bold;")
        self.btn_record.clicked.connect(self._toggle_recording)
        cap_row.addWidget(self.btn_record)
        dash_layout.addLayout(cap_row)

        # Playback bar — hidden until a video is loaded
        self.play_bar = QWidget()
        pb = QHBoxLayout(self.play_bar)
        pb.setContentsMargins(0, 0, 0, 0)
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(34)
        self.btn_play.clicked.connect(self._toggle_playback)
        pb.addWidget(self.btn_play)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self._seek_video)
        pb.addWidget(self.seek_slider, 1)
        pb.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 20.0); self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.5); self.speed_spin.setFixedWidth(60)
        self.speed_spin.setToolTip(
            "Playback speed only — sample timestamps always come from the\n"
            "file's own frame rate, so growth rates are unaffected.")
        self.speed_spin.valueChanged.connect(self._apply_playback_speed)
        pb.addWidget(self.speed_spin)
        self.play_pos_label = QLabel("0 / 0")
        pb.addWidget(self.play_pos_label)
        self.play_bar.setVisible(False)
        dash_layout.addWidget(self.play_bar)

    # -- menus ---------------------------------------------------------------

    def _build_menus(self):
        mb = self.menuBar()
        mb.setStyleSheet(
            "QMenuBar{background:#34495e;color:white;}"
            "QMenuBar::item:selected{background:#1abc9c;color:#2c3e50;}"
            "QMenu{background:#34495e;color:white;}"
            "QMenu::item:selected{background:#1abc9c;color:#2c3e50;}")

        def act(menu, text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            if tip:
                a.setStatusTip(tip)
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        m = mb.addMenu("&File")
        act(m, "Open &Video for analysis…", self._open_video, "Ctrl+O",
            "Replay a recorded growth through the full analysis pipeline")
        act(m, "Open &Image…", self._open_image, "Ctrl+I",
            "Load a still RHEED image for profile and lattice measurement")
        act(m, "Return to &Camera", self._return_to_camera, None,
            "Close the file source and resume live acquisition")
        m.addSeparator()
        act(m, "&Snap image…", self._snap_image, "Ctrl+S")
        act(m, "&Record video…", self._toggle_recording, "Ctrl+R")
        act(m, "Save &CSV…", self._save_csv, "Ctrl+E")
        m.addSeparator()
        act(m, "Save ROIs…", self._save_session)
        act(m, "Load ROIs…", self._load_session)
        m.addSeparator()
        act(m, "E&xit", self.close, "Ctrl+Q")

        m = mb.addMenu("&Camera")
        act(m, "&Rescan for cameras", lambda: self._refresh_cameras(), "F5",
            "Look for cameras again after plugging one in")
        act(m, "&Disconnect camera", self._disconnect_camera, None,
            "Release the camera so another program can use it")
        m.addSeparator()
        act(m, "Add &network camera…", self._add_network_camera, None,
            "Add an RTSP or HTTP-MJPEG stream by URL")
        act(m, "Remove network camera…", self._remove_network_camera, None,
            "Forget a stream URL added earlier")
        act(m, "&Screen capture region…", self._set_screen_region, None,
            "Choose which part of the desktop the Screen source reads")
        m.addSeparator()
        act(m, "Driver settings…", self._open_driver_dialog, None,
            "The camera driver's own dialog, where it has one")

        m = mb.addMenu("&View")
        act(m, "&Pause / Resume", self._toggle_pause, "Space")
        act(m, "&Reset zoom", self._reset_zoom, "Ctrl+0")
        act(m, "Rotate 90° CW", self._on_rotate_cw, "Ctrl+]")
        act(m, "Rotate 90° CCW", self._on_rotate_ccw, "Ctrl+[")
        m.addSeparator()
        act(m, "Fit time axis to data", self._fit_x_range)

        m = mb.addMenu("&ROI")
        act(m, "&Clear all ROIs", self._clear_rois, "Ctrl+Shift+C")
        act(m, "Delete active ROI", self._delete_active_roi, "Del")

        m = mb.addMenu("&Settings")
        act(m, "&Reset settings to defaults", self._reset_settings, None,
            "Forget the saved geometry, display, analysis and simulation "
            "settings and go back to the built-in values")
        act(m, "Open &log file location", self._show_log_location, None,
            "Where vrheed.log lives")

        m = mb.addMenu("&Help")
        act(m, "Mouse && keyboard…", self._show_help, "F1")
        act(m, "Camera &backends…", self._show_backends, None,
            "Which camera drivers are installed, and how to add the rest")
        act(m, "About VRHEED", self._show_about)

    # -- tabs ----------------------------------------------------------------

    def _build_camera_tab(self):
        page = QWidget()
        cfg = QGridLayout(page)
        # 2 rather than the 4 the other tabs use: the Source group added in 2.2
        # costs this tab a row, and the tab height is capped so that the plots
        # below keep their space.
        cfg.setSpacing(2)
        cfg.addWidget(self._build_source_group(), 0, 0, 1, 3)

        self.gain_slider, self.gain_spin = self._param_row(
            cfg, 1, "Gain:", 0.0, 40.0, 15.0, scale=10)
        self.gain_slider.valueChanged.connect(
            lambda v: (self.gain_spin.setValue(v / 10.0), self._on_gain(v / 10.0)))
        self.gain_spin.valueChanged.connect(
            lambda v: (self.gain_slider.blockSignals(True),
                       self.gain_slider.setValue(int(v * 10)),
                       self.gain_slider.blockSignals(False),
                       self._on_gain(v)))

        self.exp_slider, self.exp_spin = self._param_row(
            cfg, 2, "Exp (ms):", 1.0, 100000.0, 100.0, scale=1)
        self.exp_slider.valueChanged.connect(
            lambda v: (self.exp_spin.setValue(float(v)), self._on_exp(float(v))))
        self.exp_spin.valueChanged.connect(
            lambda v: (self.exp_slider.blockSignals(True),
                       self.exp_slider.setValue(int(v)),
                       self.exp_slider.blockSignals(False),
                       self._on_exp(v)))

        # FPS + binning row
        fb_row = QHBoxLayout()
        fb_row.addWidget(QLabel("Target FPS:"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1, 200); self.fps_spin.setValue(30.0)
        self.fps_spin.setFixedWidth(65)
        self.fps_spin.valueChanged.connect(self._on_target_fps)
        fb_row.addWidget(self.fps_spin)
        fb_row.addSpacing(8)
        fb_row.addWidget(QLabel("Binning:"))
        self._bin_group = QButtonGroup(self)
        for val, lbl in [(1, "1×"), (2, "2×"), (4, "4×")]:
            rb = QRadioButton(lbl)
            rb.setChecked(val == DEFAULT_BINNING)
            rb.toggled.connect(lambda checked, v=val: checked and self._apply_binning(v))
            fb_row.addWidget(rb)
            self._bin_group.addButton(rb, val)
        fb_row.addStretch()
        cfg.addLayout(fb_row, 3, 0, 1, 3)

        # Rotation row
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation:"))
        btn_ccw = QPushButton("↺ 90°")
        btn_ccw.setFixedWidth(55)
        btn_ccw.setToolTip("Rotate 90° counter-clockwise (clears ROIs + reference)")
        btn_ccw.clicked.connect(self._on_rotate_ccw)
        rot_row.addWidget(btn_ccw)
        btn_cw = QPushButton("↻ 90°")
        btn_cw.setFixedWidth(55)
        btn_cw.setToolTip("Rotate 90° clockwise (clears ROIs + reference)")
        btn_cw.clicked.connect(self._on_rotate_cw)
        rot_row.addWidget(btn_cw)
        rot_row.addSpacing(8)
        rot_row.addWidget(QLabel("Fine (°):"))
        self.rot_fine_spin = QDoubleSpinBox()
        self.rot_fine_spin.setRange(-10.0, 10.0)
        self.rot_fine_spin.setValue(0.0)
        self.rot_fine_spin.setSingleStep(0.5)
        self.rot_fine_spin.setDecimals(1)
        self.rot_fine_spin.setFixedWidth(65)
        self.rot_fine_spin.setToolTip("Fine rotation ±10° (clears reference; ROIs kept)")
        self.rot_fine_spin.valueChanged.connect(self._on_rot_fine_changed)
        rot_row.addWidget(self.rot_fine_spin)
        rot_row.addStretch()
        cfg.addLayout(rot_row, 4, 0, 1, 3)

        # Reference row
        ref_row = QHBoxLayout()
        btn_set = QPushButton("Set Reference")
        btn_set.setStyleSheet("background:#8e44ad;color:white;")
        btn_set.setToolTip(
            "Store the current frame and subtract it from the display.\n"
            "Display only — ROI values are always measured on the raw frame.")
        btn_set.clicked.connect(self._set_reference)
        ref_row.addWidget(btn_set)
        btn_clr = QPushButton("Clear Reference")
        btn_clr.setStyleSheet("background:#7f8c8d;color:white;")
        btn_clr.clicked.connect(self._clear_reference)
        ref_row.addWidget(btn_clr)
        self.ref_label = QLabel("No reference set")
        self.ref_label.setStyleSheet("color:#bdc3c7;")
        ref_row.addWidget(self.ref_label)
        ref_row.addStretch()
        cfg.addLayout(ref_row, 5, 0, 1, 3)

        cfg.setRowStretch(6, 1)
        return page

    def _build_source_group(self):
        """Which camera the live view comes from.

        Kept at the top of the Camera tab rather than buried in a dialog: on a
        system with more than one camera on the chamber, picking the right one
        is the first thing an operator does, and after a cable is re-seated it
        is the first thing they do again.
        """
        box = QGroupBox("Source")
        box.setStyleSheet(GROUP_QSS)
        v = QVBoxLayout(box)
        v.setSpacing(3)
        # Tighter than the default margins: this group sits on top of an
        # already-full tab, and every pixel it takes comes off the plots.
        v.setContentsMargins(6, 4, 6, 4)

        row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cam_combo.setToolTip(
            "Cameras found on this machine.\n"
            "Help ▸ Camera backends lists what VRHEED can talk to and what is "
            "missing.")
        self.cam_combo.activated.connect(self._on_camera_selected)
        row.addWidget(self.cam_combo, 1)
        self.btn_cam_refresh = QPushButton("⟳")
        self.btn_cam_refresh.setFixedWidth(30)
        self.btn_cam_refresh.setToolTip("Scan for cameras again")
        self.btn_cam_refresh.clicked.connect(lambda: self._refresh_cameras())
        row.addWidget(self.btn_cam_refresh)
        self.btn_cam_connect = QPushButton("Connect")
        self.btn_cam_connect.setFixedWidth(95)
        self.btn_cam_connect.setStyleSheet("background:#16a085;color:white;")
        self.btn_cam_connect.clicked.connect(self._toggle_connection)
        row.addWidget(self.btn_cam_connect)
        v.addLayout(row)

        # The resolution row is only meaningful for the backends that offer a
        # choice (USB cameras, essentially), so it lives in its own widget and
        # is hidden the rest of the time rather than sitting there greyed out
        # -- the Camera tab has no vertical space to spare.
        self.res_row = QWidget()
        row2 = QHBoxLayout(self.res_row)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.setEnabled(False)
        self.res_combo.setToolTip(
            "Sensor sizes this camera accepts.\n"
            "USB cameras default to 640×480 whatever the sensor can do, and "
            "those missing pixels are missing streak-spacing resolution.")
        self.res_combo.activated.connect(self._on_resolution_selected)
        row2.addWidget(self.res_combo, 1)
        self.btn_driver = QPushButton("Driver…")
        self.btn_driver.setFixedWidth(70)
        self.btn_driver.setEnabled(False)
        self.btn_driver.setToolTip(
            "Open the driver's own settings dialog (Windows USB cameras).\n"
            "The reliable place to turn a webcam's auto-exposure off.")
        self.btn_driver.clicked.connect(self._open_driver_dialog)
        row2.addWidget(self.btn_driver)
        self.res_row.setVisible(False)
        v.addWidget(self.res_row)

        self.cam_status_label = QLabel("Searching for cameras…")
        self.cam_status_label.setWordWrap(True)
        # Two lines' worth, claimed up front: the Camera tab is dense enough
        # that an unconstrained label gets squeezed to one line by the grid,
        # and the second line is the one carrying the backend's caveat.
        self.cam_status_label.setMinimumHeight(
            2 * self.cam_status_label.fontMetrics().height())
        self.cam_status_label.setStyleSheet("color:#bdc3c7;")
        v.addWidget(self.cam_status_label)
        return box

    def _build_display_tab(self):
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(4)

        self.gamma_slider, self.gamma_spin = self._param_row(
            g, 0, "Gamma:", 0.1, 2.0, 0.5, scale=10)
        self.gamma_slider.valueChanged.connect(
            lambda v: self.gamma_spin.setValue(v / 10.0))
        self.gamma_spin.valueChanged.connect(
            lambda v: (self.gamma_slider.blockSignals(True),
                       self.gamma_slider.setValue(int(v * 10)),
                       self.gamma_slider.blockSignals(False)))

        g.addWidget(QLabel("Colormap:"), 1, 0, alignment=Qt.AlignRight)
        self.cmap_combo = QComboBox()
        for name, _ in COLORMAPS:
            self.cmap_combo.addItem(name)
        g.addWidget(self.cmap_combo, 1, 1, 1, 2)

        # Contrast
        lvl_row = QHBoxLayout()
        self.contrast_auto_cb = QCheckBox("Auto contrast")
        self.contrast_auto_cb.setChecked(True)
        self.contrast_auto_cb.setToolTip(
            "Auto rescales every frame to its own min/max, which makes the\n"
            "brightness flicker in recorded video and hides slow intensity\n"
            "drifts.  Uncheck and set fixed levels for quantitative viewing.")
        lvl_row.addWidget(self.contrast_auto_cb)
        lvl_row.addWidget(QLabel("Lo:"))
        self.level_lo = QSpinBox()
        self.level_lo.setRange(0, 65535); self.level_lo.setValue(0)
        self.level_lo.setFixedWidth(70)
        lvl_row.addWidget(self.level_lo)
        lvl_row.addWidget(QLabel("Hi:"))
        self.level_hi = QSpinBox()
        self.level_hi.setRange(1, 65535); self.level_hi.setValue(255)
        self.level_hi.setFixedWidth(70)
        lvl_row.addWidget(self.level_hi)
        btn_grab = QPushButton("Grab")
        btn_grab.setFixedWidth(48)
        btn_grab.setToolTip("Fill Lo/Hi from the current frame, then hold them")
        btn_grab.clicked.connect(self._grab_levels)
        lvl_row.addWidget(btn_grab)
        lvl_row.addStretch()
        g.addLayout(lvl_row, 2, 0, 1, 3)

        # Frame averaging
        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("Frame average:"))
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 32); self.avg_spin.setValue(1)
        self.avg_spin.setFixedWidth(60)
        self.avg_spin.setToolTip(
            "Rolling mean of the last N frames.  Improves the visible\n"
            "signal-to-noise of a weak pattern at the cost of time resolution.")
        self.avg_spin.valueChanged.connect(self._on_avg_changed)
        avg_row.addWidget(self.avg_spin)
        self.avg_measure_cb = QCheckBox("Also average measurements")
        self.avg_measure_cb.setToolTip(
            "Off (default): ROI values come from single raw frames, so the\n"
            "oscillation is not low-pass filtered.  On: averaged frames are\n"
            "measured too, which smooths the trace but attenuates fast\n"
            "oscillations — check that N/fps stays well under one period.")
        avg_row.addWidget(self.avg_measure_cb)
        avg_row.addStretch()
        g.addLayout(avg_row, 3, 0, 1, 3)

        zrow = QHBoxLayout()
        btn_zr = QPushButton("Reset zoom")
        btn_zr.clicked.connect(self._reset_zoom)
        zrow.addWidget(btn_zr)
        self.zoom_label = QLabel("Zoom: 1.0×")
        self.zoom_label.setStyleSheet("color:#bdc3c7;")
        zrow.addWidget(self.zoom_label)
        zrow.addStretch()
        g.addLayout(zrow, 4, 0, 1, 3)

        g.setRowStretch(5, 1)
        return page

    def _build_analysis_tab(self):
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(4)

        g.addWidget(QLabel("ML (Å):"), 0, 0, alignment=Qt.AlignRight)
        self.ml_spin = QDoubleSpinBox()
        self.ml_spin.setRange(0.1, 100.0); self.ml_spin.setValue(2.82)
        self.ml_spin.setSingleStep(0.01); self.ml_spin.setFixedWidth(75)
        self.ml_spin.setToolTip("Thickness deposited per RHEED oscillation")
        g.addWidget(self.ml_spin, 0, 1)

        g.addWidget(QLabel("f min (Hz):"), 1, 0, alignment=Qt.AlignRight)
        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(0.001, 10.0); self.fmin_spin.setValue(0.01)
        self.fmin_spin.setSingleStep(0.01); self.fmin_spin.setDecimals(3)
        self.fmin_spin.setFixedWidth(75)
        g.addWidget(self.fmin_spin, 1, 1)

        g.addWidget(QLabel("f max (Hz):"), 2, 0, alignment=Qt.AlignRight)
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(0.01, 20.0); self.fmax_spin.setValue(2.0)
        self.fmax_spin.setSingleStep(0.1); self.fmax_spin.setFixedWidth(75)
        g.addWidget(self.fmax_spin, 2, 1)

        g.addWidget(QLabel("History (min):"), 0, 2, alignment=Qt.AlignRight)
        self.hist_spin = QDoubleSpinBox()
        self.hist_spin.setRange(1, 240); self.hist_spin.setValue(30.0)
        self.hist_spin.setSingleStep(5); self.hist_spin.setDecimals(0)
        self.hist_spin.setFixedWidth(75)
        self.hist_spin.setToolTip(
            "Maximum duration of data kept in memory.\n"
            "Existing ROI histories are resized when changed.")
        g.addWidget(self.hist_spin, 0, 3)
        self.hist_spin.valueChanged.connect(self._on_history_changed)

        g.addWidget(QLabel("Active ROI tracks:"), 3, 0, alignment=Qt.AlignRight)
        self.metric_combo = QComboBox()
        for key, label in ROI_METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.setToolTip(
            "What the Intensity-vs-Time trace records for the selected ROI.\n\n"
            "Mean intensity   — classic RHEED oscillation signal\n"
            "Centroid X / Y   — spot drift; tracks lattice relaxation and\n"
            "                   sample motion\n"
            "Profile FWHM     — streak sharpening/broadening, i.e. surface\n"
            "                   order and domain size\n"
            "Streak spacing   — in-plane lattice parameter vs time")
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        g.addWidget(self.metric_combo, 3, 1, 1, 3)

        g.addWidget(QLabel("Track feature:"), 4, 0, alignment=Qt.AlignRight)
        self.track_combo = QComboBox()
        for key, label in TRACK_MODES:
            self.track_combo.addItem(label, key)
        self.track_combo.setToolTip(
            "Re-centre the active ROI on its feature every frame, so the box\n"
            "follows a streak or spot that drifts during growth instead of\n"
            "sliding off it.  Turn this off before measuring spot position,\n"
            "since a tracking box holds the centroid at its own centre.")
        self.track_combo.currentIndexChanged.connect(self._on_track_changed)
        g.addWidget(self.track_combo, 4, 1, 1, 3)

        self.osc_note = QLabel("")
        self.osc_note.setStyleSheet("color:#bdc3c7;font-size:9pt;")
        self.osc_note.setWordWrap(True)
        g.addWidget(self.osc_note, 5, 0, 1, 4)

        g.setColumnStretch(1, 1)
        g.setRowStretch(6, 1)
        return page

    def _build_lattice_tab(self):
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(4)

        # Labels are kept short and the spin columns carry the stretch; the
        # earlier long labels were silently truncated at this panel width.
        g.addWidget(QLabel("Energy (keV):"), 0, 0, alignment=Qt.AlignRight)
        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setRange(1.0, 100.0); self.energy_spin.setValue(20.0)
        self.energy_spin.setSingleStep(0.5); self.energy_spin.setMinimumWidth(80)
        self.energy_spin.valueChanged.connect(self._mark_sim_dirty)
        g.addWidget(self.energy_spin, 0, 1)

        g.addWidget(QLabel("Known a (Å):"), 0, 2, alignment=Qt.AlignRight)
        self.known_a_spin = QDoubleSpinBox()
        self.known_a_spin.setRange(0.5, 100.0); self.known_a_spin.setValue(3.905)
        self.known_a_spin.setDecimals(4); self.known_a_spin.setSingleStep(0.01)
        self.known_a_spin.setMinimumWidth(80)
        self.known_a_spin.setToolTip(
            "Substrate in-plane lattice constant, e.g. SrTiO₃ = 3.905 Å")
        g.addWidget(self.known_a_spin, 0, 3)

        g.addWidget(QLabel("Screen L (mm):"), 1, 0, alignment=Qt.AlignRight)
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(10.0, 2000.0); self.dist_spin.setValue(300.0)
        self.dist_spin.setSingleStep(5.0); self.dist_spin.setMinimumWidth(80)
        self.dist_spin.valueChanged.connect(self._mark_sim_dirty)
        g.addWidget(self.dist_spin, 1, 1)

        g.addWidget(QLabel("Pixel (mm):"), 1, 2, alignment=Qt.AlignRight)
        self.pixel_spin = QDoubleSpinBox()
        self.pixel_spin.setRange(0.0001, 5.0); self.pixel_spin.setValue(0.100)
        self.pixel_spin.setDecimals(4); self.pixel_spin.setSingleStep(0.01)
        self.pixel_spin.setMinimumWidth(80)
        self.pixel_spin.setToolTip(
            "Size of one camera pixel projected onto the phosphor screen,\n"
            "including the camera lens magnification and current binning.")
        g.addWidget(self.pixel_spin, 1, 3)
        # The pitch box means the pitch AFTER binning (see its tooltip), so
        # the binning in force has to be visible here -- otherwise the default
        # 4x silently makes every geometry-derived lattice constant 4x wrong
        # for anyone who typed in their sensor's raw pixel size.
        self.bin_note = QLabel()
        self.bin_note.setStyleSheet("color:#f39c12;")
        self.bin_note.setWordWrap(True)
        g.addWidget(self.bin_note, 2, 0, 1, 4)

        btn_cal = QPushButton("Calibrate from pattern")
        btn_cal.setStyleSheet("background:#2980b9;color:white;")
        btn_cal.setToolTip(
            "Use the streak separation measured right now, together with the\n"
            "known lattice constant, to fix the geometry constant K = a·Δx.\n"
            "After this you never need L or the pixel pitch.")
        btn_cal.clicked.connect(self._calibrate_lattice)
        g.addWidget(btn_cal, 2, 0, 1, 2)

        btn_ref = QPushButton("Set strain reference")
        btn_ref.setStyleSheet("background:#8e44ad;color:white;")
        btn_ref.setToolTip(
            "Take the lattice constant showing now as a₀, so later readings "
            "report strain against it.")
        btn_ref.clicked.connect(self._set_strain_reference)
        g.addWidget(btn_ref, 2, 2, 1, 2)

        self.lat_result = QLabel("Tick “Mark streaks” under the slice plot and "
                                 "place an ROI across two or more streaks.")
        self.lat_result.setStyleSheet("color:#f1c40f;font-size:10pt;")
        self.lat_result.setWordWrap(True)
        g.addWidget(self.lat_result, 3, 0, 1, 4)

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        g.setRowStretch(4, 1)
        return page

    def _build_sim_tab(self):
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(4)

        def spin(lo, hi, val, dec=3, step=0.01):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setValue(val); s.setDecimals(dec)
            s.setSingleStep(step); s.setMinimumWidth(80)
            s.valueChanged.connect(self._mark_sim_dirty)
            return s

        self.sim_on_cb = QCheckBox("Overlay simulated pattern")
        self.sim_on_cb.setToolTip(
            "Kinematic Ewald-sphere construction for a 2-D surface mesh.\n"
            "Use it to identify an azimuth, confirm a reconstruction, or\n"
            "check that the streak spacing you measure is the order you think.")
        g.addWidget(self.sim_on_cb, 0, 0, 1, 4)

        g.addWidget(QLabel("a (Å):"), 1, 0, alignment=Qt.AlignRight)
        self.sim_a = spin(0.5, 100.0, 3.905); g.addWidget(self.sim_a, 1, 1)
        g.addWidget(QLabel("b (Å):"), 1, 2, alignment=Qt.AlignRight)
        self.sim_b = spin(0.5, 100.0, 3.905); g.addWidget(self.sim_b, 1, 3)

        g.addWidget(QLabel("γ (°):"), 2, 0, alignment=Qt.AlignRight)
        self.sim_gamma = spin(30.0, 150.0, 90.0, dec=1, step=1.0)
        g.addWidget(self.sim_gamma, 2, 1)
        g.addWidget(QLabel("Azimuth (°):"), 2, 2, alignment=Qt.AlignRight)
        self.sim_azim = spin(-180.0, 180.0, 0.0, dec=1, step=1.0)
        g.addWidget(self.sim_azim, 2, 3)

        g.addWidget(QLabel("θ incidence (°):"), 3, 0, alignment=Qt.AlignRight)
        self.sim_theta = spin(0.1, 15.0, 2.0, dec=2, step=0.1)
        g.addWidget(self.sim_theta, 3, 1)
        g.addWidget(QLabel("Max order:"), 3, 2, alignment=Qt.AlignRight)
        self.sim_order = QSpinBox()
        self.sim_order.setRange(1, 12); self.sim_order.setValue(4)
        self.sim_order.setMinimumWidth(80)
        self.sim_order.valueChanged.connect(self._mark_sim_dirty)
        g.addWidget(self.sim_order, 3, 3)

        g.addWidget(QLabel("Scale (px/mm):"), 4, 0, alignment=Qt.AlignRight)
        self.sim_ppm = QDoubleSpinBox()
        self.sim_ppm.setRange(0.01, 1000.0); self.sim_ppm.setValue(10.0)
        self.sim_ppm.setDecimals(3); self.sim_ppm.setMinimumWidth(80)
        g.addWidget(self.sim_ppm, 4, 1)
        btn_fit = QPushButton("Scale from streaks")
        btn_fit.setToolTip(
            "Match the simulated first-order spacing to the streak spacing\n"
            "measured on the live pattern, which sets px/mm in one click.")
        btn_fit.clicked.connect(self._fit_sim_scale)
        g.addWidget(btn_fit, 4, 2, 1, 2)

        self.btn_pick = QPushButton("Pick shadow-edge origin")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip(
            "Then click on the image where the direct beam meets the shadow\n"
            "edge.  All simulated positions are drawn relative to that point.")
        self.btn_pick.toggled.connect(self._on_pick_toggled)
        g.addWidget(self.btn_pick, 5, 0, 1, 2)
        self.sim_origin_label = QLabel("origin: 0.50, 0.75")
        self.sim_origin_label.setStyleSheet("color:#bdc3c7;")
        g.addWidget(self.sim_origin_label, 5, 2, 1, 2)
        self._sim_origin = (0.5, 0.75)

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        g.setRowStretch(6, 1)
        return page

    def _build_metrics_group(self):
        met_group = QGroupBox("Growth Metrics")
        met_group.setStyleSheet(
            "QGroupBox{color:#1abc9c;font-weight:bold;border:1px solid #7f8c8d;"
            "border-radius:4px;margin-top:10px;padding-top:6px;}"
            "QGroupBox::title{color:#1abc9c;subcontrol-origin:margin;"
            "left:8px;padding:0 4px;}")
        met_grid = QGridLayout(met_group)
        met_grid.setSpacing(3)
        self.metric_labels = {}
        rows = [('freq', 'Hz:'), ('mls', 'ML/s:'), ('angs', 'Å/s:'),
                ('nmmin', 'nm/min:'), ('umhr', 'µm/hr:'), ('period', 'Period (s):')]
        for i, (key, lbl) in enumerate(rows):
            met_grid.addWidget(QLabel(lbl), i // 2, (i % 2) * 2)
            v = QLabel("---")
            v.setStyleSheet("color:#f1c40f;font-weight:bold;font-size:11pt;")
            met_grid.addWidget(v, i // 2, (i % 2) * 2 + 1)
            self.metric_labels[key] = v

        self.osc_label = QLabel("")
        self.osc_label.setStyleSheet("color:#95a5a6;font-size:9pt;")
        self.osc_label.setWordWrap(True)
        met_grid.addWidget(self.osc_label, 3, 0, 1, 4)

        # Third, independent growth-rate method — the only one that carries an
        # uncertainty, so it is what you quote.
        self.fit_label = QLabel("")
        self.fit_label.setStyleSheet("color:#95a5a6;font-size:9pt;")
        self.fit_label.setWordWrap(True)
        self.fit_label.setToolTip(
            "Damped-sine least-squares fit over the selected region.\n"
            "Reports the rate with its fit uncertainty, the 1/e decay time of\n"
            "the oscillation amplitude, and the thickness deposited over the\n"
            "selection.")
        met_grid.addWidget(self.fit_label, 4, 0, 1, 4)

        # Label showing which ROI drives the metrics panel and FFT peak dot
        self.active_roi_label = QLabel("Metrics: no ROIs yet")
        self.active_roi_label.setStyleSheet("color:#bdc3c7;font-size:9pt;")
        self.active_roi_label.setWordWrap(True)
        self.active_roi_label.setToolTip(
            "The numbers above come from this ROI.\n"
            "Click inside a different ROI box on the camera view to switch.\n"
            "The FFT curves are drawn for ALL ROIs in their matching colors.\n"
            "The filled dot on each FFT curve marks its detected peak frequency.")
        met_grid.addWidget(self.active_roi_label, 5, 0, 1, 4)
        return met_group

    @staticmethod
    def _param_row(grid, row, label, lo, hi, default, scale=10):
        """Add label + slider + spinbox; return (slider, spinbox)."""
        grid.addWidget(QLabel(label), row, 0, alignment=Qt.AlignRight)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(int(lo * scale), int(hi * scale))
        sl.setValue(int(default * scale))
        grid.addWidget(sl, row, 1)
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(default)
        sp.setSingleStep(1.0 / scale)
        sp.setDecimals(max(0, len(str(int(scale))) - 1))
        sp.setFixedWidth(75)
        grid.addWidget(sp, row, 2)
        return sl, sp

    # -----------------------------------------------------------------------
    # Camera controls
    # -----------------------------------------------------------------------

    def _on_gain(self, v):
        if self.cam:
            self.cam.set_gain(v)

    def _on_exp(self, v):
        if self.cam:
            self.cam.set_exposure_ms(v)

    def _on_target_fps(self, v):
        if self.cam:
            self.cam.set_frame_rate(v)
        self._on_history_changed()

    def _set_reference(self):
        if self.last_raw_frame is not None:
            self.reference_frame = self.last_raw_frame.copy()
            self.ref_label.setText("Reference active")
            self.ref_label.setStyleSheet("color:#2ecc71;")

    def _clear_reference(self):
        self.reference_frame = None
        self.ref_label.setText("No reference set")
        self.ref_label.setStyleSheet("color:#bdc3c7;")

    def _on_avg_changed(self, n):
        self._avg_frames = deque(maxlen=max(1, int(n)))
        self._reset_averaging()

    def _reset_averaging(self):
        """Empty the rolling-average window and its running sum together.

        They must never disagree: a sum left over from frames of a different
        size or bit depth would be averaged into the new ones.
        """
        self._avg_frames.clear()
        self._avg_sum = None
        self._avg_count = 0

    def _averaged_frame(self, frame):
        """Rolling mean of the last N frames as float32, or the frame itself for N = 1.

        Keeps a running sum: add the incoming frame, subtract the one the deque
        is about to drop.  Identical to np.mean over the window to within
        float32 rounding; the sum is rebuilt from scratch every 1000 frames so
        that rounding cannot accumulate into a visible offset over an hour.
        """
        n_max = self._avg_frames.maxlen
        if not n_max or n_max <= 1:
            return frame
        f32 = frame.astype(np.float32)
        if (self._avg_sum is None or self._avg_sum.shape != f32.shape or
                (self._avg_frames and self._avg_frames[0].shape != f32.shape)):
            self._reset_averaging()
            self._avg_sum = np.zeros_like(f32)
        if len(self._avg_frames) == n_max:
            self._avg_sum -= self._avg_frames[0]   # append() drops this one
        self._avg_frames.append(f32)
        self._avg_sum += f32
        self._avg_count += 1
        if self._avg_count % 1000 == 0:
            self._avg_sum = np.zeros_like(f32)
            for f in self._avg_frames:
                self._avg_sum += f
        return self._avg_sum / np.float32(len(self._avg_frames))

    def _grab_levels(self):
        if self.last_raw_frame is None:
            return
        self.level_lo.setValue(int(self.last_raw_frame.min()))
        self.level_hi.setValue(max(int(self.last_raw_frame.max()), 1))
        self.contrast_auto_cb.setChecked(False)

    def _apply_binning(self, n):
        if self.cam is None or self.source_mode != 'camera':
            return
        self._stop_capture()
        self.ui_timer.stop()
        QTimer.singleShot(150, lambda: self._restart_binning(n))

    def _restart_binning(self, n):
        # Frame dimensions change, so a recording opened at the old size would
        # silently drop every subsequent frame.
        if self.is_recording:
            self._toggle_recording()
        self._reset_averaging()
        self.reference_frame = None
        self._clear_reference()
        # A calibration constant is K = a * dx_px, and dx_px scales with the
        # binning, so K measured at one binning is wrong at another.  Rescale
        # it rather than discarding it: the operator calibrated against a
        # known lattice constant and should not have to do it again for a
        # change that does not move the pattern.
        old_bin = self.cam.binning if self.cam is not None else 1
        if self._lattice_K is not None and old_bin and n:
            self._lattice_K *= float(old_bin) / float(n)
        how = ""
        try:
            self.cam.set_binning(n)
            # A camera with no binning node still bins -- in software, in the
            # backend -- and so does one that caps out at 2x when asked for 4x.
            # Say which, because only hardware binning also buys the read-noise
            # reduction and the higher frame rate.
            if self.cam.binning_is_software:
                how = " (software)"
        except Exception as e:
            QMessageBox.warning(self, "Binning", f"Could not set binning: {e}")
        # Anything still queued is at the old resolution and would be drawn
        # once, with the ROIs misplaced, before the first new frame arrives.
        self._drain_frame_queue()
        self._start_acquisition()
        self.ui_timer.start(16)
        self._update_binning_note()
        note = (" Calibration K rescaled." if self._lattice_K is not None
                else f" Pixel pitch on the Lattice tab is now {n}× larger.")
        self.statusBar().showMessage(f"Binning {n}×{n}{how}.{note}", 6000)

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("RESUME" if self.paused else "PAUSE")
        self.btn_pause.setStyleSheet(
            "background:#27ae60;color:white;font-weight:bold;" if self.paused
            else "background:#f39c12;color:white;font-weight:bold;")

    def _reset_zoom(self):
        self._zoom_level = 1.0
        self._zoom_cx, self._zoom_cy = 0.5, 0.5
        self.zoom_label.setText("Zoom: 1.0×")

    # -----------------------------------------------------------------------
    # Camera init & acquisition thread
    # -----------------------------------------------------------------------

    def _init_camera(self):
        """Find the cameras on this machine and connect to one."""
        self._refresh_cameras(auto_connect=True)

    # -- source chooser ----------------------------------------------------

    def _refresh_cameras(self, auto_connect=False):
        """Re-scan every backend and repopulate the source list.

        Enumeration opens and closes each USB device in turn, so it takes a
        second or two; it happens at start-up, and after that only when the
        operator asks.  A camera already connected is left alone -- re-scanning
        must never interrupt a growth.
        """
        self.cam_status_label.setText("Searching for cameras…")
        self.statusBar().showMessage("Searching for cameras…")
        QApplication.processEvents()
        try:
            self._cameras = vcam.enumerate_cameras()
        except Exception:
            logger.exception("Camera enumeration failed")
            self._cameras = []
        logger.info("Found %d camera(s): %s", len(self._cameras),
                    ", ".join(c.key for c in self._cameras))

        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for info in self._cameras:
            self.cam_combo.addItem(info.label, info.key)
        if not self._cameras:
            self.cam_combo.addItem("No camera found", "")
        self.cam_combo.blockSignals(False)

        # Keep the combo pointing at the connected camera if it survived.
        current = self.cam_info.key if self.cam_info else self._last_cam_key
        idx = self.cam_combo.findData(current) if current else -1
        if idx >= 0:
            self.cam_combo.setCurrentIndex(idx)

        if not auto_connect or self.cam is not None:
            self._update_source_status()
            self.statusBar().clearMessage()
            return

        target = vcam.find_camera(self._last_cam_key, self._cameras)
        if target is None:
            # First real camera.  Never the simulated source or a screen
            # region: both would happily produce frames and look exactly like
            # a working camera, which is the last thing you want to discover
            # halfway through a growth.
            target = next((c for c in self._cameras
                           if c.backend_id not in ('synthetic', 'screen')), None)
        if target is not None:
            self.cam_combo.setCurrentIndex(
                max(0, self.cam_combo.findData(target.key)))
            self._connect_camera(target)
        else:
            self._set_camera_controls_enabled(False)
            self._update_source_status()
            self.statusBar().showMessage(
                "No camera found — pick a source above, or File ▸ Open Video "
                "to analyse a recording.", 0)

    def _on_camera_selected(self, _index):
        """Combo activated by the user: connect straight away.

        Choosing a camera and then having to press Connect is a step nobody
        remembers; the button stays for the disconnect direction.
        """
        info = self._selected_camera()
        if info is not None and (self.cam_info is None or info != self.cam_info):
            self._connect_camera(info)

    def _selected_camera(self):
        key = self.cam_combo.currentData()
        return vcam.find_camera(key, self._cameras) if key else None

    def _toggle_connection(self):
        if self.cam is not None:
            self._disconnect_camera()
            self.statusBar().showMessage("Camera disconnected.", 4000)
            self._update_source_status()
        else:
            info = self._selected_camera()
            if info is None:
                self._refresh_cameras(auto_connect=True)
            else:
                self._connect_camera(info)

    def _connect_camera(self, info):
        """Open ``info`` and start streaming from it.  Returns True on success."""
        self._disconnect_camera()
        try:
            cam = vcam.open_camera(info)
        except CameraError as e:
            logger.warning("Could not open %s: %s", info.key, e)
            QMessageBox.critical(self, "Camera",
                                 f"Could not open {info.label}:\n\n{e}")
            self._set_camera_controls_enabled(False)
            self._update_source_status(f"Could not open {info.label}")
            return False

        self.cam = cam
        self.cam_info = info
        self._last_cam_key = info.key
        # Keep the chooser honest however we got here -- auto-connect at
        # start-up and the screen-region dialog both call this directly.
        idx = self.cam_combo.findData(info.key)
        if idx >= 0 and idx != self.cam_combo.currentIndex():
            self.cam_combo.blockSignals(True)
            self.cam_combo.setCurrentIndex(idx)
            self.cam_combo.blockSignals(False)
        self._reset_averaging()
        self._clear_reference()
        self._apply_control_ranges(cam)
        self._set_camera_controls_enabled(True)
        self._populate_resolutions()

        # Push the panel's current values down to the new camera, so what the
        # controls say is what the camera is doing.
        cam.set_gain(self.gain_spin.value())
        cam.set_exposure_ms(self.exp_spin.value())
        cam.set_frame_rate(self.fps_spin.value())
        cam.set_binning(self._bin_group.checkedId() or 1)

        self.btn_cam_connect.setText("Disconnect")
        self._update_binning_note()
        self._update_source_status()
        logger.info("Connected to %s", info.key)
        if self.source_mode == 'camera':
            self._drain_frame_queue()
            self._start_acquisition()
            self.statusBar().showMessage(f"Connected to {info.label}.", 5000)
        else:
            self.statusBar().showMessage(
                f"Connected to {info.label} — File ▸ Return to Camera to view it.",
                6000)
        # The long form of the caveat, once, where there is room to read it.
        if cam.note_detail:
            QTimer.singleShot(
                1200, lambda d=cam.note_detail: self.statusBar().showMessage(d, 12000))
        return True

    def _disconnect_camera(self):
        if self.cam is None:
            return
        self._stop_capture()
        try:
            self.cam.close()
        except Exception:
            logger.exception("Error closing camera")
        self.cam = None
        self.cam_info = None
        self._set_camera_controls_enabled(False)
        self.btn_cam_connect.setText("Connect")

    def _apply_control_ranges(self, cam):
        """Clamp the gain / exposure / FPS controls to what this camera allows.

        A slider that runs to 40 dB on a camera whose maximum is 24 invites an
        operator to set a value the camera silently ignores, and then the panel
        and the sensor disagree for the rest of the session.
        """
        for lo_hi, slider, spin, scale in (
                (cam.gain_range, self.gain_slider, self.gain_spin, 10),
                (cam.exposure_range_ms, self.exp_slider, self.exp_spin, 1),
                (cam.frame_rate_range, None, self.fps_spin, 1)):
            if not lo_hi:
                continue
            try:
                lo, hi = float(lo_hi[0]), float(lo_hi[1])
                if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                    continue
                # Round INWARD to what the spin box can actually display.  A
                # Blackfly's gain maximum is 47.99 dB and a one-decimal spin
                # box rounds that to 48.0, which the camera then rejects; the
                # exposure spin box shows whole milliseconds, so a 6 us
                # minimum would display as 0 and offer a value no camera
                # accepts.  Flooring the top and raising the bottom means
                # every value the operator can dial in is one the camera takes.
                step = 10.0 ** -spin.decimals()
                lo = np.ceil(lo / step) * step
                hi = np.floor(hi / step) * step
                if hi <= lo:
                    continue
                for w in (slider, spin):
                    if w is None:
                        continue
                    w.blockSignals(True)
                    if w is slider:
                        w.setRange(int(round(lo * scale)), int(round(hi * scale)))
                    else:
                        w.setRange(lo, hi)
                    w.blockSignals(False)
            except Exception as e:
                logger.debug("Could not apply control range %r: %s", lo_hi, e)

    def _populate_resolutions(self):
        """Fill the resolution combo for the backends that offer a choice."""
        self.res_combo.blockSignals(True)
        self.res_combo.clear()
        sizes = []
        if self.cam is not None and self.cam.supports_resolution:
            try:
                sizes = self.cam.list_resolutions()
            except Exception:
                logger.exception("Could not list resolutions")
        for w, h in sizes:
            self.res_combo.addItem(f"{w} × {h}", (w, h))
        if not sizes:
            self.res_combo.addItem("—", None)
        self.res_combo.blockSignals(False)
        self.res_combo.setEnabled(bool(sizes))
        has_dialog = self.cam is not None and self.cam.supports_native_dialog
        self.btn_driver.setEnabled(has_dialog)
        self.res_row.setVisible(bool(sizes) or has_dialog)

    def _on_resolution_selected(self, _index):
        size = self.res_combo.currentData()
        if self.cam is None or not size:
            return
        # Frame dimensions change, so the same care as a binning change:
        # stop, reconfigure, drop everything queued at the old size.
        self._stop_capture()
        if self.is_recording:
            self._toggle_recording()
        self._reset_averaging()
        self._clear_reference()
        try:
            self.cam.set_resolution(*size)
        except Exception as e:
            QMessageBox.warning(self, "Resolution", f"Could not set size: {e}")
        self._drain_frame_queue()
        self._start_acquisition()
        self.statusBar().showMessage(f"Resolution set to {size[0]} × {size[1]}.", 5000)

    def _open_driver_dialog(self):
        if self.cam is None:
            QMessageBox.information(self, "Driver settings",
                                    "Connect a camera first.")
            return
        if not self.cam.supports_native_dialog:
            QMessageBox.information(
                self, "Driver settings",
                f"{self.cam.backend_name} has no driver dialog.\n\n"
                "Only USB cameras on Windows expose one; set gain and exposure "
                "with the controls on this tab instead.")
            return
        try:
            self.cam.open_native_dialog()
        except Exception as e:
            QMessageBox.information(self, "Driver settings", str(e))

    def _update_source_status(self, message=None):
        """The line under the source chooser: what is connected, and caveats."""
        if message:
            self.cam_status_label.setText(message)
            self.cam_status_label.setStyleSheet("color:#e67e22;")
            return
        if self.cam is None:
            n = len(self._cameras)
            self.cam_status_label.setText(
                f"Not connected — {n} source{'' if n == 1 else 's'} found."
                if n else "No camera found. Help ▸ Camera backends shows why.")
            self.cam_status_label.setStyleSheet("color:#bdc3c7;")
            return
        text = f"Connected · {self.cam.backend_name}"
        if self.cam.note:
            text += f"\n{self.cam.note}"
        self.cam_status_label.setText(text)
        # The label is two lines tall; the tooltip is where the long form of
        # the caveat fits, whatever the operator has done to the splitter.
        self.cam_status_label.setToolTip(
            f"{self.cam_info.label}\n{self.cam.backend_name}"
            + (f"\n\n{self.cam.note_detail or self.cam.note}"
               if (self.cam.note_detail or self.cam.note) else ""))
        self.cam_status_label.setStyleSheet("color:#2ecc71;")

    def _add_network_camera(self):
        """Add an RTSP / HTTP stream URL by hand — nothing can discover these."""
        url, ok = QInputDialog.getText(
            self, "Add network camera",
            "Stream URL:\n\n"
            "  rtsp://user:password@192.168.1.64:554/Streaming/Channels/101\n"
            "  http://192.168.1.90/mjpg/video.mjpg\n")
        if not ok or not url.strip():
            return
        vcam.register_network_camera(url.strip())
        self._refresh_cameras()
        idx = self.cam_combo.findData(f"network:{url.strip()}")
        if idx >= 0:
            self.cam_combo.setCurrentIndex(idx)
            self._on_camera_selected(idx)

    def _remove_network_camera(self):
        urls = vcam.network_cameras()
        if not urls:
            QMessageBox.information(self, "Network cameras",
                                    "No network camera has been added.")
            return
        url, ok = QInputDialog.getItem(self, "Remove network camera",
                                       "Stream to forget:", urls, 0, False)
        if not ok or not url:
            return
        if self.cam_info is not None and self.cam_info.key == f"network:{url}":
            self._disconnect_camera()
        vcam.forget_network_camera(url)
        self._refresh_cameras()

    def _set_screen_region(self):
        """Choose which rectangle of the desktop the screen source captures.

        The point of screen capture is a vendor program's live-image pane, so
        the useful region is a window, not a monitor; capturing the whole 4K
        display instead would spend most of the frame budget on the toolbars.
        """
        info = self._selected_camera()
        if info is None or info.backend_id != 'screen':
            QMessageBox.information(
                self, "Screen capture region",
                "Select a Screen source in the Camera tab first.")
            return
        monitor, region = vcam.parse_screen_id(info.device_id)
        dlg = ScreenRegionDialog(self, monitor, region)
        if dlg.exec_() != QDialog.Accepted:
            return
        rect = dlg.region()
        key = vcam.screen_source_key(monitor, rect)
        new_info = vcam.CameraInfo(
            vcam.ScreenCaptureBackend, key,
            f"Screen {monitor}  ({rect[2]}×{rect[3]} at {rect[0]},{rect[1]})",
            "desktop")
        # Replace the plain-monitor entry so the region survives a refresh.
        self._cameras = [c for c in self._cameras
                         if not (c.backend_id == 'screen'
                                 and vcam.parse_screen_id(c.device_id)[0] == monitor)]
        self._cameras.append(new_info)
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for c in self._cameras:
            self.cam_combo.addItem(c.label, c.key)
        self.cam_combo.setCurrentIndex(max(0, self.cam_combo.findData(key)))
        self.cam_combo.blockSignals(False)
        self._connect_camera(new_info)

    def _show_backends(self):
        """What VRHEED can talk to on this machine, and how to add the rest."""
        rows = []
        for backend in vcam.BACKENDS:
            try:
                ok = backend.available()
                reason = "" if ok else backend.unavailable_reason()
            except Exception:
                ok, reason = False, "could not be checked"
            mark = "✔" if ok else "✖"
            colour = "#2ecc71" if ok else "#e74c3c"
            extra = f" <span style='color:#bdc3c7;'>— {reason}</span>" if reason else ""
            # How much the backend has actually been used matters more than
            # whether its driver happens to be installed, and it is not
            # something the operator can find out any other way.
            grade = getattr(backend, 'verification', 'unverified')
            note = vcam.VERIFICATION_LABEL.get(grade, grade)
            note_colour = "#e67e22" if grade == "unverified" else "#95a5a6"
            rows.append(f"<tr><td style='color:{colour};'>{mark}</td>"
                        f"<td><b>{backend.backend_name}</b>{extra}<br>"
                        f"<span style='color:{note_colour};font-size:11px;'>"
                        f"{note}</span></td></tr>")
        QMessageBox.information(self, "Camera backends", (
            "<b>Camera support on this machine</b><br><br>"
            "<table cellspacing='4'>" + "".join(rows) + "</table><br>"
            "Every driver is optional and loaded only when used. Install the "
            "one your camera needs, then press ⟳ in the Camera tab.<br><br>"
            "<b>Only the FLIR backend has been used with a real camera.</b> "
            "The ones marked UNVERIFIED are written against each vendor's "
            "published API and have never met the hardware — treat them as a "
            "starting point, and please report what you find.<br><br>"
            "<small>The GenICam / GenTL backend drives <i>any</i> GigE Vision "
            "or USB3 Vision camera once one vendor SDK is installed, including "
            "vendors with no dedicated backend here.</small>"))

    def _set_camera_controls_enabled(self, on):
        """Enable exactly the controls the connected camera can honour."""
        cam = self.cam
        def cap(flag):
            return bool(on and cam is not None and getattr(cam, flag))
        for w in (self.gain_slider, self.gain_spin):
            w.setEnabled(cap('supports_gain'))
        for w in (self.exp_slider, self.exp_spin):
            w.setEnabled(cap('supports_exposure'))
        self.fps_spin.setEnabled(cap('supports_frame_rate'))
        max_bin = cam.max_binning() if (on and cam is not None) else 1
        for btn in self._bin_group.buttons():
            btn.setEnabled(bool(on) and self._bin_group.id(btn) <= max_bin)
        if not on:
            self.res_combo.setEnabled(False)
            self.btn_driver.setEnabled(False)
            self.res_row.setVisible(False)

    # -----------------------------------------------------------------------
    # Acquisition thread
    # -----------------------------------------------------------------------

    def _start_acquisition(self):
        if self.cam is None:
            return
        # Never leave a previous grab thread running: two threads on one camera
        # handle fight over the same buffer pool, and the symptom is dropped
        # frames halfway through a growth rather than an error anyone sees.
        if self._cap_thread is not None and self._cap_thread.is_alive():
            self._stop_capture()
        self._stop_event.clear()
        try:
            self.cam.start()
        except Exception as e:
            QMessageBox.warning(self, "Acquisition", f"Could not start: {e}")
            return
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

    def _stop_capture(self):
        """Stop the grab thread and wait for it, then end acquisition.

        Without the join, restarting acquisition (binning change, camera swap,
        shutdown) could leave the old thread inside the backend's blocking read
        while a new one starts, so two threads fight over the same handle.
        """
        self._stop_event.set()
        t, self._cap_thread = self._cap_thread, None
        if t is not None and t.is_alive():
            t.join(timeout=1.5)
        if self.cam is not None:
            try: self.cam.stop()
            except Exception: pass

    def _capture_loop(self):
        """Background thread: grab frames → queue.

        Everything vendor-specific -- buffer release, incomplete frames, mono
        conversion, software binning -- happens inside the backend, so this
        loop is the same whatever the camera is.
        """
        cam = self.cam
        while not self._stop_event.is_set():
            try:
                frame = cam.get_frame(500)
                if frame is None:
                    # A backend whose read returns immediately (a USB camera
                    # that has just been unplugged) would otherwise spin this
                    # loop at 100% CPU.
                    time.sleep(0.005)
                    continue
                t = time.monotonic()
                # Drop the oldest, not the newest: a stalled UI must see
                # the most recent pattern, not one from a second ago.
                if self.frame_queue.full():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: pass
                self.frame_queue.put((frame, t))
            except Exception:
                # A persistent camera error would otherwise spin this loop at
                # 100% CPU; back off enough to stay responsive but stay cheap.
                time.sleep(0.02)

    def _drain_frame_queue(self):
        """Discard every queued frame.

        Called whenever the source changes: a live frame still queued when a
        file opens would be measured and drawn as if it were the file's first
        frame, and the reverse when returning to the camera.
        """
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self._last_frame_time = None
        self.fps_tracker.clear()

    # -----------------------------------------------------------------------
    # File sources: recorded video and stills
    # -----------------------------------------------------------------------

    # -- file-dialog helpers --------------------------------------------------

    def _dialog_path(self, default_name=""):
        """Start every file dialog in the folder the user last used."""
        d = self._last_dir if self._last_dir and os.path.isdir(self._last_dir) else ""
        return os.path.join(d, default_name) if d else default_name

    def _remember_dir(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(d):
            self._last_dir = d

    @staticmethod
    def _path_arg(path):
        """Normalise the optional path argument of the file actions.

        These methods double as Qt slots, and QAction.triggered / QPushButton.
        clicked pass their ``checked`` bool into the first free parameter, so
        ``path`` arrives as False from the GUI and as a string from code.
        """
        return path if isinstance(path, str) and path else None

    def _open_video(self, path=None):
        path = self._path_arg(path)
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open RHEED video", self._dialog_path(),
                "Video (*.mp4 *.avi *.mov *.mkv);;All files (*)")
            if not path:
                return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Open video", f"Could not open:\n{path}")
            return
        self._remember_dir(path)

        self._stop_capture()
        self.play_timer.stop()
        if self.video_cap is not None:
            self.video_cap.release()
        self.video_cap = cap
        self.source_mode = 'file'
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = float(fps) if fps and fps > 0.1 else 30.0
        self.video_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._video_index = 0
        self._play_budget = 0.0

        self._clear_rois()
        self._clear_reference()
        self._reset_averaging()
        self._drain_frame_queue()
        self.actual_fps = self.video_fps

        self.seek_slider.setRange(0, max(self.video_nframes - 1, 0))
        self.seek_slider.setValue(0)
        self.play_bar.setVisible(True)
        self._set_camera_controls_enabled(False)
        self._apply_playback_speed()
        self.play_timer.start()
        self.btn_play.setText("❚❚")
        self.setWindowTitle(f"VRHEED - {os.path.basename(path)}")
        logger.info("Opened video %s: %d frames @ %.3f fps",
                    path, self.video_nframes, self.video_fps)
        self.statusBar().showMessage(
            f"{os.path.basename(path)}: {self.video_nframes} frames @ "
            f"{self.video_fps:.2f} fps — timestamps come from the file, so "
            f"growth rates are independent of playback speed.", 8000)

    def _open_image(self, path=None):
        path = self._path_arg(path)
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open RHEED image", self._dialog_path(),
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)")
            if not path:
                return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Open image", f"Could not read:\n{path}")
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._remember_dir(path)

        self._stop_capture()
        self.play_timer.stop()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.source_mode = 'file'
        self.play_bar.setVisible(False)
        self._set_camera_controls_enabled(False)
        self._clear_rois()
        self._clear_reference()
        self._reset_averaging()
        # Drain first so the still is the ONLY frame waiting: a leftover
        # camera frame would be shown instead, and a full queue used to make
        # put_nowait drop the image silently.
        self._drain_frame_queue()
        self.frame_queue.put((img, 0.0))
        self.setWindowTitle(f"VRHEED - {os.path.basename(path)}")
        self.statusBar().showMessage(
            "Still image loaded — draw an ROI to measure profiles, FWHM and "
            "lattice spacing.", 8000)

    def _apply_playback_speed(self):
        # The tick period is fixed; speed sets how many frames each tick
        # decodes (see _feed_video_frame).  Restart the fractional budget so
        # a speed change does not release a burst of owed frames.
        self.play_timer.setInterval(PLAY_TICK_MS)
        self._play_budget = 0.0

    def _toggle_playback(self):
        if self.video_cap is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶")
        else:
            self.play_timer.start()
            self.btn_play.setText("❚❚")

    def _playback_stopped(self):
        """True when no new frames are being fed (paused or ▶ not pressed)."""
        return self.paused or not self.play_timer.isActive()

    def _seek_video(self, pos):
        """Jump to frame ``pos`` and make the histories consistent with it.

        Every sample stamped at or after the new position is dropped, so
        seeking backwards truncates the trace instead of drawing a zigzag
        back over itself, and the FFT window sees one monotonic timeline.
        When playback is stopped the frame at ``pos`` is decoded and shown so
        the operator can see where they landed.
        """
        if self.video_cap is None:
            return
        pos = int(max(0, min(int(pos), max(self.video_nframes - 1, 0))))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self._video_index = pos
        self._play_budget = 0.0
        t_new = pos / self.video_fps
        self._drain_frame_queue()
        self._truncate_history_from(t_new)

        if self._playback_stopped():
            ok, frame = self.video_cap.read()
            if ok:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.paused:
                    # Not recorded while paused, so rewind: the same frame
                    # must be measured when playback resumes.
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                else:
                    self._video_index = pos + 1
                self.frame_queue.put((frame, t_new))

        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(pos)
        self.seek_slider.blockSignals(False)
        self.play_pos_label.setText(f"{pos} / {self.video_nframes}")
        self._fft_dirty = True
        if self.histories:
            self._update_plots(force=True)

    def _truncate_history_from(self, t_abs):
        """Drop every sample whose source timestamp is >= ``t_abs`` seconds."""
        if self._t_origin is None:
            return
        # Half a frame of slack so the frame AT the new position, which will
        # be measured again, is not kept twice.
        cut = t_abs - self._t_origin - 0.5 / max(self.video_fps, 1.0)
        for i in list(self.times):
            tt, hh = self.times[i], self.histories[i]
            while tt and tt[-1] > cut and hh:
                tt.pop()
                hh.pop()
        self._roi_cache = {}

    def _feed_video_frame(self):
        """Playback tick: decode as many frames as the speed owes this tick.

        Frames are pushed to the queue and measured by _ui_loop one by one,
        so nothing is dropped as long as the queue has room; when it is full
        the remainder waits for the next tick and the effective speed simply
        tops out at what the machine can process.
        """
        if self.video_cap is None or self.paused:
            return
        per_tick = self.video_fps * self.speed_spin.value() * PLAY_TICK_MS / 1000.0
        self._play_budget += per_tick
        n = int(self._play_budget)
        if n <= 0:
            return
        self._play_budget -= n
        for _ in range(n):
            if self.frame_queue.full():
                break
            ok, frame = self.video_cap.read()
            if not ok:
                self.play_timer.stop()
                self.btn_play.setText("▶")
                self.statusBar().showMessage("End of video.", 4000)
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Timestamp from the file's own frame rate, not wall clock, so the
            # analysis is identical whatever speed you replay at.
            t = self._video_index / self.video_fps
            self._video_index += 1
            self.frame_queue.put((frame, t))
        # sliderMoved (the seek signal) only fires for user drags, so this
        # programmatic update cannot trigger a seek.
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(min(self._video_index,
                                      self.seek_slider.maximum()))
        self.seek_slider.blockSignals(False)
        self.play_pos_label.setText(f"{self._video_index} / {self.video_nframes}")

    def _return_to_camera(self):
        self.play_timer.stop()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.play_bar.setVisible(False)
        self.source_mode = 'camera'
        self._clear_rois()
        self._reset_averaging()
        self._drain_frame_queue()
        self.setWindowTitle("VRHEED - Precision Analyzer")
        if self.cam is not None:
            self._set_camera_controls_enabled(True)
            self._start_acquisition()
            self.statusBar().showMessage("Live camera resumed.", 4000)
        else:
            self.statusBar().showMessage("No camera available.", 4000)

    # -----------------------------------------------------------------------
    # UI loop (QTimer, main thread)
    # -----------------------------------------------------------------------

    def _ui_loop(self):
        """Timer tick: measure every queued frame, draw only the last one.

        Measurement and rendering have different rate requirements.  Every
        frame that reaches the queue must be measured, or a video replayed at
        10x contributes one sample in ten and the FFT sees an aliased,
        irregularly sampled trace; but drawing more than one frame per 16 ms
        tick is wasted work.  Any exception in the pipeline is logged once and
        the loop keeps running -- a bad frame must not end a growth run.
        """
        try:
            latest = None
            while True:
                try:
                    frame, t = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                latest = self._process_frame(frame, t)
            if latest is None:
                return
            self._render_frame(*latest)
            if self.histories:
                self._update_plots()
            if self.roi_boxes:
                self._update_slice_plot()
        except Exception as e:
            self._log_once("frame pipeline", e)

    def _log_once(self, where, exc):
        """Write an exception to the log once per distinct message.

        Must be called from inside the ``except`` block.  A malformed frame
        that recurs at 60 Hz would otherwise fill the 1 MB log in seconds and
        bury whatever came before it.
        """
        key = f"{where}:{type(exc).__name__}:{exc}"
        if key in self._logged_errors:
            return
        self._logged_errors.add(key)
        logger.exception("Error in %s (identical errors are not repeated)", where)
        first = str(exc).splitlines()[0] if str(exc) else ""
        self.statusBar().showMessage(
            f"Error in {where} (see {os.path.basename(LOG_PATH)}): "
            f"{type(exc).__name__}: {first}"[:200], 8000)

    def _process_frame(self, frame, t):
        """Measurement step for ONE source frame.

        Rotation, rolling average, per-ROI metrics, history append and window
        tracking -- everything whose result depends on this particular frame
        having been seen.  Returns ``(display_source, t)`` for _render_frame.
        """
        frame = self._apply_rotation(frame)
        self.last_raw_frame = frame
        self._raw_max = 65535.0 if frame.dtype == np.uint16 else 255.0

        # Measure actual FPS
        if self._last_frame_time is not None:
            dt = t - self._last_frame_time
            if dt > 0:
                self.fps_tracker.append(1.0 / dt)
                self.actual_fps = float(np.mean(self.fps_tracker))
                if self.frame_count % 15 == 0:   # update display at ~2 Hz
                    self.fps_display.setText(f"FPS: {self.actual_fps:.1f}")
        self._last_frame_time = t

        if self.frame_count % 15 == 0:
            sat = va.saturated_fraction(frame, self._raw_max)
            self.sat_display.setText(f"Sat: {sat * 100:.2f}%")
            self.sat_display.setStyleSheet(
                "color:#e74c3c;font-weight:bold;" if sat > 0.005 else
                "color:#f39c12;" if sat > 0.0005 else "color:#95a5a6;")

        # Frame averaging (display, and optionally measurement)
        avg = self._averaged_frame(frame)
        meas_frame = avg if self.avg_measure_cb.isChecked() else frame

        # ROI processing — measurement uses the full raw frame (zoom is display-only)
        fh_full, fw_full = frame.shape[:2]
        record = (not self.paused) and self._ensure_origin_ready(t)
        for i, roi in enumerate(self.roi_boxes):
            bx = roi['box']
            fx1 = max(0, int(bx[0] * fw_full));  fy1 = max(0, int(bx[1] * fh_full))
            fx2 = min(fw_full, int(bx[2] * fw_full))
            fy2 = min(fh_full, int(bx[3] * fh_full))

            bundle = None
            if fx2 > fx1 and fy2 > fy1:
                sub = meas_frame[fy1:fy2, fx1:fx2]
                bundle = self._measure_roi(roi, sub, fx1)
            self._roi_cache[i] = bundle

            if record and bundle is not None:
                value = bundle['values'].get(roi.get('metric', 'mean'))
                if value is not None and np.isfinite(value):
                    self.histories[i].append(float(value))
                    self.times[i].append(t - self._t_origin)
                self._apply_tracking(roi, bundle)

        self.frame_count += 1
        return avg, t

    def _render_frame(self, avg, t):
        """Display step: colour-map, zoom, overlays and the recording sink.

        Nothing here feeds the analysis; it can run for one frame out of many.
        """
        # Build display image (background-subtracted if reference set;
        # intensity measurements always use the raw/averaged frame)
        if (self.reference_frame is not None and
                self.reference_frame.shape == avg.shape):
            disp_src = np.clip(avg.astype(np.float32) -
                               self.reference_frame.astype(np.float32), 0, None)
        else:
            disp_src = avg

        f8 = self._to_8bit(disp_src)

        gamma = self.gamma_spin.value()
        if abs(gamma - 1.0) > 0.01:
            if gamma != self._gamma_cached:   # rebuild LUT only when gamma changes
                self._gamma_lut    = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
                self._gamma_cached = gamma
            f8 = self._gamma_lut[f8]

        cmap = COLORMAPS[self.cmap_combo.currentIndex()][1]
        display = (cv2.cvtColor(f8, cv2.COLOR_GRAY2BGR) if cmap is None
                   else cv2.applyColorMap(f8, cmap))

        # Build camera-resolution frame with ROI overlays for snap / recording
        rec_frame = display.copy()
        fh_rec, fw_rec = rec_frame.shape[:2]
        for i, roi in enumerate(self.roi_boxes):
            bx = roi['box']
            rx1, ry1 = int(bx[0] * fw_rec), int(bx[1] * fh_rec)
            rx2, ry2 = int(bx[2] * fw_rec), int(bx[3] * fh_rec)
            rc = (255, 255, 255) if i == self.active_roi else self.ROI_COLORS[i % 4]
            cv2.rectangle(rec_frame, (rx1, ry1), (rx2, ry2), rc, 2)
        self._last_display_frame = rec_frame

        if self.is_recording and self.video_writer is not None:
            if self._rec_size == (fw_rec, fh_rec):
                self.video_writer.write(rec_frame)
            else:
                self._toggle_recording()
                QMessageBox.warning(
                    self, "Recording stopped",
                    "The frame size changed while recording, so the file was "
                    "closed rather than silently dropping frames.")

        fh_full, fw_full = display.shape[:2]   # full camera-resolution dims

        # Apply zoom: crop display to the visible viewport
        if self._zoom_level > 1.0:
            vw = fw_full / self._zoom_level
            vh = fh_full / self._zoom_level
            vx1 = self._zoom_cx * fw_full - vw / 2
            vy1 = self._zoom_cy * fh_full - vh / 2
            vx1 = max(0.0, min(fw_full - vw, vx1))
            vy1 = max(0.0, min(fh_full - vh, vy1))
            vx2, vy2 = vx1 + vw, vy1 + vh
            # Update center to clamped position
            self._zoom_cx = (vx1 + vw / 2) / fw_full
            self._zoom_cy = (vy1 + vh / 2) / fh_full
            self._vp_x1 = vx1 / fw_full;  self._vp_y1 = vy1 / fh_full
            self._vp_w  = vw  / fw_full;  self._vp_h  = vh  / fh_full
            display = display[int(vy1):int(vy2), int(vx1):int(vx2)]
        else:
            self._vp_x1 = 0.0;  self._vp_y1 = 0.0
            self._vp_w  = 1.0;  self._vp_h  = 1.0

        win_w = max(self.cam_label.width(), 100)
        win_h = max(self.cam_label.height(), 100)
        h, w  = display.shape[:2]
        scale = min(win_w / w, win_h / h)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        off_x, off_y = (win_w - new_w) // 2, (win_h - new_h) // 2
        disp_sm    = cv2.resize(display, (new_w, new_h))
        final_view = cv2.copyMakeBorder(
            disp_sm, off_y, win_h - new_h - off_y, off_x, win_w - new_w - off_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Cache transform for mouse↔norm conversion
        self._off_x, self._off_y   = off_x, off_y
        self._new_w, self._new_h   = new_w, new_h
        self._frame_w, self._frame_h = fw_full, fh_full

        if self.sim_on_cb.isChecked():
            self._draw_simulation(final_view)

        # ROI overlays (boxes already updated by tracking in _process_frame)
        for i, roi in enumerate(self.roi_boxes):
            bx = roi['box']
            # Full-frame normalized → viewport normalized → display px
            dx1, dy1 = self._norm_to_disp(bx[0], bx[1])
            dx2, dy2 = self._norm_to_disp(bx[2], bx[3])
            color = (255, 255, 255) if i == self.active_roi else self.ROI_COLORS[i % 4]
            cv2.rectangle(final_view, (dx1, dy1), (dx2, dy2), color, 2)
            if i == self.active_roi:
                cv2.rectangle(final_view, (dx2 - 8, dy2 - 8), (dx2, dy2),
                              (255, 255, 255), -1)
            # Draggable horizontal slice line
            sl_y_norm = bx[1] + roi.get('slice_y', 0.5) * (bx[3] - bx[1])
            _, dsl_y = self._norm_to_disp(bx[0], sl_y_norm)
            if dy1 <= dsl_y <= dy2:
                cv2.line(final_view, (dx1, dsl_y), (dx2, dsl_y), color, 1)

        if self.temp_box:
            tb = self.temp_box   # temp_box lives in display coords while drawing
            cv2.rectangle(final_view, (tb[0], tb[1]), (tb[2], tb[3]), (200, 200, 200), 1)

        # BGR → QPixmap
        rgb = cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, win_w, win_h, win_w * 3, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qimg))

    def _to_8bit(self, src):
        """Map raw counts to 0-255 for display, honouring the contrast mode."""
        a = np.asarray(src, dtype=np.float32)
        if self.contrast_auto_cb.isChecked():
            lo, hi = float(a.min()), float(a.max())
        else:
            lo, hi = float(self.level_lo.value()), float(self.level_hi.value())
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    def _norm_to_disp(self, nx, ny):
        """Normalized full-frame coord → display-label pixel."""
        vx = (nx - self._vp_x1) / max(self._vp_w, 1e-9)
        vy = (ny - self._vp_y1) / max(self._vp_h, 1e-9)
        return (int(vx * self._new_w + self._off_x),
                int(vy * self._new_h + self._off_y))

    def _initial_x_span(self):
        """Opening width of the time axis, in seconds.

        Starting at the full history window (up to four hours) drew the first
        minutes of a growth as an unreadable sliver against the left edge, so
        the axis starts short and grows to meet the data instead.
        """
        return min(120.0, self.hist_spin.value() * 60.0)

    def _ensure_origin_ready(self, t):
        """Anchor the time axis to the first recorded sample."""
        if not self.roi_boxes:
            return False
        if self._t_origin is None:
            self._t_origin = t
            self._x_right = self._initial_x_span()
            self.plot_time.setXRange(0, self._x_right, padding=0)
        return True

    def _measure_roi(self, roi, sub, x_offset):
        """All quantities this ROI could track, computed once per frame."""
        base = va.roi_metrics(sub)
        if base is None:
            return None

        sy = roi.get('slice_y', 0.5)
        h = sub.shape[0]
        yc = int(sy * h)
        band = max(1, h // 10)
        lo = max(0, yc - band)
        hi = min(h, yc + band + 1)
        profile = sub[lo:hi].mean(axis=0).astype(np.float64)
        x_px = np.arange(x_offset, x_offset + profile.size, dtype=np.float64)

        pm = va.profile_metrics(profile, x_px,
                                prefer_fit=self.slice_fit_cb.isChecked())
        streaks = (va.find_streaks(profile, x_px)
                   if self.slice_streak_cb.isChecked() else [])
        spacing = va.mean_streak_spacing(streaks) if len(streaks) >= 2 else float('nan')

        fwhm = pm['fwhm'] if pm else float('nan')
        # Coherence length is the width of ONE streak.  Across a profile with
        # several streaks the half-max crossings span the whole group, which
        # produced a "FWHM" of hundreds of pixels and a sub-angstrom coherence
        # length.  Require a single resolved peak before reporting it.
        single = (pm is not None and np.isfinite(fwhm)
                  and fwhm < 0.5 * profile.size and len(streaks) <= 1)
        values = dict(base)
        values.update({
            'fwhm': fwhm,
            'coh': self._coherence_from_fwhm(fwhm) if single else float('nan'),
            'single_streak': single,
            'pos': pm['center'] if pm else float('nan'),
            'spacing': spacing,
        })
        return {'values': values, 'profile': profile, 'x_px': x_px,
                'pm': pm, 'streaks': streaks}

    def _coherence_from_fwhm(self, fwhm_px):
        """Streak width in pixels → in-plane coherence length in angstroms."""
        if not np.isfinite(fwhm_px) or fwhm_px <= 0:
            return float('nan')
        return va.coherence_length(
            fwhm_px, K=self._lattice_K,
            pixel_mm=self.pixel_spin.value(),
            camera_distance_mm=self.dist_spin.value(),
            energy_keV=self.energy_spin.value())

    def _apply_tracking(self, roi, bundle):
        """Re-centre an ROI on its feature (kSA-style window tracking)."""
        mode = roi.get('track', 'off')
        if mode == 'off':
            return
        v = bundle['values']
        fx = v['cx'] if mode == 'centroid' else v['px']
        fy = v['cy'] if mode == 'centroid' else v['py']
        if not (np.isfinite(fx) and np.isfinite(fy)):
            return
        bx = roi['box']
        w = bx[2] - bx[0];  h = bx[3] - bx[1]
        cx_now = bx[0] + w / 2.0;  cy_now = bx[1] + h / 2.0
        cx_new = cx_now + (bx[0] + fx * w - cx_now) * TRACK_GAIN
        cy_new = cy_now + (bx[1] + fy * h - cy_now) * TRACK_GAIN
        roi['box'] = self._sanitize_box(
            [cx_new - w / 2.0, cy_new - h / 2.0,
             cx_new + w / 2.0, cy_new + h / 2.0])

    # -----------------------------------------------------------------------
    # Plots  (PyQtGraph — setData() is O(n) C++, no Python rendering)
    # -----------------------------------------------------------------------

    def _update_plots(self, force=False):
        """Refresh the time traces every call; the spectra only when due.

        The time-trace setData is cheap C++ and runs for every displayed
        frame.  The FFT, peak markers and metric panel are recomputed at most
        every FFT_REFRESH_S, or at once when ``force`` is given or
        ``_fft_dirty`` is set (region moved, active ROI changed, seek).  The
        last spectrum per ROI is cached so the markers stay drawn in between.
        """
        sel_lo, sel_hi = self.selected_range   # absolute seconds (0.0,0.0 = full)
        current = set(self.histories.keys())

        # Remove curves for deleted ROIs
        for idx in set(self.time_curves) - current:
            self.plot_time.removeItem(self.time_curves.pop(idx))
        for idx in set(self.fft_curves) - current:
            self.plot_fft.removeItem(self.fft_curves.pop(idx))
        for idx in set(self.peak_markers) - current:
            self.plot_fft.removeItem(self.peak_markers.pop(idx))
        for idx in set(self._fft_cache) - current:
            self._fft_cache.pop(idx)

        now = time.monotonic()
        do_fft = (force or self._fft_dirty
                  or (now - self._last_fft_time) >= FFT_REFRESH_S)
        if do_fft:
            self._last_fft_time = now
            self._fft_dirty = False

        metrics_shown = False
        for i, history in self.histories.items():
            data = np.array(history, dtype=np.float64)
            # Real timestamps, one per sample.  An ROI drawn ten minutes into
            # a growth therefore starts ten minutes along the axis instead of
            # being redrawn from zero, and a saturated (wrapped) history keeps
            # its true times instead of sliding back toward the origin.
            t_sec = np.array(self.times[i], dtype=np.float64)
            if data.size != t_sec.size:          # defensive: never plot ragged
                n = min(data.size, t_sec.size)
                data, t_sec = data[-n:], t_sec[-n:]
            if data.size == 0:
                continue
            color = self.ROI_COLORS[i % 4]
            pen   = pg.mkPen(color=color, width=1)

            # Apply BG subtraction and/or normalization
            plot_data = data.copy()
            if self.time_bg_cb.isChecked():
                plot_data = plot_data - float(plot_data.min())
            if self.time_norm_cb.isChecked() and len(plot_data) > 1:
                peak = float(np.abs(plot_data).max())
                if peak != 0:
                    plot_data = plot_data / peak

            if i not in self.time_curves:
                self.time_curves[i] = self.plot_time.plot(x=t_sec, y=plot_data, pen=pen)
            else:
                self.time_curves[i].setData(x=t_sec, y=plot_data)

            if do_fft:
                # Region bounds are seconds; find them in this ROI's own timeline.
                if sel_hi > sel_lo:
                    i_s = int(np.searchsorted(t_sec, sel_lo, side='left'))
                    i_e = int(np.searchsorted(t_sec, sel_hi, side='right'))
                else:
                    i_s, i_e = 0, data.size
                sub_v = data[i_s:i_e]
                sub_t = t_sec[i_s:i_e]
                if sub_v.size < 16:
                    sub_v, sub_t = data, t_sec

                rate, mag, freq = va.compute_growth_rate(
                    sub_v, times=sub_t,
                    fmin=self.fmin_spin.value(), fmax=self.fmax_spin.value())
                self._fft_cache[i] = (rate, mag, freq, sub_t, sub_v)

            cached = self._fft_cache.get(i)
            if cached is None:
                continue
            rate, mag, freq, sub_t, sub_v = cached

            if freq is not None:
                fpen = pg.mkPen(color=color, width=1, style=Qt.SolidLine)
                if i not in self.fft_curves:
                    self.fft_curves[i] = self.plot_fft.plot(
                        x=freq, y=mag, pen=fpen, fillLevel=0,
                        brush=pg.mkBrush(*color, 40))
                elif do_fft:
                    self.fft_curves[i].setData(x=freq, y=mag)  # fast C++ update

                # Peak marker — dot + vertical line at detected growth rate
                peak_mag = float(mag[np.argmin(np.abs(freq - rate))]) if rate > 0 else 0.0
                if i not in self.peak_markers:
                    self.peak_markers[i] = pg.ScatterPlotItem(
                        x=[rate], y=[peak_mag], size=12,
                        pen=pg.mkPen('w', width=1),
                        brush=pg.mkBrush(*color))
                    self.plot_fft.addItem(self.peak_markers[i])
                elif do_fft:
                    self.peak_markers[i].setData(x=[rate], y=[peak_mag])

            if i == self.active_roi or (self.active_roi == -1 and i == 0):
                if do_fft:
                    self._update_metric_panel(i, rate, sub_t, sub_v)
                metrics_shown = True

        if not metrics_shown:
            self.active_roi_label.setText("Metrics: no ROIs yet")

        # Y label follows whatever the ROIs are tracking.
        keys = {r.get('metric', 'mean') for r in self.roi_boxes}
        self.plot_time.setLabel(
            'left', ROI_METRIC_LABEL[next(iter(keys))] if len(keys) == 1
            else "Mixed metrics")

        # Extend the x range only when data reaches 90% of the right edge, and
        # extend it by a large factor so the rescale is rare — a small nudge
        # every few frames would make the whole trace shimmer.
        if self.times and self._t_origin is not None:
            t_max = max((tt[-1] for tt in self.times.values() if tt), default=0.0)
            if t_max > self._x_right * 0.9:
                self._x_right = max(self._x_right * 1.6, t_max * 1.2)
                self.plot_time.setXRange(0, self._x_right, padding=0)

    def _update_metric_panel(self, i, rate, sub_t, sub_v):
        metric = self.roi_boxes[i].get('metric', 'mean') if i < len(self.roi_boxes) else 'mean'

        self.metric_labels['freq'].setText(f"{rate:.4f}" if rate > 0 else "---")
        if rate > 0:
            # One RHEED oscillation = one monolayer, so ML/s is the frequency
            # and the thickness rates follow from the ML thickness in angstroms.
            self.metric_labels['mls'].setText(f"{rate:.4f}")
            v = rate * self.ml_spin.value()                 # Å/s
            self.metric_labels['angs'].setText(f"{v:.3f}")
            self.metric_labels['nmmin'].setText(f"{v * 6.0:.3f}")     # Å/s → nm/min
            self.metric_labels['umhr'].setText(f"{v * 0.36:.4f}")     # Å/s → µm/hr
            self.metric_labels['period'].setText(f"{1.0 / rate:.3f}")
        else:
            for k in ('mls', 'angs', 'nmmin', 'umhr', 'period'):
                self.metric_labels[k].setText("---")

        # Peak counting and the damped-sine fit both walk the whole selection,
        # which can be tens of thousands of points.  Twice a second is far
        # faster than the numbers change and keeps the frame loop cheap.
        now = time.monotonic()
        if now - self._last_stat_time > 0.5:
            self._last_stat_time = now
            self._stat_cache = va.oscillation_stats(
                sub_t, sub_v, fmin=self.fmin_spin.value(),
                fmax=self.fmax_spin.value())
            self._fit_cache = va.damped_sine_fit(
                sub_t, sub_v, f0=rate if rate > 0 else None,
                fmin=self.fmin_spin.value(), fmax=self.fmax_spin.value())
            self._update_fit_label(sub_t)

        st = self._stat_cache
        if st:
            self.osc_label.setText(
                f"counted {st['n_peaks']} peaks · period "
                f"{st['period_s']:.2f} ± {st['period_std_s']:.2f} s "
                f"({st['rate_hz']:.4f} Hz) · damping "
                f"{st['damping'] * 100:+.1f} %/period")
        else:
            self.osc_label.setText("counted peaks: not enough oscillations yet")

        cname = self.COLOR_NAMES[i % 4]
        n_rois = len(self.histories)
        suffix = f" of {n_rois} — click another ROI to switch" if n_rois > 1 else ""
        self.active_roi_label.setText(
            f"Metrics: ROI {i} ({cname}) tracking "
            f"{ROI_METRIC_LABEL[metric].lower()}{suffix}")
        r, g, b2 = self.ROI_COLORS[i % 4]
        self.active_roi_label.setStyleSheet(f"color: rgb({r},{g},{b2}); font-size:9pt;")

        if metric != 'mean':
            self.osc_note.setText(
                f"ROI {i} is tracking {ROI_METRIC_LABEL[metric].lower()}; the "
                f"growth-rate numbers above are the FFT of that quantity, not "
                f"of the intensity.")
        else:
            self.osc_note.setText("")

    def _update_fit_label(self, sub_t):
        fit = self._fit_cache
        if not fit:
            self.fit_label.setText("damped-sine fit: needs ~2 clean oscillations")
            return
        rate, err = fit['rate_hz'], fit['rate_err_hz']
        # Thickness deposited over the selection, from the fitted rate.
        duration = float(sub_t[-1] - sub_t[0]) if len(sub_t) > 1 else 0.0
        thickness = rate * duration * self.ml_spin.value()
        err_txt = f" ± {err:.4f}" if np.isfinite(err) else ""
        # A decay far longer than the window is not a measurement of anything;
        # the fit has simply run to its bound.
        tau = fit['tau_s']
        tau_txt = (f"τ = {tau:.1f} s" if duration > 0 and tau < 10 * duration
                   else "no decay resolved")
        self.fit_label.setText(
            f"damped-sine fit: {rate:.4f}{err_txt} Hz · {tau_txt} "
            f"· R² = {fit['r2']:.3f} · {thickness:.1f} Å over "
            f"{duration:.0f} s ({fit['n_periods']:.1f} periods)")

    def _on_region_changed(self):
        # Store in absolute seconds; per-ROI sample indices are computed in
        # _update_plots against each ROI's own timestamps.
        lo, hi = self.region.getRegion()
        self.selected_range = (lo, hi)
        if self.histories:
            self._update_plots(force=True)

    def _fit_x_range(self):
        t_max = max((tt[-1] for tt in self.times.values() if tt), default=0.0)
        self._x_right = max(t_max * 1.05, 10.0)
        self.plot_time.setXRange(0, self._x_right, padding=0)

    # -----------------------------------------------------------------------
    # Mouse / ROI
    # Mouse events arrive in display coords; we convert to normalized frame
    # coords immediately so stored boxes are resolution-independent.
    # -----------------------------------------------------------------------

    def _disp_to_norm(self, x, y):
        """Convert display-label pixel → normalized full-frame coord (0-1)."""
        nw = max(self._new_w, 1);  nh = max(self._new_h, 1)
        vp_nx = (x - self._off_x) / nw
        vp_ny = (y - self._off_y) / nh
        return self._vp_x1 + vp_nx * self._vp_w, self._vp_y1 + vp_ny * self._vp_h

    @staticmethod
    def _sanitize_box(box):
        """Order, clamp and size-floor a normalized ROI box.

        A box dragged inside-out or off the frame produced an empty slice, so
        that ROI silently stopped recording while the others kept going.
        """
        x1, x2 = sorted((float(box[0]), float(box[2])))
        y1, y2 = sorted((float(box[1]), float(box[3])))
        x1 = min(max(x1, 0.0), 1.0 - MIN_ROI_FRAC)
        y1 = min(max(y1, 0.0), 1.0 - MIN_ROI_FRAC)
        x2 = min(max(x2, x1 + MIN_ROI_FRAC), 1.0)
        y2 = min(max(y2, y1 + MIN_ROI_FRAC), 1.0)
        return [x1, y1, x2, y2]

    def _on_mouse_down(self, x, y, button):
        nx, ny = self._disp_to_norm(x, y)

        if self._pick_origin and button == Qt.LeftButton:
            self._set_sim_origin(nx, ny)
            self.btn_pick.setChecked(False)
            return

        if button == Qt.RightButton:
            for i, roi in enumerate(self.roi_boxes):
                bx = roi['box']
                if bx[0] < nx < bx[2] and bx[1] < ny < bx[3]:
                    self._delete_roi(i); return
            return

        self.start_mouse = (x, y)
        self.active_roi  = -1
        for i, roi in enumerate(self.roi_boxes):
            bx = roi['box']
            # Resize handle: check bottom-right corner in display coords
            dx2, dy2 = self._norm_to_disp(bx[2], bx[3])
            if abs(x - dx2) < 15 and abs(y - dy2) < 15:
                self.active_roi, self.drag_mode = i, 'resize'
                self._sync_metric_combo(); return
            # Slice line hit-test — must come before 'move'
            sl_y_norm = bx[1] + roi.get('slice_y', 0.5) * (bx[3] - bx[1])
            _, dsl_y = self._norm_to_disp(bx[0], sl_y_norm)
            if bx[0] < nx < bx[2] and abs(y - dsl_y) < 8:
                self.active_roi, self.drag_mode = i, 'slice_line'
                self._sync_metric_combo(); return
            if bx[0] < nx < bx[2] and bx[1] < ny < bx[3]:
                self.active_roi, self.drag_mode = i, 'move'
                self._sync_metric_combo(); return
        self.drag_mode, self.temp_box = 'create', [x, y, x, y]

    def _on_mouse_move(self, x, y):
        if self.drag_mode == 'create':
            self.temp_box[2], self.temp_box[3] = x, y
        elif self.drag_mode == 'move' and self.active_roi != -1:
            # Delta in display px → delta in normalized coords, clamped so the
            # box slides along the edge instead of leaving the frame.
            nw = max(self._new_w, 1);  nh = max(self._new_h, 1)
            dx, dy = x - self.start_mouse[0], y - self.start_mouse[1]
            bx = self.roi_boxes[self.active_roi]['box']
            w = bx[2] - bx[0];  h = bx[3] - bx[1]
            nx1 = min(max(bx[0] + dx / nw * self._vp_w, 0.0), 1.0 - w)
            ny1 = min(max(bx[1] + dy / nh * self._vp_h, 0.0), 1.0 - h)
            bx[0], bx[1] = nx1, ny1
            bx[2], bx[3] = nx1 + w, ny1 + h
            self.start_mouse = (x, y)
        elif self.drag_mode == 'slice_line' and self.active_roi != -1:
            bx = self.roi_boxes[self.active_roi]['box']
            _, ny = self._disp_to_norm(x, y)
            roi_h = bx[3] - bx[1]
            if roi_h > 0:
                sy = (ny - bx[1]) / roi_h
                self.roi_boxes[self.active_roi]['slice_y'] = max(0.02, min(0.98, sy))
        elif self.drag_mode == 'resize' and self.active_roi != -1:
            roi = self.roi_boxes[self.active_roi]
            bx = roi['box']
            nx, ny = self._disp_to_norm(x, y)
            roi['box'] = self._sanitize_box([bx[0], bx[1], nx, ny])

    def _on_mouse_up(self, *_):
        if self.drag_mode == 'create' and self.temp_box:
            b = self.temp_box
            if abs(b[2] - b[0]) > 5 and abs(b[3] - b[1]) > 5:
                x1, y1 = self._disp_to_norm(min(b[0], b[2]), min(b[1], b[3]))
                x2, y2 = self._disp_to_norm(max(b[0], b[2]), max(b[1], b[3]))
                self._add_roi(self._sanitize_box([x1, y1, x2, y2]))
        self.drag_mode, self.temp_box = None, None

    def _add_roi(self, box, slice_y=0.5, metric=None, track=None):
        idx = len(self.roi_boxes)
        if metric is None:
            metric = self.metric_combo.currentData() or 'mean'
        if track is None:
            track = self.track_combo.currentData() or 'off'
        self.roi_boxes.append({'box': box, 'slice_y': slice_y,
                               'metric': metric, 'track': track})
        self.histories[idx] = deque(maxlen=self._history_maxlen())
        self.times[idx]     = deque(maxlen=self._history_maxlen())
        self.active_roi = idx
        self._sync_metric_combo()
        # Give the FFT selector a sensible default window on the first ROI.
        if idx == 0:
            self.region.setRegion([0.0, min(60.0, self._initial_x_span() * 0.5)])
        return idx

    def _sync_metric_combo(self):
        """Show the active ROI's metric/tracking in the combos.

        Called whenever the active ROI changes, so it also flags the spectrum
        cache: the metric panel must switch ROIs at once, not after the next
        250 ms refresh.  With no active ROI (after a delete) the combos keep
        their last selection, which is what a new ROI will inherit.
        """
        self._fft_dirty = True
        # The peak-count and damped-sine lines have their own 0.5 s throttle
        # (_update_metric_panel); without this they keep showing the previous
        # ROI's numbers for up to half a second after the switch.
        self._last_stat_time = 0.0
        if not (0 <= self.active_roi < len(self.roi_boxes)):
            return
        roi = self.roi_boxes[self.active_roi]
        for combo, key in ((self.metric_combo, roi.get('metric', 'mean')),
                           (self.track_combo, roi.get('track', 'off'))):
            i = combo.findData(key)
            if i >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)

    def _on_track_changed(self):
        if not (0 <= self.active_roi < len(self.roi_boxes)):
            return
        key = self.track_combo.currentData()
        self.roi_boxes[self.active_roi]['track'] = key
        self.statusBar().showMessage(
            f"ROI {self.active_roi} tracking: {TRACK_LABEL[key].lower()}", 4000)

    def _on_metric_changed(self):
        if not (0 <= self.active_roi < len(self.roi_boxes)):
            return
        key = self.metric_combo.currentData()
        roi = self.roi_boxes[self.active_roi]
        if roi.get('metric') == key:
            return
        roi['metric'] = key
        # The old samples measured a different quantity; keeping them would
        # splice two unrelated traces into one curve.
        self.histories[self.active_roi].clear()
        self.times[self.active_roi].clear()
        self._fft_cache.pop(self.active_roi, None)
        self._fft_dirty = True
        self.statusBar().showMessage(
            f"ROI {self.active_roi} now tracks {ROI_METRIC_LABEL[key].lower()} "
            f"— its history was cleared.", 5000)

    def _on_wheel(self, x, y, delta):
        # Frame point currently under the cursor (full-frame normalized)
        nx, ny = self._disp_to_norm(x, y)

        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(1.0, min(20.0, self._zoom_level * factor))

        if new_zoom <= 1.0:
            self._reset_zoom()
            return

        # Cursor position as a fraction within the displayed image [0, 1]
        nw = max(self._new_w, 1);  nh = max(self._new_h, 1)
        px = (x - self._off_x) / nw
        py = (y - self._off_y) / nh

        # Keep point (nx, ny) fixed at display fraction (px, py):
        #   vp_x1_new + px * (1/new_zoom) = nx
        #   cx_new = vp_x1_new + 0.5/new_zoom = nx + (0.5 - px) / new_zoom
        half = 0.5 / new_zoom
        self._zoom_cx = nx + (0.5 - px) / new_zoom
        self._zoom_cy = ny + (0.5 - py) / new_zoom
        # Clamp so viewport stays within [0, 1]
        self._zoom_cx = max(half, min(1.0 - half, self._zoom_cx))
        self._zoom_cy = max(half, min(1.0 - half, self._zoom_cy))
        self._zoom_level = new_zoom
        self.zoom_label.setText(f"Zoom: {new_zoom:.1f}×")

    def _delete_active_roi(self):
        if 0 <= self.active_roi < len(self.roi_boxes):
            self._delete_roi(self.active_roi)

    def _delete_roi(self, idx):
        self.roi_boxes.pop(idx)
        n = len(self.roi_boxes)
        self.histories = {i: self.histories[i if i < idx else i + 1] for i in range(n)}
        self.times     = {i: self.times[i if i < idx else i + 1] for i in range(n)}
        # Indices above idx shift down, so every cached per-index result is
        # now attached to the wrong ROI.
        self._roi_cache = {}
        self._fft_cache = {}
        self.active_roi = -1
        self._sync_metric_combo()
        if not self.roi_boxes:
            self._reset_time_axis()

    def _clear_rois(self):
        self.roi_boxes = []; self.histories = {}; self.times = {}
        self.active_roi = -1; self._roi_cache = {}; self._fft_cache = {}
        self._fft_dirty = True
        for c in list(self.time_curves.values()): self.plot_time.removeItem(c)
        for c in list(self.fft_curves.values()):  self.plot_fft.removeItem(c)
        for c in list(self.peak_markers.values()): self.plot_fft.removeItem(c)
        for c in list(self.slice_curves.values()): self.plot_slice.removeItem(c)
        for d in list(self.fwhm_items.values()):
            for item in d.values(): self.plot_slice.removeItem(item)
        for ln in self.streak_lines: self.plot_slice.removeItem(ln)
        self.time_curves.clear(); self.fft_curves.clear(); self.peak_markers.clear()
        self.slice_curves.clear(); self.fwhm_items.clear(); self.streak_lines.clear()
        self._reset_time_axis()
        self.active_roi_label.setText("Metrics: no ROIs yet")
        self.osc_label.setText("")
        for v in self.metric_labels.values():
            v.setText("---")

    def _reset_time_axis(self):
        """Put the time axis back to its initial span.

        Without this the axis stayed stretched to the previous run's length,
        so a fresh ROI drew a 2-pixel-wide trace at the far left.
        """
        self._t_origin = None
        self._x_right = self._initial_x_span()
        self.plot_time.setXRange(0, self._x_right, padding=0)
        self.selected_range = (0.0, 0.0)
        self.region.blockSignals(True)
        self.region.setRegion([0.0, min(60.0, self._x_right * 0.5)])
        self.region.blockSignals(False)

    # -----------------------------------------------------------------------
    # Rotation
    # -----------------------------------------------------------------------

    def _apply_rotation(self, frame):
        base = self._rotation_deg % 360
        if base == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif base == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif base == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fine = self.rot_fine_spin.value()
        if abs(fine) > 0.01:
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), fine, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
        return frame

    def _on_rotate_cw(self):
        self._rotate_by(90)

    def _on_rotate_ccw(self):
        self._rotate_by(-90)

    def _rotate_by(self, deg):
        if self.is_recording:
            self._toggle_recording()
        self._rotation_deg = (self._rotation_deg + deg) % 360
        self._reset_averaging()
        self._clear_rois()
        self._clear_reference()
        self.statusBar().showMessage(
            f"Rotated → {self._rotation_deg}° total — ROIs and reference cleared.",
            4000)

    def _on_rot_fine_changed(self):
        self._clear_reference()
        self._reset_averaging()
        fine = self.rot_fine_spin.value()
        if abs(fine) > 0.01:
            self.statusBar().showMessage(
                f"Fine rotation {fine:+.1f}° — reference cleared.", 3000)

    # -----------------------------------------------------------------------
    # Slice plot, FWHM, streaks and lattice readout
    # -----------------------------------------------------------------------

    def _update_slice_plot(self):
        current = set(range(len(self.roi_boxes)))
        # Remove stale curves/items
        for idx in set(self.slice_curves) - current:
            self.plot_slice.removeItem(self.slice_curves.pop(idx))
        for idx in set(self.fwhm_items) - current:
            for item in self.fwhm_items.pop(idx).values():
                self.plot_slice.removeItem(item)

        for i in range(len(self.roi_boxes)):
            bundle = self._roi_cache.get(i)
            if bundle is None:
                continue
            profile, x_px, pm = bundle['profile'], bundle['x_px'], bundle['pm']
            color = self.ROI_COLORS[i % 4]
            pen   = pg.mkPen(color=color, width=1)

            if i not in self.slice_curves:
                self.slice_curves[i] = self.plot_slice.plot(x=x_px, y=profile, pen=pen)
            else:
                self.slice_curves[i].setData(x=x_px, y=profile)

            if pm is not None:
                lbl = f"FWHM {pm['fwhm']:.2f} px @ {pm['center']:.1f}"
                if pm['method'] == 'gaussian':
                    lbl += f"  (fit R²={pm['r2']:.3f})"
                if i not in self.fwhm_items:
                    half_line = pg.InfiniteLine(
                        angle=0, pos=pm['half'],
                        pen=pg.mkPen(color=color, width=1, style=Qt.DashLine))
                    text = pg.TextItem(lbl, color=color, anchor=(0.5, 1.0))
                    self.plot_slice.addItem(half_line)
                    self.plot_slice.addItem(text)
                    self.fwhm_items[i] = {'half_line': half_line, 'text': text}
                else:
                    self.fwhm_items[i]['half_line'].setPos(pm['half'])
                    self.fwhm_items[i]['text'].setText(lbl)
                self.fwhm_items[i]['text'].setPos(
                    float(pm['center']), float(profile.max()))
            elif i in self.fwhm_items:
                for item in self.fwhm_items.pop(i).values():
                    self.plot_slice.removeItem(item)

        self._update_streak_markers()

    def _update_streak_markers(self):
        idx = self.active_roi if self.active_roi >= 0 else 0
        bundle = self._roi_cache.get(idx)
        streaks = bundle['streaks'] if bundle else []

        # Reuse the existing lines; creating and destroying items every frame
        # makes pyqtgraph rebuild its scene graph and the plot visibly stutters.
        while len(self.streak_lines) > len(streaks):
            self.plot_slice.removeItem(self.streak_lines.pop())
        while len(self.streak_lines) < len(streaks):
            ln = pg.InfiniteLine(angle=90, pen=pg.mkPen('#ecf0f1', width=1,
                                                        style=Qt.DotLine))
            self.plot_slice.addItem(ln)
            self.streak_lines.append(ln)
        for ln, s in zip(self.streak_lines, streaks):
            ln.setPos(s['pos'])

        if len(streaks) >= 2:
            dx = va.mean_streak_spacing(streaks)
            self._last_spacing_px = dx
            self.slice_info.setText(f"{len(streaks)} streaks · Δx = {dx:.2f} px")
        else:
            dx = float('nan')
            self._last_spacing_px = dx
            self.slice_info.setText("")
        if self.slice_streak_cb.isChecked():
            self._update_lattice_readout(dx)

    def _update_binning_note(self):
        """Keep the Lattice tab honest about the binning in force."""
        n = self.cam.binning if self.cam is not None else (
            self._bin_group.checkedId() or 1)
        if n and n > 1:
            self.bin_note.setText(
                f"Binning is {n}×{n}: the pixel pitch above must be your "
                f"sensor pitch × {n} (× lens magnification).")
        else:
            self.bin_note.setText("")

    def _update_lattice_readout(self, dx_px):
        """Report whatever the active ROI can currently support.

        Lattice spacing needs two or more streaks in the ROI; coherence length
        needs exactly one.  The two are mutually exclusive, so the panel shows
        whichever applies rather than blanking out for the other.
        """
        idx = self.active_roi if self.active_roi >= 0 else 0
        lines = []

        if np.isfinite(dx_px) and dx_px > 0:
            if self._lattice_K is not None:
                a = va.lattice_from_calibration(dx_px, self._lattice_K)
                how = "calibrated"
            else:
                a = va.spacing_to_lattice(dx_px, self.pixel_spin.value(),
                                          self.dist_spin.value(),
                                          self.energy_spin.value())
                how = "geometry"
            self._last_a = a
            lines.append(f"Δx = {dx_px:.2f} px   →   a∥ = {a:.4f} Å  ({how})")
            if self._a_ref is not None:
                lines.append(f"strain vs {self._a_ref:.4f} Å  =  "
                             f"{va.strain_percent(a, self._a_ref):+.3f} %")
        else:
            lines.append(f"ROI {idx} spans fewer than two streaks — widen it "
                         f"to measure lattice spacing.")

        # Streak width converts with the same lambda*L/x relation, so the same
        # calibration gives the in-plane coherence length for free.
        bundle = self._roi_cache.get(idx)
        if bundle:
            coh = bundle['values'].get('coh', float('nan'))
            if np.isfinite(coh):
                lines.append(f"streak FWHM = {bundle['values']['fwhm']:.2f} px "
                             f"→ coherence length = {coh:.1f} Å")
            else:
                lines.append("coherence length needs an ROI around a single "
                             "streak.")

        lines.append(f"λ = {va.wavelength_A(self.energy_spin.value() * 1000.0):.5f} Å "
                     f"at {self.energy_spin.value():.1f} keV")
        self.lat_result.setText("\n".join(lines))

    def _calibrate_lattice(self):
        if not np.isfinite(self._last_spacing_px):
            QMessageBox.information(
                self, "Calibrate",
                "No streak spacing measured yet.\n\n"
                "Tick “Mark streaks” under the slice plot and place an ROI "
                "across at least two streaks.")
            return
        self._lattice_K = va.calibration_constant(
            self.known_a_spin.value(), self._last_spacing_px)
        self._update_lattice_readout(self._last_spacing_px)
        self.statusBar().showMessage(
            f"Calibrated: K = {self._lattice_K:.2f} Å·px from "
            f"a = {self.known_a_spin.value():.4f} Å at "
            f"Δx = {self._last_spacing_px:.2f} px. "
            f"Screen distance and pixel pitch are no longer used.", 8000)

    def _set_strain_reference(self):
        if not np.isfinite(self._last_a):
            QMessageBox.information(self, "Strain reference",
                                    "No lattice constant measured yet.")
            return
        self._a_ref = self._last_a
        self._update_lattice_readout(self._last_spacing_px)
        self.statusBar().showMessage(
            f"Strain reference set to a = {self._a_ref:.4f} Å.", 5000)

    # -----------------------------------------------------------------------
    # Kinematic simulation overlay
    # -----------------------------------------------------------------------

    def _mark_sim_dirty(self, *_):
        self._sim_dirty = True

    def _set_sim_origin(self, nx, ny):
        """Shadow-edge origin in normalized frame coords, with its readout."""
        self._sim_origin = (min(max(float(nx), 0.0), 1.0),
                            min(max(float(ny), 0.0), 1.0))
        self.sim_origin_label.setText(
            f"origin: {self._sim_origin[0]:.3f}, {self._sim_origin[1]:.3f}")

    def _on_pick_toggled(self, on):
        self._pick_origin = on
        if on:
            self.statusBar().showMessage(
                "Click on the image where the beam axis meets the shadow edge.",
                0)
        else:
            self.statusBar().clearMessage()

    def _sim_data(self):
        if self._sim_dirty or not self._sim_pattern:
            try:
                self._sim_pattern = va.simulate_pattern(
                    self.sim_a.value(), self.sim_b.value(), self.sim_gamma.value(),
                    self.sim_azim.value(), self.energy_spin.value(),
                    self.sim_theta.value(), self.dist_spin.value(),
                    hk_max=self.sim_order.value())
            except Exception:
                self._sim_pattern = []
            self._sim_dirty = False
        return self._sim_pattern

    def _fit_sim_scale(self):
        """Set px/mm so the simulated first order matches the measured streaks."""
        if not np.isfinite(self._last_spacing_px):
            QMessageBox.information(
                self, "Scale from streaks",
                "Measure a streak spacing first (tick “Mark streaks”).")
            return
        pat = self._sim_data()
        firsts = [abs(p['x_mm']) for p in pat
                  if abs(p['h']) == 1 and p['k'] == 0 and abs(p['x_mm']) > 1e-9]
        if not firsts:
            QMessageBox.information(self, "Scale from streaks",
                                    "The simulation has no first-order streak.")
            return
        self.sim_ppm.setValue(self._last_spacing_px / min(firsts))
        self.statusBar().showMessage(
            f"Overlay scale set to {self.sim_ppm.value():.3f} px/mm.", 5000)

    def _draw_simulation(self, view):
        pat = self._sim_data()
        if not pat:
            return
        ppm = self.sim_ppm.value()
        ox, oy = self._sim_origin
        fw, fh = max(self._frame_w, 1), max(self._frame_h, 1)

        # Shadow edge
        _, ey = self._norm_to_disp(ox, oy)
        cv2.line(view, (0, ey), (view.shape[1], ey), (120, 120, 120), 1)

        # Laue arcs
        for r_mm in va.laue_radii_mm(pat, tol_mm=max(0.2, 2.0 / max(ppm, 1e-6))):
            rx = int(r_mm * ppm / fw * self._new_w * (1.0 / max(self._vp_w, 1e-9)))
            ry = int(r_mm * ppm / fh * self._new_h * (1.0 / max(self._vp_h, 1e-9)))
            ex, _ = self._norm_to_disp(ox, oy)
            if 2 < rx < 8000 and 2 < ry < 8000:
                cv2.ellipse(view, (ex, ey), (rx, ry), 0, 180, 360,
                            (90, 90, 90), 1)

        for p in pat:
            nx = ox + p['x_mm'] * ppm / fw
            ny = oy - p['y_mm'] * ppm / fh
            if not (-0.2 < nx < 1.2 and -0.2 < ny < 1.2):
                continue
            dx, dy = self._norm_to_disp(nx, ny)
            if p['specular']:
                cv2.circle(view, (dx, dy), 7, (255, 255, 255), 2)
                cv2.putText(view, "00", (dx + 9, dy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                cv2.circle(view, (dx, dy), 4, (0, 255, 200), 1)
                if abs(p['h']) <= 1 and abs(p['k']) <= 1:
                    cv2.putText(view, f"{p['h']}{p['k']}", (dx + 6, dy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1)

    # -----------------------------------------------------------------------
    # History helpers
    # -----------------------------------------------------------------------

    def _history_maxlen(self):
        """Samples to keep = history_minutes × 60 × frame rate.

        In file mode the file's own frame rate is the truth; live, use
        whichever of the target and measured rates is higher so a camera
        running faster than asked still gets the full requested duration.
        """
        if self.source_mode == 'file' and self.video_cap is not None:
            rate = max(self.video_fps, 1.0)
        else:
            rate = max(self.fps_spin.value(), self.actual_fps, 1.0)
        return max(256, int(self.hist_spin.value() * 60 * rate))

    def _on_history_changed(self):
        """Resize every existing deque to the new maxlen, preserving data."""
        new_len = self._history_maxlen()
        for store in (self.histories, self.times):
            for idx, old in store.items():
                store[idx] = deque(old, maxlen=new_len)
        if not self.roi_boxes:
            self._reset_time_axis()

    # -----------------------------------------------------------------------
    # Snap / Record
    # -----------------------------------------------------------------------

    def _snap_image(self):
        if self.last_raw_frame is None:
            QMessageBox.warning(self, "Snap", "No frame available yet.")
            return
        default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save RHEED Image", self._dialog_path(default_name),
            "TIFF (*.tif *.tiff);;PNG (*.png)")
        if not path:
            return
        self._remember_dir(path)
        # Save raw grayscale frame (preserves original bit depth)
        cv2.imwrite(path, self.last_raw_frame)
        # Also save colorized display frame and the settings that produced it
        base = os.path.splitext(path)[0]
        if self._last_display_frame is not None:
            cv2.imwrite(base + "_display.png", self._last_display_frame)
        try:
            with open(base + "_meta.txt", 'w') as f:
                for key, val in self._collect_metadata():
                    f.write(f"{key}: {val}\n")
        except Exception:
            pass
        self.statusBar().showMessage(
            f"Saved {os.path.basename(path)} (+ _display.png, _meta.txt)", 5000)

    def _toggle_recording(self):
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self._rec_size = None
            self.btn_record.setText("REC VIDEO")
            self.btn_record.setStyleSheet("background:#c0392b;color:white;font-weight:bold;")
            self.statusBar().showMessage("Recording stopped.", 4000)
        else:
            # Start recording — ask for file path first
            if self._last_display_frame is None:
                QMessageBox.warning(self, "Record", "No frame available yet.")
                return
            default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save RHEED Video", self._dialog_path(default_name),
                "MP4 (*.mp4);;AVI (*.avi)")
            if not path:
                return
            self._remember_dir(path)
            h, w = self._last_display_frame.shape[:2]
            fps  = max(self.actual_fps, 1.0)
            ext  = os.path.splitext(path)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'MJPG'))
            self.video_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not self.video_writer.isOpened():
                QMessageBox.critical(self, "Record", "Could not open video writer.")
                self.video_writer = None
                return
            self._rec_size = (w, h)
            self.is_recording = True
            self.btn_record.setText("STOP REC")
            self.btn_record.setStyleSheet(
                "background:#e74c3c;color:white;font-weight:bold;"
                "border: 2px solid #ff0000;")
            self.statusBar().showMessage(
                f"Recording {w}×{h} @ {fps:.1f} fps → {os.path.basename(path)}. "
                f"Fixed contrast levels are recommended so the brightness does "
                f"not flicker.", 8000)

    # -----------------------------------------------------------------------
    # Session / CSV
    # -----------------------------------------------------------------------

    # Simulate-tab spin boxes by their session/settings key.
    def _sim_spins(self):
        return (('a', self.sim_a), ('b', self.sim_b), ('gamma', self.sim_gamma),
                ('azimuth', self.sim_azim), ('theta', self.sim_theta),
                ('ppm', self.sim_ppm))

    def _save_session(self, path=None):
        path = self._path_arg(path)
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save ROIs", self._dialog_path(), "JSON (*.json)")
            if not path:
                return
        self._remember_dir(path)
        payload = {
            'version': 3,
            'app_version': __version__,
            'rois': [{'box': r['box'], 'slice_y': r.get('slice_y', 0.5),
                      'metric': r.get('metric', 'mean'),
                      'track': r.get('track', 'off')} for r in self.roi_boxes],
            'lattice': {
                'K': self._lattice_K, 'a_ref': self._a_ref,
                'energy_keV': self.energy_spin.value(),
                'distance_mm': self.dist_spin.value(),
                'pixel_mm': self.pixel_spin.value(),
                'known_a': self.known_a_spin.value(),
            },
            'analysis': {
                'ml_A': self.ml_spin.value(),
                'fmin': self.fmin_spin.value(), 'fmax': self.fmax_spin.value(),
                'history_min': self.hist_spin.value(),
            },
            # v3: the simulated-pattern setup belongs with the ROIs because
            # both describe the same geometry of the same sample.
            'sim': dict({key: spin.value() for key, spin in self._sim_spins()},
                        on=self.sim_on_cb.isChecked(),
                        order=self.sim_order.value(),
                        origin=list(self._sim_origin)),
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        self.statusBar().showMessage(f"Saved → {os.path.basename(path)}", 5000)

    def _load_session(self, path=None):
        path = self._path_arg(path)
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load ROIs", self._dialog_path(), "JSON (*.json)")
            if not path:
                return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load ROIs", f"Could not read file:\n{e}")
            return
        self._remember_dir(path)

        self._clear_rois()
        for item in data.get('rois', []):
            # Backward-compat: v1 stored bare lists, then dicts without a metric
            if isinstance(item, list):
                box, sy, metric, track = item, 0.5, 'mean', 'off'
            else:
                box = item.get('box')
                sy = item.get('slice_y', 0.5)
                metric = item.get('metric', 'mean')
                track = item.get('track', 'off')
            if not box or len(box) != 4:
                continue
            if metric not in ROI_METRIC_LABEL:
                metric = 'mean'
            if track not in TRACK_LABEL:
                track = 'off'
            self._add_roi(self._sanitize_box(box), sy, metric, track)

        lat = data.get('lattice') or {}
        self._lattice_K = lat.get('K')
        self._a_ref = lat.get('a_ref')
        for key, spin in (('energy_keV', self.energy_spin),
                          ('distance_mm', self.dist_spin),
                          ('pixel_mm', self.pixel_spin),
                          ('known_a', self.known_a_spin)):
            if lat.get(key) is not None:
                spin.setValue(float(lat[key]))
        ana = data.get('analysis') or {}
        for key, spin in (('ml_A', self.ml_spin), ('fmin', self.fmin_spin),
                          ('fmax', self.fmax_spin), ('history_min', self.hist_spin)):
            if ana.get(key) is not None:
                spin.setValue(float(ana[key]))

        # v3 only; v1/v2 files simply leave the Simulate tab as it is.
        sim = data.get('sim') or {}
        for key, spin in self._sim_spins():
            if sim.get(key) is not None:
                spin.setValue(float(sim[key]))
        if sim.get('order') is not None:
            self.sim_order.setValue(int(sim['order']))
        if sim.get('on') is not None:
            self.sim_on_cb.setChecked(bool(sim['on']))
        origin = sim.get('origin')
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            self._set_sim_origin(*origin)
        self._mark_sim_dirty()

        # The calibration and geometry just changed, so the lattice readout
        # (which caches its last numbers) must be recomputed.
        self._update_lattice_readout(self._last_spacing_px)
        self.statusBar().showMessage(
            f"Loaded {len(self.roi_boxes)} ROI(s) from {os.path.basename(path)} "
            f"(v{data.get('version', 1)})", 5000)

    def _collect_metadata(self):
        """Return a list of (key, value) strings describing current settings."""
        # Whatever the backend knows about the camera: vendor, model, serial,
        # pixel format, binning.  Backends fill in what their SDK exposes and
        # leave out the rest, so the header never carries invented values.
        cam_meta = {}
        if self.cam is not None:
            try:
                cam_meta = self.cam.metadata()
            except Exception:
                logger.exception("Could not read camera metadata")
        cam_meta.setdefault("Binning",
                            f"{self._bin_group.checkedId() or 1}×"
                            f"{self._bin_group.checkedId() or 1}")

        roi_lines = []
        for i, roi in enumerate(self.roi_boxes):
            b = roi['box']
            roi_lines.append(
                f"  ROI_{i} ({self.COLOR_NAMES[i % 4]}, "
                f"{ROI_METRIC_LABEL[roi.get('metric', 'mean')]}): "
                f"x=[{b[0]:.4f}, {b[2]:.4f}]  y=[{b[1]:.4f}, {b[3]:.4f}]")

        meta = [
            ("Saved",              datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("VRHEED version",     __version__),
            ("Log file",           LOG_PATH),
            ("Source",             self.source_mode),
        ] + [(k, v) for k, v in cam_meta.items()
             if k not in ("Frame width (px)", "Frame height (px)", "Binning")] + [
            ("Frame width (px)",   str(self._frame_w)),
            ("Frame height (px)",  str(self._frame_h)),
            ("Binning",            cam_meta["Binning"]),
            ("Rotation (deg)",     f"{self._rotation_deg} + {self.rot_fine_spin.value():+.1f}"),
            ("Gain (dB)",          f"{self.gain_spin.value():.2f}"),
            ("Exposure (ms)",      f"{self.exp_spin.value():.2f}"),
            ("Target FPS",         f"{self.fps_spin.value():.1f}"),
            ("Actual FPS",         f"{self.actual_fps:.2f}"),
            ("Frame average (N)",  str(self.avg_spin.value())),
            ("Averaged measurement", str(self.avg_measure_cb.isChecked())),
            ("Gamma (display)",    f"{self.gamma_spin.value():.2f}"),
            ("Colormap",           self.cmap_combo.currentText()),
            ("Contrast",           "auto" if self.contrast_auto_cb.isChecked()
                                   else f"fixed [{self.level_lo.value()}, "
                                        f"{self.level_hi.value()}]"),
            ("ML thickness (Å)",   f"{self.ml_spin.value():.3f}"),
            ("FFT f_min (Hz)",     f"{self.fmin_spin.value():.4f}"),
            ("FFT f_max (Hz)",     f"{self.fmax_spin.value():.2f}"),
            ("History window (min)", f"{self.hist_spin.value():.0f}"),
            ("Beam energy (keV)",  f"{self.energy_spin.value():.2f}"),
            ("Electron wavelength (Å)",
             f"{va.wavelength_A(self.energy_spin.value() * 1000.0):.5f}"),
            ("Screen distance (mm)", f"{self.dist_spin.value():.2f}"),
            ("Pixel pitch (mm)",   f"{self.pixel_spin.value():.4f}"),
            ("Lattice calibration K (Å·px)",
             "none" if self._lattice_K is None else f"{self._lattice_K:.4f}"),
            ("Strain reference a (Å)",
             "none" if self._a_ref is None else f"{self._a_ref:.4f}"),
            ("Last streak spacing (px)",
             "n/a" if not np.isfinite(self._last_spacing_px)
             else f"{self._last_spacing_px:.3f}"),
            ("Last lattice constant (Å)",
             "n/a" if not np.isfinite(self._last_a) else f"{self._last_a:.4f}"),
            ("Reference frame",    "active" if self.reference_frame is not None
                                   else "none"),
            ("ROI count",          str(len(self.roi_boxes))),
        ]
        for line in roi_lines:
            meta.append(("ROI", line))
        return meta

    def _save_csv(self, path=None):
        if not self.histories:
            QMessageBox.information(self, "Save CSV", "No ROI data to save.")
            return
        path = self._path_arg(path)
        if path is None:
            default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", self._dialog_path(default_name), "CSV (*.csv)")
            if not path:
                return
        self._remember_dir(path)

        idxs = sorted(self.histories)
        vals = {i: list(self.histories[i]) for i in idxs}
        tims = {i: list(self.times[i]) for i in idxs}
        max_len = max((len(v) for v in vals.values()), default=0)

        with open(path, 'w', newline='') as f:
            # Metadata block — lines starting with # are skipped by
            # pandas read_csv(comment='#') and numpy genfromtxt(comments='#')
            f.write("# VRHEED export\n")
            for key, val in self._collect_metadata():
                f.write(f"# {key}: {val}\n")
            f.write("# Time_s is ROI_0's timeline; each ROI also has its own\n"
                    "# t_ROI_n column because ROIs created later start later.\n")
            f.write("#\n")
            w = csv.writer(f)
            w.writerow(["Frame", "Time_s"]
                       + [f"ROI_{i}" for i in idxs]
                       + [f"t_ROI_{i}_s" for i in idxs])
            first = idxs[0]
            for r in range(max_len):
                t0 = f"{tims[first][r]:.4f}" if r < len(tims[first]) else ''
                row = [r, t0]
                row += [vals[i][r] if r < len(vals[i]) else '' for i in idxs]
                row += [f"{tims[i][r]:.4f}" if r < len(tims[i]) else '' for i in idxs]
                w.writerow(row)

        self.statusBar().showMessage(
            f"Saved {len(idxs)} ROI{'s' if len(idxs) != 1 else ''} "
            f"({max_len} samples max) → {os.path.basename(path)}", 6000)

    # -----------------------------------------------------------------------
    # Help
    # -----------------------------------------------------------------------

    def _show_help(self):
        QMessageBox.information(self, "Mouse & keyboard", (
            "Camera view\n"
            "  Left drag on empty space ....... draw a new ROI\n"
            "  Left drag inside an ROI ........ move it\n"
            "  Left drag the bottom-right ..... resize it\n"
            "  Left drag the thin line ........ move the profile slice\n"
            "  Right click inside an ROI ...... delete it\n"
            "  Scroll wheel ................... zoom about the cursor\n"
            "  Double click ................... reset zoom\n\n"
            "Keyboard\n"
            "  Space ........ pause / resume        Ctrl+0 ..... reset zoom\n"
            "  Del .......... delete active ROI     Ctrl+Shift+C  clear ROIs\n"
            "  Ctrl+O ....... open video            Ctrl+I ..... open image\n"
            "  Ctrl+S ....... snap image            Ctrl+R ..... record\n"
            "  Ctrl+E ....... export CSV            Ctrl+[ / ] . rotate\n"
            "  F5 ........... rescan for cameras\n\n"
            "Camera\n"
            "  The Source box at the top of the Camera tab lists every camera "
            "found.\n"
            "  Help ▸ Camera backends shows which drivers are installed and "
            "what to\n  install for the rest.\n\n"
            "Zoom, gamma, colormap and contrast are display-only. Every ROI "
            "value is measured on the raw frame."))

    def _show_log_location(self):
        QMessageBox.information(
            self, "Log file",
            f"Errors and session events are written to:\n\n{LOG_PATH}\n\n"
            f"The file rotates at 1 MB and keeps three older copies.")

    def _show_about(self):
        QMessageBox.about(self, "About VRHEED", (
            f"<b>VRHEED — Precision Analyzer</b> &nbsp; v{__version__}<br><br>"
            "Live RHEED acquisition and analysis.<br><br>"
            "Growth rate from the windowed FFT with sub-bin peak refinement, "
            "cross-checked by direct oscillation counting.<br>"
            "Per-ROI tracking of intensity, spot centroid, profile FWHM and "
            "streak spacing.<br>"
            "In-plane lattice constant and strain from streak separation.<br>"
            "Kinematic Ewald-sphere pattern overlay.<br>"
            "Recorded video and still images can be re-analysed offline.<br><br>"
            "Camera support: FLIR/Spinnaker (verified on hardware), USB "
            "cameras, network streams, screen capture and a simulated source, "
            "plus Basler, Allied Vision, GenICam and scientific-camera "
            "backends that are written but unverified — see "
            "Help ▸ Camera backends.<br><br>"
            f"<small>Python {platform.python_version()} · numpy {np.__version__} "
            f"· OpenCV {cv2.__version__}<br>"
            f"Camera drivers installed: {_installed_backends()}<br>"
            f"Log: {LOG_PATH}</small>"))

    # -----------------------------------------------------------------------
    # Persistent settings (QSettings)
    # -----------------------------------------------------------------------

    @staticmethod
    def _settings():
        """The settings store.

        VRHEED_SETTINGS_FILE redirects it to an INI file, so the smoke test can
        exercise save/restore without touching the operator's real settings.
        """
        path = os.environ.get('VRHEED_SETTINGS_FILE')
        if path:
            return QSettings(path, QSettings.IniFormat)
        return QSettings("UCF", "VRHEED")

    def _setting_widgets(self):
        """key -> widget for every value that persists between sessions.

        Camera gain/exposure/fps are deliberately NOT here: they belong to the
        camera and the operator sets them per growth from the pattern.
        """
        return {
            'lattice/energy_keV':    self.energy_spin,
            'lattice/distance_mm':   self.dist_spin,
            'lattice/pixel_mm':      self.pixel_spin,
            'lattice/known_a':       self.known_a_spin,
            'analysis/ml_A':         self.ml_spin,
            'analysis/fmin':         self.fmin_spin,
            'analysis/fmax':         self.fmax_spin,
            'analysis/history_min':  self.hist_spin,
            'display/colormap':      self.cmap_combo,
            'display/contrast_auto': self.contrast_auto_cb,
            'display/level_lo':      self.level_lo,
            'display/level_hi':      self.level_hi,
            'display/gamma':         self.gamma_spin,
            'display/avg_n':         self.avg_spin,
            'sim/on':                self.sim_on_cb,
            'sim/a':                 self.sim_a,
            'sim/b':                 self.sim_b,
            'sim/gamma':             self.sim_gamma,
            'sim/azimuth':           self.sim_azim,
            'sim/theta':             self.sim_theta,
            'sim/order':             self.sim_order,
            'sim/ppm':               self.sim_ppm,
        }

    @staticmethod
    def _widget_get(w):
        if isinstance(w, QComboBox):
            return w.currentIndex()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        return w.value()

    @staticmethod
    def _widget_set(w, v):
        """Apply a stored value with the type the widget expects.

        QSettings hands back strings from INI files and platform-typed values
        from the registry, so every branch coerces explicitly; a value that
        cannot be coerced raises and the caller keeps the default.
        """
        if isinstance(w, QComboBox):
            i = int(v)
            if not (0 <= i < w.count()):
                raise ValueError(f"combo index {i} out of range")
            w.setCurrentIndex(i)
        elif isinstance(w, QCheckBox):
            if isinstance(v, str):
                v = v.strip().lower() in ('true', '1', 'yes', 'on')
            w.setChecked(bool(v))
        elif isinstance(w, QSpinBox):
            w.setValue(int(float(v)))
        else:
            f = float(v)
            if not np.isfinite(f):
                raise ValueError("non-finite value")
            w.setValue(f)

    def _settings_snapshot(self):
        """Current values of everything _restore_settings can set."""
        snap = {k: self._widget_get(w) for k, w in self._setting_widgets().items()}
        snap['sim/origin'] = list(self._sim_origin)
        return snap

    def _apply_setting_values(self, values):
        widgets = self._setting_widgets()
        for key, val in values.items():
            try:
                if key == 'sim/origin':
                    if isinstance(val, str):
                        val = [float(p) for p in val.split(',')]
                    if len(val) != 2:
                        raise ValueError("origin needs two numbers")
                    self._set_sim_origin(*val)
                elif key in widgets:
                    self._widget_set(widgets[key], val)
            except Exception as e:
                logger.warning("Ignoring stored setting %s=%r: %s", key, val, e)
        self._mark_sim_dirty()

    def _restore_settings(self):
        s = self._settings()
        values = {}
        for key in list(self._setting_widgets()) + ['sim/origin']:
            if s.contains(key):
                val = s.value(key)
                if key == 'sim/origin' and isinstance(val, (list, tuple)):
                    val = [float(p) for p in val]
                values[key] = val
        self._apply_setting_values(values)

        for key, restore in (('window/geometry', self.restoreGeometry),
                             ('window/main_splitter', self.main_splitter.restoreState),
                             ('window/left_splitter', self.left_splitter.restoreState)):
            try:
                ba = s.value(key)
                if isinstance(ba, QByteArray) and not ba.isEmpty():
                    restore(ba)
            except Exception as e:
                logger.warning("Ignoring stored %s: %s", key, e)
        try:
            d = s.value('files/last_dir', "")
            if isinstance(d, str) and os.path.isdir(d):
                self._last_dir = d
        except Exception as e:
            logger.warning("Ignoring stored last_dir: %s", e)
        # Which camera to reconnect to, and the stream URLs that nothing can
        # discover.  Stored as keys and URLs rather than list positions: a
        # camera's position changes the moment a different one is unplugged.
        try:
            key = s.value('camera/last_key', "")
            if isinstance(key, str):
                self._last_cam_key = key
        except Exception as e:
            logger.warning("Ignoring stored camera key: %s", e)
        try:
            urls = s.value('camera/network_urls', [])
            if isinstance(urls, str):
                urls = [urls] if urls else []
            for url in (urls or []):
                vcam.register_network_camera(str(url))
        except Exception as e:
            logger.warning("Ignoring stored network cameras: %s", e)
        if values:
            logger.info("Restored %d settings", len(values))

    def _save_settings(self):
        s = self._settings()
        for key, val in self._settings_snapshot().items():
            if key == 'sim/origin':
                val = f"{val[0]:.6f},{val[1]:.6f}"   # portable across backends
            s.setValue(key, val)
        s.setValue('window/geometry', self.saveGeometry())
        s.setValue('window/main_splitter', self.main_splitter.saveState())
        s.setValue('window/left_splitter', self.left_splitter.saveState())
        s.setValue('files/last_dir', self._last_dir)
        s.setValue('camera/last_key',
                   self.cam_info.key if self.cam_info else self._last_cam_key)
        s.setValue('camera/network_urls', vcam.network_cameras())
        s.setValue('app/version', __version__)
        s.sync()

    def _reset_settings(self):
        """Forget the stored settings and put the controls back to built-in defaults."""
        s = self._settings()
        s.clear()
        s.sync()
        self._apply_setting_values(self._defaults)
        self._last_dir = ""
        self.main_splitter.setSizes([1140, 460])
        self.left_splitter.setSizes([750, 200])
        logger.info("Settings reset to defaults")
        self.statusBar().showMessage(
            "Settings reset to defaults (ROIs and camera settings untouched).", 6000)

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.play_timer.stop()
        try:
            self._save_settings()
        except Exception:
            logger.exception("Could not save settings")
        # An open writer left unreleased leaves an unplayable video file.
        if self.video_writer is not None:
            try: self.video_writer.release()
            except Exception: pass
            self.video_writer = None
        if self.video_cap is not None:
            try: self.video_cap.release()
            except Exception: pass
            self.video_cap = None
        self._disconnect_camera()
        # Process-wide driver handles (the Spinnaker System, the GenTL
        # producers) outlive individual cameras and have to be released last.
        try:
            vcam.shutdown()
        except Exception:
            logger.exception("Camera subsystem shutdown failed")
        logger.info("VRHEED closed")
        event.accept()


# ---------------------------------------------------------------------------


def _check() -> int:
    """Headless start-up check: build the window offscreen and exit.

    Proves a frozen build can actually import numpy/scipy/cv2/PyQt5 and
    construct the UI -- the failure mode packaging introduces -- without a
    camera, a display, or the operator's settings file.  The full behavioural
    smoke test is test_app_smoke.py, which needs the source tree.
    Used by .github/workflows/build.yml to gate every release binary.
    """
    import tempfile

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Discovery opens every local video device in turn.  On a CI runner that
    # is seconds of nothing, and this check is about whether the binary can
    # import and draw, not about hardware.
    os.environ["VRHEED_NO_CAMERA_SCAN"] = "1"
    tmp = tempfile.mkdtemp(prefix="vrheed_check_")
    os.environ.setdefault("VRHEED_SETTINGS_FILE", os.path.join(tmp, "settings.ini"))
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    win = VRHEED_App()
    win.show()
    app.processEvents()
    win.close()
    # Which camera drivers made it into this build -- the one thing about a
    # frozen binary you cannot tell by looking at it.
    drivers = [name for name, ok, _ in vcam.backend_status() if ok]
    print(f"PASS: VRHEED {__version__} starts ({platform.system()}); "
          f"camera backends: {', '.join(drivers) or 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    # Logging and the excepthook go in before anything Qt can raise from, so
    # the frozen exe (no console, stderr = None) still leaves a trace.
    _setup_logging()
    _install_excepthook()
    if "--check" in sys.argv[1:]:
        sys.exit(_check())
    # Must be set before the QApplication exists, or a 4K/150% Windows display
    # renders the controls at the wrong size.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VRHEED_App()
    win.show()
    sys.exit(app.exec_())
