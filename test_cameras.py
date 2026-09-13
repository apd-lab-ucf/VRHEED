"""Tests for the camera backends and the app's source chooser.

Run with ``QT_QPA_PLATFORM=offscreen python test_cameras.py`` -- no camera
needed.  The parts that need hardware are covered with a fake backend that is
registered for the duration of the test, so the wiring between the app and
``vrheed_cameras`` is exercised end to end: connect, stream, change gain and
binning, disconnect, and reconnect to something else.

Exit status is non-zero on any failure, so it can gate a build.
"""

import os
import sys
import tempfile
import platform
import threading
import time
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_TMP = tempfile.mkdtemp(prefix="vrheed_cams_")
os.environ['VRHEED_SETTINGS_FILE'] = os.path.join(_TMP, "settings.ini")
os.environ['VRHEED_LOG_FILE'] = os.path.join(_TMP, "vrheed.log")

try:
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QMessageBox
except ImportError as e:
    print(f"SKIP  {e.name or e} not available")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vrheed_cameras as vc

# A modal dialog in a headless test waits forever for a click, and the error
# paths under test are exactly the ones that raise dialogs.  Record them
# instead, so a test can also assert that the operator was actually told.
_dialogs = []


def _silence_dialogs():
    for name in ("critical", "warning", "information", "about"):
        setattr(QMessageBox, name,
                staticmethod(lambda *a, _n=name, **k: _dialogs.append(
                    (_n, a[2] if len(a) > 2 else ""))))


_fails = []


def check(name, cond, detail=""):
    # detail is whatever was handy at the call site -- a tuple, a shape, a
    # list of keys -- so coerce rather than make every caller remember str().
    detail = "" if detail == "" or detail is None else str(detail)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


def pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# A fake camera, so the hardware paths can be tested without hardware
# ---------------------------------------------------------------------------

class FakeBackend(vc.CameraBackend):
    """16-bit camera with hardware binning that records what it was told."""

    backend_id = "fake"
    backend_name = "Fake test camera"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_hardware_binning = True
    gain_range = (0.0, 24.0)
    exposure_range_ms = (0.05, 5000.0)

    instances = []

    @classmethod
    def available(cls):
        return True

    @classmethod
    def enumerate(cls):
        return [vc.CameraInfo(cls, "A", "Fake camera A", "test"),
                vc.CameraInfo(cls, "B", "Fake camera B", "test")]

    def _open(self):
        self.calls = []
        self.closed = False
        self._size = (480, 640)
        self._meta["Camera model"] = f"Fake {self.device_id}"
        FakeBackend.instances.append(self)

    def _close(self):
        self.closed = True

    def _read(self, timeout_ms):
        h, w = self._size
        return np.full((h, w), 1000, np.uint16)

    def _set_gain(self, v):          self.calls.append(("gain", v))
    def _set_exposure_ms(self, v):   self.calls.append(("exp", v))
    def _set_frame_rate(self, v):    self.calls.append(("fps", v))

    def _set_hw_binning(self, n):
        n = min(n, 2)                      # this "camera" only does 1x and 2x
        self.calls.append(("bin", n))
        self._size = (480 // n, 640 // n)
        return n


class BrokenBackend(vc.CameraBackend):
    """Refuses to open, the way a camera held by another program does."""

    backend_id = "broken"
    backend_name = "Broken test camera"

    @classmethod
    def available(cls):
        return True

    @classmethod
    def enumerate(cls):
        return [vc.CameraInfo(cls, "X", "Broken camera", "test")]

    def _open(self):
        raise vc.CameraError("camera is in use by another program")


# ---------------------------------------------------------------------------
# Module-level behaviour
# ---------------------------------------------------------------------------

def test_registry():
    ids = [b.backend_id for b in vc.BACKENDS]
    check("every backend has a unique id", len(ids) == len(set(ids)), str(ids))
    check("synthetic source is always available", vc.SyntheticBackend.available())
    check("backend_status covers every backend",
          len(vc.backend_status()) == len(vc.BACKENDS))
    for name, ok, reason in vc.backend_status():
        if not ok:
            check(f"unavailable backend explains itself: {name}", bool(reason), reason)
    check("enumerate never raises", isinstance(vc.enumerate_cameras(), list))
    check("a backend with no driver contributes nothing",
          all(c.backend_id != 'spinnaker' for c in vc.enumerate_cameras())
          or vc.SpinnakerBackend.available())


def test_console_output_is_ascii():
    """The diagnostic must survive a cp1252 Windows console.

    Its whole job is to be readable when nothing else works, so it must not be
    the thing that dies on an encoding.  A Windows console is often cp1252 and
    a redirected stream uses the locale encoding; a stray em dash there is a
    UnicodeEncodeError, not a mojibake.
    """
    import io
    source = io.open(vc.__file__, encoding="utf-8").read()
    offenders = sorted({c for c in source if ord(c) > 127})
    check("vrheed_cameras.py is pure ASCII",
          not offenders,
          "".join(f"U+{ord(c):04X} " for c in offenders))

    # Everything the diagnostic prints, through the narrowest encoding there is.
    printable = [name + reason for name, _ok, reason in vc.backend_status()]
    printable += [i.label + i.detail for i in vc.enumerate_cameras()]
    printable += vc.spinnaker_report()
    bad = [t for t in printable
           if any(ord(c) > 127 for c in t)]
    check("every string the diagnostic prints is ASCII", not bad, str(bad[:2]))


def test_driver_absent_vs_broken():
    """"Not installed" and "installed but will not load" need different fixes.

    A PySpin whose Spinnaker runtime is missing raises "DLL load failed";
    reporting that as "PySpin not installed" sends someone to reinstall what
    they already have, which is the single most expensive wrong message this
    app can print.
    """
    check("a dotted module name is not misread as broken",
          vc._missing("No module named 'pypylon'", "pypylon.pylon"))
    check("an unrelated ancestor is not matched",
          not vc._missing("No module named 'pylon'", "pypylon.pylon"))
    check("a load failure is not read as absence",
          not vc._missing("DLL load failed while importing _PySpin", "PySpin"))

    # A module that exists but blows up on import, like a mismatched wheel.
    import importlib
    broken = types.ModuleType("vrheed_fake_driver")
    real_import = importlib.import_module

    def fake_import(name, *a, **k):
        if name == "PySpin":
            raise ImportError("DLL load failed while importing _PySpin: "
                              "The specified module could not be found.")
        return real_import(name, *a, **k)

    importlib.import_module = fake_import
    try:
        ok = vc.SpinnakerBackend.available()
        reason = vc.SpinnakerBackend.unavailable_reason()
    finally:
        importlib.import_module = real_import
        vc.SpinnakerBackend.available()      # reset the cached error
    check("a driver that will not load is still unavailable", ok is False)
    check("...and is reported as loading-failed, not missing",
          "will not load" in reason and "DLL load failed" in reason, reason)
    check("...and says what usually causes it",
          "Spinnaker SDK runtime" in reason or "wheel" in reason, reason)
    del broken


def test_numpy_abi_detection():
    """The numpy 1-vs-2 ABI break is detected, not assumed.

    Which vendor wheels were built against numpy 1 is not knowable in advance,
    so the code reads the loader's own complaint instead of the docs asserting
    it.  That only helps if it fires on the real message and stays quiet
    otherwise.
    """
    import numpy as _np
    on_numpy2 = int(_np.__version__.split(".")[0]) >= 2

    real = "ImportError: numpy.core.multiarray failed to import"
    note = vc._numpy_abi_problem(real)
    if on_numpy2:
        check("the ABI break is recognised under numpy 2",
              "numpy 1-vs-2" in note and "numpy<2" in note, note[:80])
        check("...and names the interpreter to fix",
              sys.executable in note, note[:80])
    else:
        check("no numpy-2 advice is given on numpy 1", note == "", note)

    check("an unrelated DLL failure is not blamed on numpy",
          vc._numpy_abi_problem("DLL load failed while importing _PySpin") == "")
    check("a plain missing module is not blamed on numpy",
          vc._numpy_abi_problem("No module named 'PySpin'") == "")


def test_sdk_version_lookup():
    """Reading the installed SDK version must be safe off Windows."""
    version = vc._spinnaker_sdk_installed_version()
    check("SDK version lookup returns a string and never raises",
          isinstance(version, str), repr(version))
    if platform.system() != "Windows":
        check("...and is empty off Windows", version == "", repr(version))


def test_spinnaker_report_without_pyspin():
    """The --flir diagnosis must work even with no PySpin at all."""
    lines = vc.spinnaker_report()
    joined = "\n".join(lines)
    check("the FLIR report always returns something",
          isinstance(lines, list) and lines, lines)
    # Checked by content, not by position: the report leads with whether the
    # Spinnaker SDK is on disk, because the SDK and the binding are separate
    # installs and "is the SDK even here" comes first.  An earlier version of
    # this test asserted line 0, and adding that section broke it.
    check("it reports the SDK-on-disk search",
          "Spinnaker SDK" in joined, joined[:120])
    check("it reports the PySpin import outcome",
          "PySpin import FAILED" in joined or "PySpin imported OK" in joined,
          joined[:120])
    check("every line is a string, so printing cannot fail",
          all(isinstance(line, str) for line in lines))


def test_software_binning():
    frame = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    out = vc.CameraBackend._software_bin(frame, 2)
    check("software binning halves both axes", out.shape == (2, 3), str(out.shape))
    check("software binning is a mean, not a decimation",
          out[0, 0] == int(np.mean(frame[:2, :2])), f"got {out[0, 0]}")
    check("software binning keeps the dtype", out.dtype == frame.dtype)
    odd = np.ones((5, 7), np.uint16)
    check("a size that is not a multiple is cropped, not padded",
          vc.CameraBackend._software_bin(odd, 2).shape == (2, 3))
    check("binning by 1 is a no-op",
          vc.CameraBackend._software_bin(frame, 1) is frame)


def test_mono_conversion():
    colour = np.zeros((8, 8, 3), np.uint8)
    colour[..., 1] = 255
    check("BGR frames are reduced to 2-D",
          vc.CameraBackend._to_mono(colour).ndim == 2)
    bgra = np.zeros((8, 8, 4), np.uint8)
    check("BGRA frames (screen capture) are reduced to 2-D",
          vc.CameraBackend._to_mono(bgra).ndim == 2)
    mono16 = np.zeros((8, 8), np.uint16)
    check("mono frames keep their bit depth",
          vc.CameraBackend._to_mono(mono16).dtype == np.uint16)


def test_synthetic():
    info = vc.SyntheticBackend.enumerate()[0]
    cam = vc.open_camera(info)
    cam.start()
    frame = cam.get_frame(500)
    check("synthetic camera yields a 2-D mono frame",
          frame is not None and frame.ndim == 2 and frame.dtype == np.uint8,
          None if frame is None else f"{frame.shape} {frame.dtype}")

    # The whole point of the synthetic source is a known answer, so measure it.
    cam.set_frame_rate(60)
    h, w = frame.shape
    vals, t0 = [], time.monotonic()
    while time.monotonic() - t0 < 12.0:
        f = cam.get_frame(500)
        if f is not None:
            vals.append(float(f[int(h * 0.50):int(h * 0.60),
                               int(w * 0.47):int(w * 0.53)].mean()))
    dur = time.monotonic() - t0
    v = np.asarray(vals) - np.mean(vals)
    mag = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    freq = np.fft.rfftfreq(len(v), dur / len(v))
    peak = float(freq[int(np.argmax(mag[1:])) + 1])
    check("simulated specular spot oscillates at the documented 0.25 Hz",
          abs(peak - vc.SyntheticBackend.OSC_HZ) < 0.02, f"got {peak:.3f} Hz")
    check("simulated camera keeps up with at least 25 fps",
          len(vals) / dur > 25, f"{len(vals) / dur:.0f} fps")

    # Software binning through the public path.
    cam.set_binning(2)
    binned = cam.get_frame(500)
    check("binning 2x through get_frame halves the frame",
          binned.shape == (frame.shape[0] // 2, frame.shape[1] // 2),
          str(binned.shape))
    check("metadata marks software binning as such",
          "software" in cam.metadata()["Binning"], cam.metadata()["Binning"])
    cam.close()
    check("closing an unopened camera twice is harmless",
          (cam.close() or True) and not cam.is_open)


def test_hardware_binning_fallback():
    """A camera that maxes out at 2x must still deliver 4x, in software."""
    info = FakeBackend.enumerate()[0]
    cam = vc.open_camera(info)
    cam.start()
    full = cam.get_frame(500)
    cam.set_binning(4)
    out = cam.get_frame(500)
    check("4x on a 2x-capable camera still comes out 4x",
          out.shape == (full.shape[0] // 4, full.shape[1] // 4), str(out.shape))
    check("the hardware was asked for what it could do",
          ("bin", 2) in cam.calls, str(cam.calls))
    check("mixed hardware/software binning is labelled",
          "software" in cam.metadata()["Binning"], cam.metadata()["Binning"])
    cam.close()


def test_control_errors_are_contained():
    """A control the camera refuses must not propagate into the UI thread."""
    class Grumpy(FakeBackend):
        backend_id = "grumpy"
        def _set_gain(self, v):
            raise RuntimeError("no such node")

    cam = Grumpy(vc.CameraInfo(Grumpy, "A", "Grumpy", ""))
    cam.open()
    try:
        cam.set_gain(3.0)
        ok = True
    except Exception:
        ok = False
    check("a refused control setter does not raise", ok)
    cam.close()


def test_network_registry():
    url = "rtsp://user:secret@10.0.0.5:554/stream"
    vc.register_network_camera(url)
    vc.register_network_camera(url)              # twice, on purpose
    check("a stream URL is stored once", vc.network_cameras().count(url) == 1)
    info = [c for c in vc.NetworkCameraBackend.enumerate() if c.device_id == url]
    check("the stream appears as a camera", len(info) == 1)
    check("the password is not shown in the label",
          "secret" not in info[0].label, info[0].label)
    vc.forget_network_camera(url)
    check("a stream URL can be forgotten", url not in vc.network_cameras())


def test_screen_ids():
    check("plain monitor id parses", vc.parse_screen_id("2") == (2, None))
    check("monitor + region parses",
          vc.parse_screen_id("1@10,20,300,400") == (1, (10, 20, 300, 400)))
    check("a malformed region degrades to the whole monitor",
          vc.parse_screen_id("1@nonsense") == (1, None))
    check("round-trip through screen_source_key",
          vc.parse_screen_id(vc.screen_source_key(3, (1, 2, 30, 40)))
          == (3, (1, 2, 30, 40)))


# ---------------------------------------------------------------------------
# A stand-in for PySpin, so the FLIR path is tested without a FLIR camera
# ---------------------------------------------------------------------------
#
# VRHEED's original and still most important camera is the Blackfly S on the
# MBE, and 2.2 rewrote how it is driven.  This stub models it closely enough
# to check the whole sequence against what 2.1 did: enumeration by serial,
# every auto control turned off with the SDK's own integer constants and in
# the right order, gain in dB and exposure in microseconds, incomplete frames
# dropped, every image released, binning reconfiguring the sensor window, and
# shutdown unwinding EndAcquisition -> DeInit -> ReleaseInstance.

class _SpinNode:
    def __init__(self, name, value=0.0, lo=None, hi=None):
        self.name, self.value, self.lo, self.hi = name, value, lo, hi

    def SetValue(self, v):
        if self.hi is not None and not (self.lo <= v <= self.hi):
            raise RuntimeError(f"{self.name}: {v} out of range")
        _SPIN_LOG.append(("set", self.name, v))
        self.value = v

    def GetValue(self):  return self.value
    def GetMin(self):    return self.lo
    def GetMax(self):    return self.hi

    def FromString(self, text):
        _SPIN_LOG.append(("fromstring", self.name, text))


class _SpinImage:
    def __init__(self, arr, incomplete=False):
        self._arr, self._inc = arr, incomplete

    def IsIncomplete(self):  return self._inc
    def GetNDArray(self):
        if self._inc:
            raise RuntimeError("incomplete image has no data")
        return self._arr

    def Release(self):
        _SPIN_LOG.append(("release", "image", None))


class _SpinCamera:
    SERIAL = "21055099"
    MODEL = "Blackfly S BFS-U3-51S5M"

    def __init__(self):
        tl = type("TL", (), {})()
        tl.DeviceModelName = _SpinNode("TL.Model", self.MODEL)
        tl.DeviceSerialNumber = _SpinNode("TL.Serial", self.SERIAL)
        self.TLDevice = tl
        self._nodes = {
            # Enumerations, which must be set with PySpin's integer constants.
            "GainAuto": _SpinNode("GainAuto"),
            "ExposureAuto": _SpinNode("ExposureAuto"),
            "BlackLevelAuto": _SpinNode("BlackLevelAuto"),
            "BalanceWhiteAuto": _SpinNode("BalanceWhiteAuto"),
            "AcquisitionMode": _SpinNode("AcquisitionMode"),
            # Booleans.
            "BlackLevelClampingEnable": _SpinNode("BlackLevelClampingEnable"),
            "GammaEnable": _SpinNode("GammaEnable"),
            "AcquisitionFrameRateEnable": _SpinNode("AcquisitionFrameRateEnable"),
            # Ranged values, with the Blackfly's real limits.
            "Gain": _SpinNode("Gain", 0.0, 0.0, 47.99),
            "ExposureTime": _SpinNode("ExposureTime", 10000.0, 6.0, 30000000.0),
            "AcquisitionFrameRate": _SpinNode("AcquisitionFrameRate", 30.0, 1.0, 120.0),
            "BinningHorizontal": _SpinNode("BinningHorizontal", 1, 1, 2),
            "BinningVertical": _SpinNode("BinningVertical", 1, 1, 2),
            "OffsetX": _SpinNode("OffsetX", 0, 0, 2448),
            "OffsetY": _SpinNode("OffsetY", 0, 0, 2048),
            "Width": _SpinNode("Width", 2448, 8, 2448),
            "Height": _SpinNode("Height", 2048, 8, 2048),
            "DeviceVendorName": _SpinNode("DeviceVendorName", "FLIR"),
            "DeviceModelName": _SpinNode("DeviceModelName", self.MODEL),
            "DeviceSerialNumber": _SpinNode("DeviceSerialNumber", self.SERIAL),
            "PixelFormat": _SpinNode("PixelFormat", "Mono16"),
        }
        self.serial = self.SERIAL
        self.acquiring = False
        self._served = 0

    # SharpnessEnable and anything else absent raises, like a model that
    # simply does not have the node.
    def __getattr__(self, name):
        nodes = self.__dict__.get("_nodes", {})
        if name in nodes:
            return nodes[name]
        raise AttributeError(name)

    def Init(self):             _SPIN_LOG.append(("init", self.serial, None))
    def DeInit(self):           _SPIN_LOG.append(("deinit", self.serial, None))

    def BeginAcquisition(self):
        self.acquiring = True
        _SPIN_LOG.append(("begin", self.serial, None))

    def EndAcquisition(self):
        self.acquiring = False
        _SPIN_LOG.append(("end", self.serial, None))

    def GetNextImage(self, timeout):
        if not self.acquiring:
            raise RuntimeError("not acquiring")
        self._served += 1
        n = int(self._nodes["BinningHorizontal"].value)
        return _SpinImage(np.full((2048 // n, 2448 // n), 1234, np.uint16),
                          incomplete=(self._served % 5 == 0))


class _SpinCameraList:
    def __init__(self, cams):  self._cams = cams
    def GetSize(self):         return len(self._cams)
    def __getitem__(self, i):  return self._cams[i]
    def Clear(self):           _SPIN_LOG.append(("list_clear", None, None))

    def GetBySerial(self, serial):
        for c in self._cams:
            if c.serial == serial:
                return c
        return None


_SPIN_LOG = []
_SPIN_CAMS = [_SpinCamera()]
_SPIN_RELEASES = []


def _install_pyspin_stub():
    """Put the stub in sys.modules and return an undo callable."""
    import types
    mod = types.ModuleType("PySpin")
    # The integer constants the 2.1 code used, and 2.2 must keep using.
    mod.GainAuto_Off = 0
    mod.ExposureAuto_Off = 0
    mod.BlackLevelAuto_Off = 0
    mod.BalanceWhiteAuto_Off = 0
    mod.AcquisitionMode_Continuous = 2

    class _Sys:
        def GetCameras(self):     return _SpinCameraList(_SPIN_CAMS)
        def ReleaseInstance(self):
            _SPIN_RELEASES.append(1)
            _SPIN_LOG.append(("release_system", None, None))

    mod.System = type("System", (), {"GetInstance": staticmethod(_Sys)})
    previous = sys.modules.get("PySpin")

    def undo():
        if previous is None:
            sys.modules.pop("PySpin", None)
        else:
            sys.modules["PySpin"] = previous

    sys.modules["PySpin"] = mod
    return undo


def test_spinnaker_against_stub(app):
    import main
    undo = _install_pyspin_stub()
    saved = list(vc.BACKENDS)
    vc.BACKENDS[:] = [vc.SpinnakerBackend, vc.SyntheticBackend]
    _SPIN_LOG.clear()
    _SPIN_RELEASES.clear()
    try:
        check("Spinnaker reports itself available", vc.SpinnakerBackend.available())
        cams = vc.SpinnakerBackend.enumerate()
        check("the FLIR camera is found", len(cams) == 1, [c.key for c in cams])
        check("it is keyed by serial, not by index",
              cams[0].key == f"spinnaker:{_SpinCamera.SERIAL}", cams[0].key)
        check("enumeration releases the CameraList",
              ("list_clear", None, None) in _SPIN_LOG)

        # The previous test left "reconnect to synthetic:0" in the sandboxed
        # settings file, and honouring it is correct behaviour -- but what is
        # under test here is the no-preference path, so clear it first.
        store = main.VRHEED_App._settings()
        store.remove('camera/last_key')
        store.sync()

        _SPIN_LOG.clear()
        win = main.VRHEED_App()
        try:
            check("a FLIR camera is auto-connected in preference to the simulator",
                  win.cam_info is not None
                  and win.cam_info.key == f"spinnaker:{_SpinCamera.SERIAL}",
                  str(win.cam_info))

            sets = {name: v for kind, name, v in _SPIN_LOG if kind == "set"}
            order = [name for kind, name, v in _SPIN_LOG if kind == "set"]

            # The reason any of this matters: each of these, left on, puts a
            # discrete step in the intensity trace that reads as a transient.
            for feat in ("GainAuto", "ExposureAuto", "BlackLevelAuto",
                         "BalanceWhiteAuto"):
                check(f"{feat} off, set with the SDK's own constant",
                      sets.get(feat) == 0, sets.get(feat))
            for feat in ("BlackLevelClampingEnable", "GammaEnable"):
                check(f"{feat} disabled", sets.get(feat) is False, sets.get(feat))
            check("no enumeration needed the FromString fallback",
                  not [x for x in _SPIN_LOG if x[0] == "fromstring"],
                  [x for x in _SPIN_LOG if x[0] == "fromstring"])
            check("a node this model does not have is skipped quietly",
                  "SharpnessEnable" not in sets)
            check("the autos go off before gain and exposure are set",
                  order.index("GainAuto") < order.index("Gain")
                  and order.index("ExposureAuto") < order.index("ExposureTime"),
                  str(order[:8]))
            check("frame rate is locked, so FFT frequencies mean something",
                  sets.get("AcquisitionFrameRateEnable") is True
                  and sets.get("AcquisitionFrameRate") == 30.0, str(sets.get(
                      "AcquisitionFrameRate")))
            check("gain is sent in dB", sets.get("Gain") == 15.0, sets.get("Gain"))
            check("exposure is sent in microseconds",
                  sets.get("ExposureTime") == 100000.0, sets.get("ExposureTime"))

            # Ranges must round inward: the Blackfly's 47.99 dB maximum shown
            # in a one-decimal box would come back as 48.0, which it rejects.
            check("every gain the operator can dial in is one the camera takes",
                  0.0 <= win.gain_spin.minimum()
                  and win.gain_spin.maximum() <= 47.99,
                  (win.gain_spin.minimum(), win.gain_spin.maximum()))
            check("exposure range is clamped to whole milliseconds",
                  win.exp_spin.minimum() >= 1.0 and win.exp_spin.maximum() <= 30000.0,
                  (win.exp_spin.minimum(), win.exp_spin.maximum()))

            pump(app, 0.4)
            frame = win.last_raw_frame
            # The app starts at DEFAULT_BINNING.  This Blackfly caps at 2x in
            # hardware, so 4x is 2x on the sensor and 2x in software -- the
            # point being that the operator asked for 4x and got 4x.
            d = main.DEFAULT_BINNING
            check(f"frames arrive binned {d}x{d}, the startup default",
                  frame is not None
                  and frame.shape == (2048 // d, 2448 // d)
                  and frame.dtype == np.uint16,
                  None if frame is None else (frame.shape, frame.dtype))
            check("the default binning reached the camera",
                  win.cam.binning == d, win.cam.binning)
            check("the Lattice tab warns that the pitch must include binning",
                  "Binning is" in win.bin_note.text()
                  and str(d) in win.bin_note.text(),
                  win.bin_note.text())
            check("16-bit full scale is recognised", win._raw_max == 65535.0,
                  win._raw_max)
            check("incomplete frames are dropped, never measured",
                  frame is not None and int(frame.max()) == 1234)
            check("every grabbed image is released (no buffer-pool leak)",
                  len([x for x in _SPIN_LOG if x[0] == "release"]) > 5)

            _SPIN_LOG.clear()
            win.gain_spin.setValue(8.0)
            win.exp_spin.setValue(50.0)
            win.fps_spin.setValue(60.0)
            sets = {name: v for kind, name, v in _SPIN_LOG if kind == "set"}
            check("the gain box reaches the camera in dB", sets.get("Gain") == 8.0,
                  sets.get("Gain"))
            check("the exposure box reaches the camera in microseconds",
                  sets.get("ExposureTime") == 50000.0, sets.get("ExposureTime"))
            check("the FPS box reaches the camera",
                  sets.get("AcquisitionFrameRate") == 60.0,
                  sets.get("AcquisitionFrameRate"))

            _SPIN_LOG.clear()
            win._apply_binning(2)
            pump(app, 0.6)
            sets = {name: v for kind, name, v in _SPIN_LOG if kind == "set"}
            check("2x binning is done by the sensor",
                  sets.get("BinningHorizontal") == 2
                  and sets.get("BinningVertical") == 2, str(sets))
            check("offsets reset and the window re-maximised, as in 2.1",
                  sets.get("OffsetX") == 0 and sets.get("Width") == 2448, str(sets))
            check("acquisition stops and restarts around a binning change",
                  ("end", _SpinCamera.SERIAL, None) in _SPIN_LOG
                  and ("begin", _SpinCamera.SERIAL, None) in _SPIN_LOG)
            check("hardware binning is not mislabelled as software",
                  not win.cam.binning_is_software)
            check("the 2x frame is half size",
                  win.last_raw_frame.shape == (1024, 1224),
                  win.last_raw_frame.shape)

            # This model caps at 2x. 2.1 greyed the 4x button out; 2.2 makes up
            # the difference in software and says so.
            win._apply_binning(4)
            pump(app, 0.6)
            check("4x on a 2x-capable Blackfly still yields 4x",
                  win.last_raw_frame.shape == (512, 612), win.last_raw_frame.shape)
            check("the mixed hardware+software case is labelled software",
                  win.cam.binning_is_software
                  and "software" in win.cam.metadata()["Binning"],
                  win.cam.metadata()["Binning"])
            win._apply_binning(1)
            pump(app, 0.6)
            check("only ever one capture thread on the camera handle",
                  sum(1 for t in threading.enumerate()
                      if getattr(getattr(t, "_target", None), "__name__", "")
                      == "_capture_loop" and t.is_alive()) <= 1)

            # A lattice calibration must survive a binning change.  K = a*dx
            # and dx scales with binning, so K has to scale the other way or
            # the lattice constant silently changes by that factor.
            import vrheed_analysis as va
            win._apply_binning(1)
            pump(app, 0.6)
            a_known, dx_at_1x = 3.905, 40.0
            win._lattice_K = va.calibration_constant(a_known, dx_at_1x)
            win._apply_binning(2)
            pump(app, 0.6)
            # The same feature now spans half as many pixels.
            a_after = va.lattice_from_calibration(dx_at_1x / 2, win._lattice_K)
            check("a lattice calibration survives a binning change",
                  abs(a_after - a_known) < 1e-9,
                  f"{a_after:.6f} A vs {a_known} A")
            win._lattice_K = None
            win._apply_binning(1)
            pump(app, 0.6)

            meta = dict(x for x in win._collect_metadata() if isinstance(x, tuple))
            check("the CSV header carries model, serial and vendor",
                  _SpinCamera.MODEL in meta.get("Camera model", "")
                  and meta.get("Camera serial") == _SpinCamera.SERIAL
                  and meta.get("Camera vendor") == "FLIR", str(meta.get("Camera model")))

            hint = vc._spinnaker_hint(RuntimeError("Camera on wrong subnet (-1015)"))
            check("the GigE wrong-subnet fix is still spelled out",
                  "Auto Force IP" in hint and "169.254" in hint)

            _SPIN_LOG.clear()
            win.close()
            kinds = [x[0] for x in _SPIN_LOG]
            check("shutdown ends acquisition, then DeInits, then releases the system",
                  kinds.index("end") < kinds.index("deinit")
                  < kinds.index("release_system"), str(kinds))
            check("the Spinnaker System is released exactly once",
                  len(_SPIN_RELEASES) == 1, len(_SPIN_RELEASES))
        finally:
            win.close()
    finally:
        vc.BACKENDS[:] = saved
        vc.shutdown()          # drop the stub System from the module global
        undo()


# ---------------------------------------------------------------------------
# The app's source chooser

# ---------------------------------------------------------------------------

def test_app_camera_wiring(app):
    import main

    # Register the fakes for the duration of this test only.
    saved = list(vc.BACKENDS)
    vc.BACKENDS[:] = [FakeBackend, BrokenBackend, vc.SyntheticBackend]
    try:
        win = main.VRHEED_App()
        try:
            keys = [win.cam_combo.itemData(i) for i in range(win.cam_combo.count())]
            check("every discovered camera is offered",
                  {"fake:A", "fake:B", "synthetic:0"} <= set(keys), str(keys))
            check("a real camera is auto-connected in preference to the simulator",
                  win.cam_info is not None and win.cam_info.key == "fake:A",
                  str(win.cam_info))
            check("the chooser shows what is connected",
                  win.cam_combo.currentData() == "fake:A")
            check("connecting starts the capture thread",
                  win._cap_thread is not None and win._cap_thread.is_alive())

            pump(app, 0.4)
            check("frames from the backend reach the display",
                  win.last_raw_frame is not None and win.last_raw_frame.ndim == 2)
            check("16-bit frames are recognised as 16-bit",
                  win._raw_max == 65535.0, str(win._raw_max))

            cam = win.cam
            check("panel values are pushed to a newly connected camera",
                  any(c[0] == "gain" for c in cam.calls)
                  and any(c[0] == "exp" for c in cam.calls), str(cam.calls))
            check("control ranges follow the camera",
                  win.gain_spin.maximum() == 24.0, str(win.gain_spin.maximum()))

            win.gain_spin.setValue(7.5)
            check("moving the gain spin box reaches the camera",
                  ("gain", 7.5) in cam.calls)

            # Switch cameras the way the operator does.
            idx = win.cam_combo.findData("fake:B")
            win.cam_combo.setCurrentIndex(idx)
            win._on_camera_selected(idx)
            check("selecting another camera connects to it",
                  win.cam_info.key == "fake:B", str(win.cam_info))
            check("the previous camera was released", cam.closed)

            # A camera that refuses to open must leave the app usable.
            broken = vc.find_camera("broken:X", win._cameras)
            if broken is None:
                broken = BrokenBackend.enumerate()[0]
                win._cameras.append(broken)
            _dialogs.clear()
            ok = win._connect_camera(broken)
            check("a camera that will not open reports failure", ok is False)
            check("...and says why, rather than failing silently",
                  any("in use by another program" in str(d[1]) for d in _dialogs),
                  str(_dialogs))
            check("a failed connection leaves no camera attached", win.cam is None)
            check("a failed connection greys the camera controls out",
                  not win.gain_spin.isEnabled())
            pump(app, 0.2)

            # ...and the operator can still pick the simulator and carry on.
            sim = vc.find_camera("synthetic:0", win._cameras)
            check("the simulated source connects after a failure",
                  win._connect_camera(sim) is True)
            pump(app, 0.3)
            check("capability flags drive the controls: no hardware binning "
                  "on the simulator still offers 4x",
                  all(b.isEnabled() for b in win._bin_group.buttons()))

            meta = dict(x for x in win._collect_metadata() if isinstance(x, tuple))
            check("exported metadata names the backend",
                  meta.get("Camera backend") == "Simulated RHEED pattern",
                  str(meta.get("Camera backend")))
            check("exported metadata still carries the frame size",
                  "Frame width (px)" in meta)

            win._disconnect_camera()
            check("disconnect releases the camera", win.cam is None)
            check("disconnect stops the capture thread",
                  win._cap_thread is None or not win._cap_thread.is_alive())
            check("the connect button offers to reconnect",
                  win.btn_cam_connect.text() == "Connect")

            # The reconnect key is what gets written to the settings file.
            check("the last camera is remembered for next session",
                  win._last_cam_key == "synthetic:0", win._last_cam_key)
        finally:
            win.close()
    finally:
        vc.BACKENDS[:] = saved


def main_():
    app = QApplication.instance() or QApplication(sys.argv)
    _silence_dialogs()
    test_registry()
    test_console_output_is_ascii()
    test_driver_absent_vs_broken()
    test_numpy_abi_detection()
    test_sdk_version_lookup()
    test_spinnaker_report_without_pyspin()
    test_software_binning()
    test_mono_conversion()
    test_synthetic()
    test_hardware_binning_fallback()
    test_control_errors_are_contained()
    test_network_registry()
    test_screen_ids()
    test_app_camera_wiring(app)
    test_spinnaker_against_stub(app)

    print()
    if _fails:
        print(f"{len(_fails)} check(s) failed:")
        for name in _fails:
            print(f"  - {name}")
        return 1
    print(f"all checks passed  ({_TMP})")
    return 0


if __name__ == "__main__":
    sys.exit(main_())
