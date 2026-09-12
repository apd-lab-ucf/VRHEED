"""Camera backends for VRHEED.

VRHEED started as a FLIR/Spinnaker-only application.  A RHEED screen, though,
gets imaged by whatever the lab already owns, and that is rarely the same
camera twice: machine-vision cameras from half a dozen vendors, scientific
CCD/sCMOS cameras on the high-end systems, an analogue CCD on a USB frame
grabber on the old ones, and — more often than anyone admits — a webcam
pointed at the phosphor screen.  This module hides all of that behind one
small interface so the rest of the app never has to know.

Every backend returns a **2-D mono** ``numpy`` array (uint8 or uint16).  The
analysis pipeline assumes that; colour sensors are converted here, once, in
:meth:`CameraBackend.get_frame`.

Adding a backend
----------------
Subclass :class:`CameraBackend`, implement ``_open`` / ``_start`` / ``_read``
(plus whatever controls the hardware supports), set the ``supports_*`` class
flags, and add the class to :data:`BACKENDS`.  Discovery, software binning,
mono conversion and the GUI wiring then come for free.

Dependencies
------------
Every third-party driver is imported lazily, inside the backend that needs it.
Nothing here is a hard requirement: a machine with none of them installed
still gets the synthetic camera, the OpenCV backends (USB / network), and
screen capture if ``mss`` is present.  That matters for a public release —
users should not have to install a 1 GB vendor SDK to try the analysis on a
recorded video.
"""

from __future__ import annotations

import logging
import os
import platform
import importlib
import re
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("vrheed.cameras")

__all__ = [
    "CameraError", "CameraInfo", "CameraBackend", "BACKENDS",
    "enumerate_cameras", "open_camera", "find_camera", "backend_status",
    "shutdown", "register_network_camera", "forget_network_camera",
    "network_cameras", "parse_screen_id", "screen_source_key",
]


class CameraError(Exception):
    """Anything that goes wrong talking to a camera."""


def _probe_import(module_name):
    """Import ``module_name``, returning ``(module, error_text)``.

    The distinction that matters: "not installed" and "installed but will not
    load" need completely different fixes, and a vendor binding hits the
    second far more often than anyone expects.  PySpin whose Spinnaker runtime
    is missing or a different version than the wheel raises
    "DLL load failed while importing _PySpin", and telling that user to
    install PySpin -- which they plainly did -- wastes an afternoon.
    """
    try:
        return importlib.import_module(module_name), ""
    except ImportError as e:
        return None, str(e)
    except Exception as e:                    # a driver can fail any way it likes
        return None, f"{type(e).__name__}: {e}"


def _missing(error_text, module_name):
    """True when the module is simply absent, rather than present and broken.

    Checks every ancestor of a dotted name: importing ``pypylon.pylon`` when
    pypylon is not installed reports "No module named 'pypylon'", and reading
    that as a load failure would tell the user their working installation is
    broken.
    """
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        name = ".".join(parts[:i])
        if (f"No module named '{name}'" in error_text
                or f"No module named {name}" in error_text):
            return True
    return False


# ---------------------------------------------------------------------------
# Descriptor for a camera the app may connect to
# ---------------------------------------------------------------------------

class CameraInfo:
    """One connectable source.

    ``key`` is the stable string written to the settings file so the app can
    reconnect to the same camera next session.  It is ``"<backend_id>:<device_id>"``
    — the device id is a serial number wherever the SDK exposes one, because
    an index changes the moment somebody unplugs a different camera.
    """

    def __init__(self, backend_cls, device_id, label, detail=""):
        self.backend_cls = backend_cls
        self.device_id = str(device_id)
        self.label = label
        self.detail = detail

    @property
    def backend_id(self):
        return self.backend_cls.backend_id

    @property
    def key(self):
        return f"{self.backend_id}:{self.device_id}"

    def __repr__(self):
        return f"<CameraInfo {self.key} {self.label!r}>"

    def __eq__(self, other):
        return isinstance(other, CameraInfo) and other.key == self.key

    def __hash__(self):
        return hash(self.key)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CameraBackend:
    """Common behaviour for every camera source.

    Subclasses override the ``_``-prefixed hooks; the public methods here add
    the parts that must behave identically whatever the hardware is — mono
    conversion, the software-binning fallback, and never raising out of a
    control setter (a camera that lacks a node must not take the UI down).
    """

    backend_id = "base"
    backend_name = "Camera"

    # Capability flags — the GUI greys out controls the camera cannot honour.
    supports_gain = False
    supports_exposure = False
    supports_frame_rate = False
    supports_hardware_binning = False
    supports_software_binning = True   # mean-pool in get_frame; almost always OK
    supports_resolution = False
    supports_native_dialog = False

    # Control ranges the GUI clamps its spin boxes to.  ``None`` means "leave
    # the built-in range alone"; a backend that can query the hardware fills
    # these in during _open.
    gain_range = None            # (min_dB, max_dB)
    exposure_range_ms = None     # (min_ms, max_ms)
    frame_rate_range = None      # (min_fps, max_fps)

    # Shown under the source chooser after connecting, for the caveats that
    # apply to one backend only (best-effort exposure on UVC, and so on).
    # ``note`` has to fit two short lines in the panel -- keep it under about
    # 52 characters; ``note_detail`` is the full version, which the app puts
    # in the tooltip and the status bar where there is room for it.
    note = ""
    note_detail = ""

    def __init__(self, info):
        self.info = info
        self.device_id = info.device_id
        self._open_flag = False
        self._running = False
        self._bin = 1
        self._hw_bin = 1          # what the hardware is actually doing
        self._meta = {}           # filled by _open, surfaced by metadata()

    # -- lifecycle ---------------------------------------------------------

    def open(self):
        if self._open_flag:
            return
        self._open()
        self._open_flag = True
        logger.info("Opened %s (%s)", self.info.label, self.info.key)

    def start(self):
        if not self._open_flag:
            self.open()
        if self._running:
            return
        self._start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        try:
            self._stop()
        except Exception:
            logger.exception("Error stopping %s", self.info.key)
        self._running = False

    def close(self):
        self.stop()
        if self._open_flag:
            try:
                self._close()
            except Exception:
                logger.exception("Error closing %s", self.info.key)
            self._open_flag = False

    @property
    def is_open(self):
        return self._open_flag

    @property
    def is_running(self):
        return self._running

    # -- frames ------------------------------------------------------------

    def get_frame(self, timeout_ms=500):
        """One frame as a 2-D mono array, or ``None`` on timeout.

        Never raises for an ordinary timeout or a single dropped frame — the
        capture thread polls this in a tight loop and a live growth must not
        end because one packet went missing.
        """
        frame = self._read(timeout_ms)
        if frame is None:
            return None
        frame = self._to_mono(frame)
        if self._bin > self._hw_bin:
            # Whatever binning the hardware could not do, do here.
            frame = self._software_bin(frame, self._bin // max(self._hw_bin, 1))
        return frame

    @staticmethod
    def _to_mono(frame):
        """2-D mono view of whatever the SDK handed back.

        Colour sensors are averaged to luminance; a Bayer-raw frame arrives
        2-D already and is left alone, which is what you want for RHEED —
        the de-mosaic would only interpolate intensity the sensor never saw.
        """
        frame = np.asarray(frame)
        if frame.ndim == 3:
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = frame[:, :, 0]
        return frame

    @staticmethod
    def _software_bin(frame, n):
        """n x n mean pooling, keeping the input dtype.

        Same signal-to-noise trade as hardware binning (n^2 more photons per
        output pixel) minus the read-noise advantage and the frame-rate gain,
        which is exactly what a camera without a binning node can offer.  The
        edge rows/columns that do not fill a whole bin are cropped, so the
        result is never padded with invented pixels.
        """
        if n <= 1:
            return frame
        h, w = frame.shape[:2]
        h -= h % n
        w -= w % n
        if h <= 0 or w <= 0:
            return frame
        view = frame[:h, :w].reshape(h // n, n, w // n, n)
        return view.mean(axis=(1, 3)).astype(frame.dtype)

    # -- controls (all best-effort; never raise) ---------------------------

    def set_gain(self, value_db):
        if self.supports_gain:
            self._guard("gain", self._set_gain, float(value_db))

    def set_exposure_ms(self, value_ms):
        if self.supports_exposure:
            self._guard("exposure", self._set_exposure_ms, float(value_ms))

    def set_frame_rate(self, fps):
        if self.supports_frame_rate:
            self._guard("frame rate", self._set_frame_rate, float(fps))

    def set_binning(self, n):
        """Ask for n x n binning, in hardware where possible.

        Returns the factor actually in force.  Hardware binning is preferred
        (it raises the frame rate as well as the signal); anything the camera
        cannot do is made up in :meth:`get_frame`.
        """
        n = max(1, int(n))
        self._bin = n
        self._hw_bin = 1
        if self.supports_hardware_binning:
            try:
                self._hw_bin = max(1, int(self._set_hw_binning(n)))
            except Exception as e:
                logger.warning("Hardware binning %sx failed on %s: %s",
                               n, self.info.key, e)
                self._hw_bin = 1
        return n

    @property
    def binning(self):
        return self._bin

    def max_binning(self):
        """Largest binning factor offered.  4 by default — the software path
        can bin any sensor, so the UI never needs to grey the buttons out."""
        return 4

    def list_resolutions(self):
        return []

    def set_resolution(self, width, height):
        pass

    def open_native_dialog(self):
        """Show the driver's own settings dialog (UVC/DirectShow only)."""

    def _guard(self, what, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            logger.debug("Could not set %s on %s: %s", what, self.info.key, e)

    # -- metadata ----------------------------------------------------------

    @property
    def binning_is_software(self):
        """True when some of the requested binning is being done here, not by
        the sensor -- either the camera has no binning node at all, or it
        capped out below what was asked for."""
        return self._bin > self._hw_bin

    def metadata(self):
        """(key, value) pairs written into exported CSV headers and sessions."""
        meta = {
            "Camera backend": self.backend_name,
            "Camera source": self.info.label,
        }
        meta.update(self._meta)
        meta["Binning"] = (f"{self._bin}x{self._bin}"
                           + ("" if self._hw_bin == self._bin else " (software)"))
        return meta

    # -- hooks for subclasses ---------------------------------------------

    def _open(self):        raise NotImplementedError
    def _start(self):       pass
    def _stop(self):        pass
    def _close(self):       pass
    def _read(self, timeout_ms):  raise NotImplementedError
    def _set_gain(self, v):       pass
    def _set_exposure_ms(self, v): pass
    def _set_frame_rate(self, v): pass
    def _set_hw_binning(self, n): return 1

    # -- discovery ---------------------------------------------------------

    # Name of the module :meth:`available` imports, and what to tell someone
    # who does not have it.  Backends with more to check (a GenTL producer
    # file, two possible module names) override the methods instead.
    driver_module = ""
    install_hint = "driver not installed"
    _import_error = ""

    @classmethod
    def available(cls):
        """True when this backend's driver can be imported on this machine."""
        if not cls.driver_module:
            return False
        module, error = _probe_import(cls.driver_module)
        cls._import_error = error
        return module is not None

    @classmethod
    def unavailable_reason(cls):
        """Why not -- as an instruction, not a label.

        Distinguishes a driver that is absent from one that is present and
        broken; the second reports the loader's own words, because that text
        is what identifies a version mismatch.
        """
        error = cls._import_error
        if error and not _missing(error, cls.driver_module):
            return (f"{cls.driver_module} is installed but will not load: "
                    f"{error}")
        return cls.install_hint

    @classmethod
    def enumerate(cls):
        """List the cameras this backend can see.  Must never raise."""
        return []


# ---------------------------------------------------------------------------
# Helpers shared by the GenICam-family backends
# ---------------------------------------------------------------------------
#
# Spinnaker, pylon, Vimba and any GenTL producer all expose the same SFNC
# feature names (Gain, ExposureTime, AcquisitionFrameRate, BinningHorizontal).
# Only the syntax for reaching them differs, so the "turn every automatic
# control off" sequence is written once as a list of feature/value pairs and
# each backend applies it with its own accessor.
#
# Why it matters for RHEED specifically: auto gain, auto exposure, auto black
# level and black-level clamping all move in discrete steps, and every step
# puts a sharp edge in the intensity-vs-time trace that looks exactly like the
# start of a growth transient.  Hardware gamma is worse — it makes intensity
# non-linear, so oscillation amplitudes stop meaning anything.

GENICAM_MANUAL_SETUP = [
    ("GainAuto", "Off"),
    ("ExposureAuto", "Off"),
    ("BlackLevelAuto", "Off"),
    ("BalanceWhiteAuto", "Off"),
    ("BlackLevelClampingEnable", False),
    ("GammaEnable", False),
    ("SharpnessEnable", False),
    ("AcquisitionMode", "Continuous"),
]

# Feature names that carry the device identity, in the order we try them.
GENICAM_ID_FEATURES = [
    ("Camera vendor", "DeviceVendorName"),
    ("Camera model", "DeviceModelName"),
    ("Camera serial", "DeviceSerialNumber"),
    ("Sensor pixel format", "PixelFormat"),
]


# ---------------------------------------------------------------------------
# FLIR / Point Grey — Spinnaker (PySpin)
# ---------------------------------------------------------------------------

_spin_system = None


def _spinnaker_system():
    """The one Spinnaker System instance, created on first use.

    Spinnaker refcounts this globally and misbehaves if a second instance is
    taken while the first is alive, so it lives here rather than in the app.
    """
    global _spin_system
    import PySpin
    if _spin_system is None:
        _spin_system = PySpin.System.GetInstance()
    return _spin_system


class SpinnakerBackend(CameraBackend):
    """FLIR / Point Grey cameras through the Spinnaker SDK.

    This is the backend VRHEED was originally written against — the Blackfly S
    on the MBE — so its behaviour is the reference the others imitate.
    """

    backend_id = "spinnaker"
    backend_name = "FLIR Spinnaker"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_hardware_binning = True

    driver_module = "PySpin"
    install_hint = ("PySpin not installed — it ships inside the FLIR Spinnaker "
                    "SDK installer as a .whl, not on PyPI. The wheel must match "
                    "your Python version (the cp311 in its filename).")

    @classmethod
    def unavailable_reason(cls):
        error = cls._import_error
        if error and not _missing(error, "PySpin"):
            # Overwhelmingly the wheel/SDK mismatch, and the loader error is
            # the only thing that says so.  Spell out the check rather than
            # leaving a DLL message on screen with no next step.
            return (f"PySpin is installed but will not load: {error}\n"
                    "Usually the Spinnaker SDK runtime is missing or is a "
                    "different version than the PySpin wheel. Check that "
                    "SpinView opens the camera, and that the wheel's cpXX "
                    "matches your Python version.")
        return cls.install_hint

    @classmethod
    def enumerate(cls):
        if not cls.available():
            return []
        out = []
        try:
            system = _spinnaker_system()
            cams = system.GetCameras()
            for i in range(cams.GetSize()):
                cam = cams[i]
                model = serial = ""
                try:
                    # The transport-layer node map is readable before Init(),
                    # so enumerating does not disturb a camera already in use
                    # by another program.
                    model = cam.TLDevice.DeviceModelName.GetValue()
                    serial = cam.TLDevice.DeviceSerialNumber.GetValue()
                except Exception:
                    pass
                del cam
                out.append(CameraInfo(cls, serial or str(i),
                                      f"FLIR {model or 'camera'}"
                                      + (f" [{serial}]" if serial else f" #{i}"),
                                      "Spinnaker"))
            cams.Clear()
        except Exception:
            logger.exception("Spinnaker enumeration failed")
        return out

    def _open(self):
        import PySpin
        self._PySpin = PySpin
        system = _spinnaker_system()
        self._cam_list = system.GetCameras()
        # Serial first, always.  FLIR serials are numeric ("21055099"), so a
        # "does it look like an index?" test picks the wrong lookup for every
        # real camera; the index path exists only for the case where the
        # transport layer would not give up a serial during enumeration.
        cam = None
        try:
            cam = self._cam_list.GetBySerial(self.device_id)
        except Exception:
            cam = None
        if cam is None and self.device_id.isdigit():
            idx = int(self.device_id)
            if idx < self._cam_list.GetSize():
                cam = self._cam_list[idx]
        if cam is None:
            raise CameraError(
                f"Camera {self.device_id} is no longer connected.\n\n"
                "Press the refresh button in the Camera tab to scan again.")
        self._cam = cam
        try:
            self._cam.Init()
        except Exception as e:
            raise CameraError(_spinnaker_hint(e)) from e

        for feat, val in GENICAM_MANUAL_SETUP:
            self._guard(feat, self._set_feature, feat, val)
        for label, feat in GENICAM_ID_FEATURES:
            try:
                self._meta[label] = str(getattr(self._cam, feat).GetValue())
            except Exception:
                pass
        for attr, dest in (("Gain", "gain_range"),
                           ("ExposureTime", "exposure_range_ms"),
                           ("AcquisitionFrameRate", "frame_rate_range")):
            try:
                node = getattr(self._cam, attr)
                lo, hi = float(node.GetMin()), float(node.GetMax())
                if dest == "exposure_range_ms":
                    lo, hi = lo / 1000.0, hi / 1000.0     # SDK works in us
                setattr(self, dest, (lo, hi))
            except Exception:
                pass
        self._guard("frame rate enable", self._set_feature,
                    "AcquisitionFrameRateEnable", True)

    def _set_feature(self, name, value):
        node = getattr(self._cam, name)
        if not isinstance(value, str):
            node.SetValue(value)
            return
        # Enumerations: use the SDK's own integer constant (PySpin.GainAuto_Off
        # and friends) exactly as VRHEED 2.1 did.  FromString is the GenApi
        # fallback for a node whose constant this PySpin build does not define;
        # it is second because the constants are the documented idiom and the
        # one this app has years of running time on.
        const = getattr(self._PySpin, f"{name}_{value}", None)
        if const is not None:
            node.SetValue(const)
        else:
            node.FromString(value)

    def _start(self):
        self._cam.BeginAcquisition()

    def _stop(self):
        try:
            self._cam.EndAcquisition()
        except Exception:
            pass

    def _close(self):
        try:
            self._cam.DeInit()
        except Exception:
            pass
        self._cam = None
        try:
            self._cam_list.Clear()
        except Exception:
            pass

    def _read(self, timeout_ms):
        res = None
        try:
            res = self._cam.GetNextImage(int(timeout_ms))
            if res is None or res.IsIncomplete():
                return None
            return res.GetNDArray().copy()
        except Exception:
            return None
        finally:
            # Spinnaker hands out a fixed pool of buffers; an image that is
            # never released is a buffer lost for the rest of the session,
            # and after a handful the stream stalls.
            if res is not None:
                try:
                    res.Release()
                except Exception:
                    pass

    def _set_gain(self, v):          self._cam.Gain.SetValue(v)
    def _set_exposure_ms(self, v):   self._cam.ExposureTime.SetValue(v * 1000.0)
    def _set_frame_rate(self, v):    self._cam.AcquisitionFrameRate.SetValue(v)

    def _set_hw_binning(self, n):
        n = min(n, int(self._cam.BinningHorizontal.GetMax()))
        self._cam.BinningHorizontal.SetValue(n)
        self._cam.BinningVertical.SetValue(n)
        # Binning shrinks the sensor's addressable area; without resetting the
        # offsets and re-maximising width/height the camera keeps the old,
        # now out-of-range window and refuses the change.
        for feat, val in (("OffsetX", 0), ("OffsetY", 0)):
            self._guard(feat, self._set_feature, feat, val)
        for feat in ("Width", "Height"):
            try:
                node = getattr(self._cam, feat)
                node.SetValue(node.GetMax())
            except Exception:
                pass
        return n


def _spinnaker_hint(exc):
    """Turn the two Spinnaker errors operators actually hit into instructions."""
    msg = str(exc)
    low = msg.lower()
    if "-1015" in msg or "wrong subnet" in low:
        return (f"{msg}\n\n"
                "This is a GigE network configuration problem — the NIC connected\n"
                "to the camera is on a different IP subnet than the camera.\n\n"
                "Fix:\n"
                "  1. Open Control Panel -> Network Adapters\n"
                "  2. Set the camera NIC to a static IP on the camera's subnet\n"
                "     (camera is likely 169.254.x.x -> set NIC to 169.254.0.1 / 255.255.0.0)\n"
                "  3. Or run Spinnaker's SpinView -> Action -> Auto Force IP")
    if "-1004" in msg or "access" in low:
        return (f"{msg}\n\n"
                "The camera is already open in another program (SpinView, or a "
                "second copy of VRHEED).  Close it and try again.")
    return msg


# ---------------------------------------------------------------------------
# Basler — pylon (pypylon)
# ---------------------------------------------------------------------------

class PylonBackend(CameraBackend):
    """Basler ace / dart / boost cameras through pypylon.

    Basler is the most common machine-vision brand on MBE systems after FLIR,
    and pypylon is a plain ``pip install pypylon`` — no vendor installer — so
    this is the easiest backend for a new user to get running.
    """

    backend_id = "pylon"
    backend_name = "Basler pylon"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_hardware_binning = True

    driver_module = "pypylon.pylon"
    install_hint = "pypylon not installed  (pip install pypylon)"

    @classmethod
    def enumerate(cls):
        if not cls.available():
            return []
        out = []
        try:
            from pypylon import pylon
            for dev in pylon.TlFactory.GetInstance().EnumerateDevices():
                serial = dev.GetSerialNumber()
                model = dev.GetModelName()
                out.append(CameraInfo(cls, serial or model,
                                      f"Basler {model} [{serial}]",
                                      dev.GetDeviceClass()))
        except Exception:
            logger.exception("pylon enumeration failed")
        return out

    def _open(self):
        from pypylon import pylon
        self._pylon = pylon
        tlf = pylon.TlFactory.GetInstance()
        target = None
        for dev in tlf.EnumerateDevices():
            if self.device_id in (dev.GetSerialNumber(), dev.GetModelName()):
                target = dev
                break
        if target is None:
            raise CameraError("Camera is no longer connected.")
        self._cam = pylon.InstantCamera(tlf.CreateDevice(target))
        self._cam.Open()
        for feat, val in GENICAM_MANUAL_SETUP:
            self._guard(feat, self._set_feature, feat, val)
        for label, feat in GENICAM_ID_FEATURES:
            try:
                self._meta[label] = str(getattr(self._cam, feat).GetValue())
            except Exception:
                pass
        # GigE ace classic (pylon 1.x feature set) spells the analogue
        # controls differently from USB3 ace 2 / dart, so every accessor below
        # tries the modern name first and falls back to the legacy one.
        self._guard("frame rate enable", self._set_feature,
                    "AcquisitionFrameRateEnable", True)

    def _node(self, *names):
        for n in names:
            node = getattr(self._cam, n, None)
            if node is not None:
                try:
                    node.GetValue()
                except Exception:
                    continue
                return node
        raise CameraError("no such node: " + "/".join(names))

    def _set_feature(self, name, value):
        # pypylon's node objects take the symbolic string for an enumeration
        # and the plain value for everything else, through the same setter.
        getattr(self._cam, name).SetValue(value)

    def _start(self):
        # LatestImageOnly: a UI that stalls for a moment must resume on the
        # live pattern, not replay a queue of stale frames.
        self._cam.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)

    def _stop(self):
        try:
            self._cam.StopGrabbing()
        except Exception:
            pass

    def _close(self):
        try:
            self._cam.Close()
        except Exception:
            pass
        self._cam = None

    def _read(self, timeout_ms):
        res = None
        try:
            res = self._cam.RetrieveResult(int(timeout_ms),
                                           self._pylon.TimeoutHandling_Return)
            if res is None or not res.GrabSucceeded():
                return None
            return res.Array.copy()
        except Exception:
            return None
        finally:
            if res is not None:
                try:
                    res.Release()
                except Exception:
                    pass

    def _set_gain(self, v):
        try:
            self._node("Gain").SetValue(v)
        except Exception:
            # Legacy GigE takes raw ADC counts, not dB; 1 dB ~ 32 raw on ace.
            self._node("GainRaw").SetValue(int(v * 32))

    def _set_exposure_ms(self, v):
        self._node("ExposureTime", "ExposureTimeAbs").SetValue(v * 1000.0)

    def _set_frame_rate(self, v):
        self._node("AcquisitionFrameRate", "AcquisitionFrameRateAbs").SetValue(v)

    def _set_hw_binning(self, n):
        node = self._node("BinningHorizontal")
        n = min(n, int(node.GetMax()))
        node.SetValue(n)
        self._node("BinningVertical").SetValue(n)
        for feat in ("OffsetX", "OffsetY"):
            self._guard(feat, self._set_feature, feat, 0)
        for feat in ("Width", "Height"):
            try:
                nd = getattr(self._cam, feat)
                nd.SetValue(nd.GetMax())
            except Exception:
                pass
        return n


# ---------------------------------------------------------------------------
# Allied Vision — Vimba X / Vimba (vmbpy, vimba)
# ---------------------------------------------------------------------------

class VimbaBackend(CameraBackend):
    """Allied Vision Manta / Alvium / Mako through Vimba.

    Vimba's Python API is written around ``with`` blocks, which do not fit an
    app that opens a camera in one method and reads it in another thread, so
    the contexts are entered and exited explicitly here.  That is supported —
    they are ordinary context managers — but it does mean ``close()`` has to
    unwind them in the right order or the SDK leaks the device handle.
    """

    backend_id = "vimba"
    backend_name = "Allied Vision Vimba"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_hardware_binning = True

    @staticmethod
    def _module():
        try:
            import vmbpy
            return vmbpy, vmbpy.VmbSystem
        except Exception:
            pass
        import vimba                                  # Vimba 5/6 (deprecated)
        return vimba, vimba.Vimba

    install_hint = ("vmbpy not installed  (ships with the Allied Vision "
                    "Vimba X SDK)")

    @classmethod
    def available(cls):
        # Vimba X ships vmbpy; Vimba 5/6 shipped vimba.  Report the newer
        # one's failure unless it is simply absent and the older one loads.
        module, error = _probe_import("vmbpy")
        if module is not None:
            cls._import_error = ""
            return True
        if _missing(error, "vmbpy"):
            legacy, legacy_error = _probe_import("vimba")
            if legacy is not None:
                cls._import_error = ""
                return True
            cls._import_error = error if _missing(legacy_error, "vimba") else legacy_error
        else:
            cls._import_error = error
        return False

    @classmethod
    def unavailable_reason(cls):
        error = cls._import_error
        if error and not _missing(error, "vmbpy") and not _missing(error, "vimba"):
            return f"vmbpy is installed but will not load: {error}"
        return cls.install_hint

    @classmethod
    def enumerate(cls):
        if not cls.available():
            return []
        out = []
        try:
            _mod, api = cls._module()
            with api.get_instance() as sysobj:
                for cam in sysobj.get_all_cameras():
                    cid = cam.get_id()
                    try:
                        model = cam.get_model()
                    except Exception:
                        model = "camera"
                    out.append(CameraInfo(cls, cid, f"Allied Vision {model}", cid))
        except Exception:
            logger.exception("Vimba enumeration failed")
        return out

    def _open(self):
        _mod, api = self._module()
        self._sys_ctx = api.get_instance()
        sysobj = self._sys_ctx.__enter__()
        try:
            cam = sysobj.get_camera_by_id(self.device_id)
        except Exception as e:
            self._sys_ctx.__exit__(None, None, None)
            raise CameraError(f"Camera {self.device_id} not found: {e}") from e
        self._cam_ctx = cam
        self._cam = cam.__enter__()
        for feat, val in GENICAM_MANUAL_SETUP:
            self._guard(feat, self._set_feature, feat, val)
        for label, feat in GENICAM_ID_FEATURES:
            try:
                self._meta[label] = str(self._cam.get_feature_by_name(feat).get())
            except Exception:
                pass

    def _set_feature(self, name, value):
        self._cam.get_feature_by_name(name).set(value)

    def _close(self):
        # Camera first, then the system: Vimba leaks the device handle if the
        # system context closes while a camera is still inside it.
        for ctx in (getattr(self, "_cam_ctx", None),
                    getattr(self, "_sys_ctx", None)):
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
        self._cam = self._cam_ctx = self._sys_ctx = None

    def _read(self, timeout_ms):
        try:
            frame = self._cam.get_frame(timeout_ms=int(timeout_ms))
            arr = frame.as_numpy_ndarray()
            return np.array(arr, copy=True)
        except Exception:
            return None

    def _set_gain(self, v):        self._set_feature("Gain", float(v))
    def _set_exposure_ms(self, v): self._set_feature("ExposureTime", float(v) * 1000.0)

    def _set_frame_rate(self, v):
        self._guard("fps enable", self._set_feature,
                    "AcquisitionFrameRateEnable", True)
        for name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
            try:
                self._set_feature(name, float(v))
                return
            except Exception:
                continue

    def _set_hw_binning(self, n):
        self._set_feature("BinningHorizontal", int(n))
        self._set_feature("BinningVertical", int(n))
        return n


# ---------------------------------------------------------------------------
# Any GigE Vision / USB3 Vision camera — GenICam GenTL (harvesters)
# ---------------------------------------------------------------------------

def _gentl_producers():
    """Paths of the GenTL producer (.cti) libraries installed on this machine.

    A .cti is the vendor's GenTL driver.  Every GenICam-compliant vendor ships
    one, and any of them will usually drive any other vendor's GigE Vision or
    USB3 Vision camera — which is the whole point of this backend: install one
    SDK, get every standards-compliant camera, including the ones VRHEED has
    never heard of.
    """
    paths = []
    # The standard environment variables, set by every vendor's installer.
    for var in ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"):
        for d in os.environ.get(var, "").split(os.pathsep):
            if d and os.path.isdir(d):
                paths.append(d)
    # ...plus the usual install locations, for the machines where the
    # installer did not set the variable (or the app was launched from a
    # shell that never saw it).
    if platform.system() == "Windows":
        paths += [
            r"C:\Program Files\Basler\pylon 7\Runtime\x64",
            r"C:\Program Files\Basler\pylon 6\Runtime\x64",
            r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64",
            r"C:\Program Files\Allied Vision\Vimba X\cti",
            r"C:\Program Files\Allied Vision\Vimba_6.0\VimbaGigETL\Bin\Win64",
            r"C:\Program Files\Point Grey Research\Spinnaker\bin64\vs2015",
            r"C:\Program Files\FLIR Systems\Spinnaker\bin64\vs2015",
            r"C:\Program Files\Lucid Vision Labs\Arena SDK\x64Release",
            r"C:\Program Files\Baumer\Baumer GAPI SDK\Components\Bin\x64",
            r"C:\Program Files\XIMEA\GenTL Producer",
            r"C:\Program Files\IDS\ids_peak\generic_sdk\api\lib\x86_64",
        ]
    else:
        paths += [
            "/opt/pylon/lib/gentlproducer/gtl",
            "/opt/pylon/lib64/gentlproducer/gtl",
            "/opt/VimbaX/cti",
            "/opt/mvIMPACT_Acquire/lib/x86_64",
            "/opt/ArenaSDK/lib64",
            "/usr/lib/ids/cti",
        ]
    found = []
    for d in paths:
        try:
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if name.lower().endswith(".cti"):
                    full = os.path.join(d, name)
                    if full not in found:
                        found.append(full)
        except Exception:
            continue
    return found


class GenICamBackend(CameraBackend):
    """Any GenICam camera, through a GenTL producer, via harvesters.

    Covers the vendors with no dedicated backend here — IDS, Lucid, JAI,
    Baumer, Ximea, Matrix Vision, Photonfocus, Emergent and the rest — plus
    Basler and Allied Vision when their Python bindings are not installed but
    their SDK is.  If a camera says "GigE Vision" or "USB3 Vision" on the
    datasheet, this is the backend that will talk to it.
    """

    backend_id = "genicam"
    backend_name = "GenICam / GenTL"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_hardware_binning = True

    _harvester = None     # one Harvester for the process; producers are heavy

    install_hint = "harvesters not installed  (pip install harvesters)"

    @classmethod
    def available(cls):
        # Two things have to be true: the Python binding, and a vendor's GenTL
        # producer for it to drive.  They fail independently and need
        # different fixes, so they are reported separately.
        module, error = _probe_import("harvesters.core")
        cls._import_error = error
        return module is not None and bool(_gentl_producers())

    @classmethod
    def unavailable_reason(cls):
        error = cls._import_error
        if error:
            if not _missing(error, "harvesters"):
                return f"harvesters is installed but will not load: {error}"
            return cls.install_hint
        return ("no GenTL producer (.cti) found — install any vendor SDK, or set "
                "GENICAM_GENTL64_PATH to the folder holding its .cti file")

    @classmethod
    def _get_harvester(cls, rescan=False):
        from harvesters.core import Harvester
        if cls._harvester is None:
            cls._harvester = Harvester()
            for cti in _gentl_producers():
                try:
                    cls._harvester.add_file(cti)
                except Exception as e:
                    logger.debug("GenTL producer %s rejected: %s", cti, e)
            rescan = True
        if rescan:
            cls._harvester.update()
        return cls._harvester

    @classmethod
    def enumerate(cls):
        if not cls.available():
            return []
        out = []
        try:
            h = cls._get_harvester(rescan=True)
            for i, dev in enumerate(h.device_info_list):
                model = getattr(dev, "model", "") or "camera"
                vendor = getattr(dev, "vendor", "") or ""
                serial = getattr(dev, "serial_number", "") or str(i)
                out.append(CameraInfo(cls, serial,
                                      f"{vendor} {model} [{serial}]".strip(),
                                      "GenTL"))
        except Exception:
            logger.exception("GenICam enumeration failed")
        return out

    def _open(self):
        h = self._get_harvester()
        index = None
        for i, dev in enumerate(h.device_info_list):
            if str(getattr(dev, "serial_number", "")) == self.device_id:
                index = i
                break
        if index is None:
            h.update()
            for i, dev in enumerate(h.device_info_list):
                if str(getattr(dev, "serial_number", "")) == self.device_id:
                    index = i
                    break
        if index is None:
            raise CameraError("Camera is no longer connected.")
        # harvesters renamed create_image_acquirer() to create() in 1.4.
        self._ia = (h.create(index) if hasattr(h, "create")
                    else h.create_image_acquirer(index))
        self._nm = self._ia.remote_device.node_map
        for feat, val in GENICAM_MANUAL_SETUP:
            self._guard(feat, self._set_feature, feat, val)
        for label, feat in GENICAM_ID_FEATURES:
            try:
                self._meta[label] = str(getattr(self._nm, feat).value)
            except Exception:
                pass

    def _set_feature(self, name, value):
        getattr(self._nm, name).value = value

    def _start(self):
        self._ia.start()

    def _stop(self):
        try:
            self._ia.stop()
        except Exception:
            pass

    def _close(self):
        try:
            self._ia.destroy()
        except Exception:
            pass
        self._ia = self._nm = None

    def _read(self, timeout_ms):
        try:
            with self._ia.fetch(timeout=timeout_ms / 1000.0) as buf:
                comp = buf.payload.components[0]
                h, w = int(comp.height), int(comp.width)
                data = comp.data
                # Colour payloads arrive flat; reshape to (h, w, channels) so
                # the base class can reduce them the same way as everything else.
                chan = max(1, data.size // max(h * w, 1))
                arr = data.reshape(h, w) if chan == 1 else data.reshape(h, w, chan)
                # The buffer is recycled the moment this block exits.
                return np.array(arr, copy=True)
        except Exception:
            return None

    def _set_gain(self, v):          self._set_feature("Gain", float(v))
    def _set_exposure_ms(self, v):   self._set_feature("ExposureTime", float(v) * 1000.0)

    def _set_frame_rate(self, v):
        self._guard("fps enable", self._set_feature,
                    "AcquisitionFrameRateEnable", True)
        self._set_feature("AcquisitionFrameRate", float(v))

    def _set_hw_binning(self, n):
        self._set_feature("BinningHorizontal", int(n))
        self._set_feature("BinningVertical", int(n))
        return n


# ---------------------------------------------------------------------------
# Scientific CCD / sCMOS cameras — pylablib
# ---------------------------------------------------------------------------
#
# The high-sensitivity end of RHEED: Andor iXon/Zyla, Hamamatsu ORCA,
# Princeton Instruments PIXIS, Thorlabs scientific cameras, pco.edge — and,
# just as usefully, National Instruments IMAQ frame grabbers, which is how a
# 1990s analogue RHEED camera gets digitised on a system nobody wants to
# rewire.  pylablib wraps all of them behind one API, so one backend class
# with a family table covers the lot.

# (family id, label, "how many / which are there", "open the n-th one")
# Each entry is resolved lazily so that importing one broken vendor DLL cannot
# stop the others from being listed.
PYLABLIB_FAMILIES = [
    ("andor_sdk3",  "Andor sCMOS",             "Andor",  "get_cameras_number_SDK3", "AndorSDK3Camera"),
    ("andor_sdk2",  "Andor CCD/EMCCD",         "Andor",  "get_cameras_number_SDK2", "AndorSDK2Camera"),
    ("dcam",        "Hamamatsu",               "DCAM",   "get_cameras_number",      "DCAMCamera"),
    ("thorlabs_tl", "Thorlabs scientific",     "Thorlabs", "list_cameras_tlcam",    "ThorlabsTLCamera"),
    ("picam",       "Princeton Instruments",   "PrincetonInstruments", "list_cameras", "PicamCamera"),
    ("pvcam",       "Photometrics",            "Photometrics", "list_cameras",      "PvcamCamera"),
    ("pco",         "PCO",                     "PCO",    "get_cameras_number",      "PCOSC2Camera"),
    ("uc480",       "IDS uEye / Thorlabs DCx", "uc480",  "get_cameras_number",      "UC480Camera"),
    ("imaqdx",      "NI IMAQdx",               "IMAQdx", "list_cameras",            "IMAQdxCamera"),
    ("imaq",        "NI IMAQ frame grabber",   "IMAQ",   "list_cameras",            "IMAQCamera"),
]


class PylablibBackend(CameraBackend):
    """Scientific cameras and frame grabbers through pylablib.

    pylablib's camera objects share one interface whatever the hardware is:
    ``setup_acquisition`` / ``start_acquisition`` / ``wait_for_frame`` /
    ``read_newest_image``.  Not every camera implements every control, so each
    setter here is a best-effort call — the base class swallows what the
    hardware refuses and the GUI keeps working.
    """

    backend_id = "pylablib"
    backend_name = "Scientific camera (pylablib)"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = False    # exposure sets the rate on most of these
    supports_hardware_binning = True
    note = "Controls this camera lacks are ignored."
    note_detail = ("Scientific cameras vary in which controls they expose; "
                   "anything this camera refuses is simply ignored.")

    driver_module = "pylablib"
    install_hint = "pylablib not installed  (pip install pylablib)"

    @staticmethod
    def _family(fam_id):
        for entry in PYLABLIB_FAMILIES:
            if entry[0] == fam_id:
                return entry
        raise CameraError(f"unknown camera family {fam_id!r}")

    @staticmethod
    def _module(mod_name):
        from pylablib import devices
        return getattr(devices, mod_name)

    @classmethod
    def enumerate(cls):
        if not cls.available():
            return []
        out = []
        for fam_id, label, mod_name, lister, _opener in PYLABLIB_FAMILIES:
            try:
                mod = cls._module(mod_name)
                found = getattr(mod, lister)()
            except Exception as e:
                # A missing vendor DLL is the normal case, not an error.
                logger.debug("pylablib %s unavailable: %s", fam_id, e)
                continue
            try:
                items = (list(range(int(found))) if isinstance(found, int)
                         else list(found))
            except Exception:
                continue
            for item in items:
                out.append(CameraInfo(cls, f"{fam_id}/{item}",
                                      f"{label} #{item}", "pylablib"))
        return out

    def _open(self):
        fam_id, _, mod_name, _, opener = self._family(self.device_id.split("/", 1)[0])
        ident = self.device_id.split("/", 1)[1]
        mod = self._module(mod_name)
        cls_ = getattr(mod, opener)
        try:
            self._cam = cls_(int(ident))
        except (ValueError, TypeError):
            self._cam = cls_(ident)
        if hasattr(self._cam, "open") and not getattr(self._cam, "is_opened", lambda: True)():
            self._cam.open()
        try:
            info = self._cam.get_device_info()
            self._meta["Camera model"] = str(getattr(info, "model", info))
            serial = getattr(info, "serial_number", None)
            if serial:
                self._meta["Camera serial"] = str(serial)
        except Exception:
            pass
        self._meta.setdefault("Camera model", self.info.label)

    def _start(self):
        # A short ring buffer: enough that a slow UI tick does not tear a
        # frame, small enough that "newest image" is still the live pattern.
        try:
            self._cam.setup_acquisition(mode="sequence", nframes=16)
        except Exception:
            pass
        self._cam.start_acquisition()

    def _stop(self):
        try:
            self._cam.stop_acquisition()
        except Exception:
            pass

    def _close(self):
        try:
            self._cam.close()
        except Exception:
            pass
        self._cam = None

    def _read(self, timeout_ms):
        try:
            self._cam.wait_for_frame(timeout=timeout_ms / 1000.0)
            img = self._cam.read_newest_image()
            return None if img is None else np.asarray(img)
        except Exception:
            return None

    def _set_gain(self, v):
        # Andor EMCCD and the sCMOS families spell this differently; try the
        # generic setter first, then the EM-gain one.
        for name in ("set_gain", "set_EMCCD_gain"):
            fn = getattr(self._cam, name, None)
            if fn is not None:
                fn(v)
                return

    def _set_exposure_ms(self, v):
        self._cam.set_exposure(v / 1000.0)      # pylablib works in seconds

    def _set_hw_binning(self, n):
        self._cam.set_roi(hbin=int(n), vbin=int(n))
        return n


# ---------------------------------------------------------------------------
# USB / UVC webcams, and analogue cameras on a USB frame grabber — OpenCV
# ---------------------------------------------------------------------------

def _uvc_backend_id():
    """The OpenCV capture backend to use for local video devices.

    Picked per platform rather than left to OpenCV's auto-selection: on
    Windows the default MSMF path takes seconds to open a device and reports
    the wrong frame rate, and DirectShow is also the only backend that exposes
    the driver's own property dialog.
    """
    sysname = platform.system()
    if sysname == "Windows":
        return getattr(cv2, "CAP_DSHOW", 0)
    if sysname == "Darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", 0)
    return getattr(cv2, "CAP_V4L2", 0)


# Resolutions probed when a UVC camera is connected.  Webcams default to
# 640x480 whatever the sensor can do, and on a RHEED screen those missing
# pixels are missing streak-spacing resolution, so it is worth asking.
_UVC_PROBE_SIZES = [(3840, 2160), (2592, 1944), (1920, 1080), (1600, 1200),
                    (1280, 1024), (1280, 720), (1024, 768), (800, 600),
                    (640, 480), (320, 240)]


class UsbCameraBackend(CameraBackend):
    """Any device the OS presents as a video capture device.

    That is a much bigger set than "webcam": USB machine-vision cameras in UVC
    mode, USB microscope cameras, HDMI and composite capture dongles, and the
    analogue-to-USB frame grabbers that a pre-2005 RHEED camera plugs into.
    They all arrive here as an index.

    Gain and exposure are best-effort.  UVC drivers disagree about the units
    (DirectShow wants log2 seconds, V4L2 wants 100 us ticks) and many webcams
    quietly ignore the request; where the driver has its own dialog, the
    "Driver settings" button is the reliable way in.
    """

    backend_id = "usb"
    backend_name = "USB / UVC camera"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_resolution = True
    note = "Approximate gain/exposure; turn auto-exposure off."
    note_detail = ("UVC gain and exposure are driver-dependent and "
                   "approximate. Every auto-exposure step puts an edge in the "
                   "intensity trace that looks like a growth transient, so "
                   "turn auto-exposure off -- on Windows the Driver button "
                   "opens the camera's own property sheet.")

    # Enumeration opens every device in turn, which is slow enough to be worth
    # doing once per Refresh rather than on every dialog.
    _cache = None

    @classmethod
    def available(cls):
        return True

    @classmethod
    def supports_native_dialog_here(cls):
        return platform.system() == "Windows"

    @classmethod
    def enumerate(cls, max_index=8):
        out = []
        api = _uvc_backend_id()
        for idx in range(max_index):
            cap = None
            try:
                cap = cv2.VideoCapture(idx, api) if api else cv2.VideoCapture(idx)
                if not cap.isOpened():
                    # Index gaps are real on Linux (a device can be
                    # /dev/video2 with nothing on 0 or 1), so keep probing
                    # instead of stopping at the first miss.
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                out.append(CameraInfo(cls, str(idx),
                                      f"USB camera {idx}  ({w}x{h})", "UVC"))
            except Exception:
                continue
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return out

    def _open(self):
        api = _uvc_backend_id()
        idx = int(self.device_id)
        self._cap = cv2.VideoCapture(idx, api) if api else cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open USB camera {idx}.\n\n"
                "It may be in use by another program, or (on macOS) VRHEED may "
                "not have been granted camera access in System Settings > "
                "Privacy & Security > Camera.")
        self.supports_native_dialog = self.supports_native_dialog_here()
        # A one-frame driver buffer: with the default 4-deep queue the live
        # display runs a visible fraction of a second behind the shutter.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._meta["Camera model"] = f"UVC device {idx}"
        self._resolutions = None

    def _close(self):
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _read(self, timeout_ms):
        # VideoCapture.read() blocks until the next frame; the capture thread
        # is the only caller, so blocking there is exactly what we want.
        try:
            ok, frame = self._cap.read()
            return frame if ok else None
        except Exception:
            return None

    def _set_gain(self, v):
        self._cap.set(cv2.CAP_PROP_GAIN, float(v))

    def _set_exposure_ms(self, v):
        # Take the camera out of auto first, or the next auto update simply
        # overwrites whatever we set — and an auto-exposure step in the middle
        # of a growth is indistinguishable from a real intensity transient.
        self._disable_auto_exposure()
        self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_value(v))

    def _disable_auto_exposure(self):
        # 0.25 is the DirectShow/UVC "manual" magic value; V4L2 uses 1.
        for val in (0.25, 1):
            try:
                if self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                    return
            except Exception:
                continue

    @staticmethod
    def _exposure_value(ms):
        """Exposure in the units this platform's UVC driver expects."""
        ms = max(ms, 1e-3)
        if platform.system() == "Windows":
            # DirectShow: exposure is log2(seconds), clamped to the range
            # every webcam supports.
            return float(np.clip(round(np.log2(ms / 1000.0)), -13, 0))
        # V4L2 and AVFoundation: exposure_absolute in 100 us units.
        return float(max(1, round(ms * 10)))

    def _set_frame_rate(self, v):
        self._cap.set(cv2.CAP_PROP_FPS, float(v))

    def list_resolutions(self):
        """Sizes the device actually accepts, probed once on first ask.

        OpenCV has no query for this, so each candidate is set and read back:
        a driver that cannot do 1920x1080 silently gives you 640x480, and the
        read-back is how you find out.
        """
        if self._resolutions is not None:
            return self._resolutions
        found = []
        cur = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for w, h in _UVC_PROBE_SIZES:
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                got = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                if got not in found and got[0] > 0:
                    found.append(got)
            except Exception:
                continue
        self.set_resolution(*cur)
        self._resolutions = sorted(found, key=lambda wh: -wh[0] * wh[1])
        return self._resolutions

    def set_resolution(self, width, height):
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def open_native_dialog(self):
        """DirectShow's own property sheet — the only place some webcams let
        you turn auto-exposure and auto-white-balance off for good."""
        try:
            self._cap.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception as e:
            raise CameraError(f"No driver dialog available: {e}") from e

    def metadata(self):
        meta = super().metadata()
        try:
            meta["Frame width (px)"] = str(int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            meta["Frame height (px)"] = str(int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except Exception:
            pass
        return meta


# ---------------------------------------------------------------------------
# Network cameras — RTSP / HTTP-MJPEG / GigE streamers
# ---------------------------------------------------------------------------

# URLs the operator has added by hand.  They cannot be discovered, so the app
# stores them in its settings and hands them back here at start-up.
_network_urls = []


def register_network_camera(url):
    """Remember a stream URL so it appears in the camera list."""
    url = url.strip()
    if url and url not in _network_urls:
        _network_urls.append(url)
    return url


def forget_network_camera(url):
    if url in _network_urls:
        _network_urls.remove(url)


def network_cameras():
    return list(_network_urls)


class NetworkCameraBackend(CameraBackend):
    """A camera reached over the network rather than over a cable.

    Two situations make this the right answer:  the RHEED camera is an IP
    camera (an Axis or Hikvision on the chamber, streaming RTSP), or the
    camera is wired to a different computer — typically the one running the
    vendor's own software — which re-streams it.  Either way VRHEED only needs
    the URL:

        rtsp://user:password@192.168.1.64:554/Streaming/Channels/101
        http://192.168.1.90/mjpg/video.mjpg

    Exposure and gain belong to the camera's own web interface; VRHEED only
    receives the decoded pictures.  Bear in mind that the stream is usually
    H.264, so the intensities you measure have been through a lossy codec —
    fine for oscillation timing, not for absolute photometry.
    """

    backend_id = "network"
    backend_name = "Network camera (RTSP/HTTP)"
    note = "Compressed stream: timing valid, intensity is not."
    note_detail = ("The stream is usually H.264, so intensities have been "
                   "through a lossy codec: fine for oscillation timing, not "
                   "for absolute photometry. Gain and exposure belong to the "
                   "camera's own web interface.")

    @classmethod
    def available(cls):
        return True

    @classmethod
    def enumerate(cls):
        return [CameraInfo(cls, url, f"Network: {_redact(url)}", "stream")
                for url in _network_urls]

    def _open(self):
        # FFMPEG is the backend that handles RTSP and MJPEG-over-HTTP; the
        # platform defaults do not.
        self._cap = cv2.VideoCapture(self.device_id,
                                     getattr(cv2, "CAP_FFMPEG", 0))
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open stream:\n{_redact(self.device_id)}\n\n"
                "Check the URL, that the camera is reachable from this machine, "
                "and that any username/password is included in the URL.")
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._meta["Camera model"] = _redact(self.device_id)

    def _close(self):
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _read(self, timeout_ms):
        try:
            ok, frame = self._cap.read()
            return frame if ok else None
        except Exception:
            return None


def _redact(url):
    """Hide the password in a stream URL before it reaches a log or a label."""
    return re.sub(r"://[^/@]*:([^/@]*)@", "://***:***@", url)


# ---------------------------------------------------------------------------
# Screen capture — digitising whatever the vendor software is already showing
# ---------------------------------------------------------------------------

class ScreenCaptureBackend(CameraBackend):
    """Capture a rectangle of the desktop as if it were a camera.

    This is the escape hatch for the systems where the RHEED camera is not
    available to VRHEED at all: a kSA 400 or Staib installation whose software
    owns the camera exclusively, a frame grabber with a Windows-only viewer and
    no SDK, or a camera on a machine you are only allowed to remote into.
    Point this at the live-image pane of whatever is already running and every
    VRHEED measurement works on it.

    Two honest caveats.  What you measure is the vendor software's *display*,
    so its own contrast curve and any 8-bit conversion are baked in — relative
    oscillations survive, absolute intensities do not.  And the frame rate is
    the capture rate, not the camera's, so the FFT axis is only as good as the
    screen refresh.
    """

    backend_id = "screen"
    backend_name = "Screen capture"
    supports_frame_rate = True
    note = "Another program's display: relative changes only."
    note_detail = ("You are measuring the vendor software's display, so its "
                   "contrast curve and 8-bit conversion are baked in: "
                   "relative changes survive, absolute intensities do not, "
                   "and the frame rate is the capture rate rather than the "
                   "camera's. Camera > Screen capture region crops it.")

    driver_module = "mss"
    install_hint = "mss not installed  (pip install mss)"

    @classmethod
    def enumerate(cls):
        if not cls.available():
            return []
        out = []
        try:
            import mss
            with mss.mss() as sct:
                # monitors[0] is the union of every screen; 1..n are the
                # individual ones, which is what an operator means by "the
                # second monitor".
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue
                    out.append(CameraInfo(
                        cls, str(i),
                        f"Screen {i}  ({mon['width']}x{mon['height']})",
                        "desktop"))
        except Exception:
            logger.exception("Screen enumeration failed")
        return out

    def __init__(self, info):
        super().__init__(info)
        # device_id is "<monitor>" or "<monitor>@x,y,w,h" in desktop pixels.
        self.monitor_index, self.region = parse_screen_id(info.device_id)
        self._target_fps = 30.0
        self._next_due = 0.0
        self._sct = None
        self._sct_thread = None

    def _open(self):
        import mss
        self._mss = mss
        with mss.mss() as sct:
            if self.monitor_index >= len(sct.monitors):
                raise CameraError(f"Screen {self.monitor_index} is not connected.")
            mon = sct.monitors[self.monitor_index]
        if self.region:
            x, y, w, h = self.region
            self._box = {"left": int(mon["left"]) + int(x),
                         "top": int(mon["top"]) + int(y),
                         "width": int(w), "height": int(h)}
        else:
            self._box = {k: mon[k] for k in ("left", "top", "width", "height")}
        self._meta["Camera model"] = (
            f"Screen {self.monitor_index} region "
            f"{self._box['width']}x{self._box['height']} "
            f"at ({self._box['left']}, {self._box['top']})")

    def _close(self):
        self._release_sct()

    def _release_sct(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        self._sct = None
        self._sct_thread = None

    def _read(self, timeout_ms):
        # An mss instance belongs to the thread that created it (it holds an
        # X11/Quartz/GDI connection), so make it here, in the capture thread,
        # and rebuild it if the thread ever changes.
        me = threading.get_ident()
        if self._sct is None or self._sct_thread != me:
            self._release_sct()
            self._sct = self._mss.mss()
            self._sct_thread = me
        # Nothing throttles a screen grab, so pace it to the requested rate;
        # otherwise this backend would spin a core flat out.
        now = time.monotonic()
        wait = self._next_due - now
        if wait > 0:
            time.sleep(min(wait, timeout_ms / 1000.0))
        self._next_due = max(now, self._next_due) + 1.0 / max(self._target_fps, 1.0)
        try:
            shot = self._sct.grab(self._box)
            # mss returns BGRA; the base class reduces it to luminance.
            return np.asarray(shot)
        except Exception:
            return None

    def _set_frame_rate(self, v):
        self._target_fps = max(1.0, float(v))


def parse_screen_id(device_id):
    """'2@100,50,800,600' -> (2, (100, 50, 800, 600));  '2' -> (2, None)."""
    if "@" in device_id:
        idx, rect = device_id.split("@", 1)
        try:
            nums = tuple(int(float(p)) for p in rect.split(","))
            if len(nums) == 4 and nums[2] > 0 and nums[3] > 0:
                return int(idx), nums
        except Exception:
            pass
        return int(idx), None
    return int(device_id or 1), None


def screen_source_key(monitor_index, region=None):
    """Build the CameraInfo device id for a screen region."""
    if region:
        x, y, w, h = (int(v) for v in region)
        return f"{int(monitor_index)}@{x},{y},{w},{h}"
    return str(int(monitor_index))


# ---------------------------------------------------------------------------
# Synthetic camera — demo, teaching and tests, no hardware
# ---------------------------------------------------------------------------

class SyntheticBackend(CameraBackend):
    """A simulated RHEED pattern: streaks plus an oscillating specular spot.

    Always present, so VRHEED can be installed, opened and learned on a laptop
    with nothing attached — which is how most people will first meet it — and
    so the ROI, FFT and lattice tools have a source with a *known* answer:
    the specular intensity oscillates at exactly 0.25 Hz (4 s per monolayer)
    and the streaks sit at a fixed spacing.
    """

    backend_id = "synthetic"
    backend_name = "Simulated RHEED pattern"
    supports_gain = True
    supports_exposure = True
    supports_frame_rate = True
    supports_hardware_binning = False
    note = "Simulated pattern - specular beats at 0.25 Hz."
    note_detail = ("Simulated source with a known answer: the specular spot "
                   "oscillates at exactly 0.25 Hz, so the measured growth "
                   "rate should read 4.00 s/ML.")

    WIDTH, HEIGHT = 1024, 768
    OSC_HZ = 0.25

    @classmethod
    def available(cls):
        return True

    @classmethod
    def enumerate(cls):
        return [CameraInfo(cls, "0", "Simulated RHEED pattern", "no hardware")]

    def _open(self):
        h, w = self.HEIGHT, self.WIDTH
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # The pattern is static apart from the specular intensity, so the two
        # parts are built once here: a per-frame exp() over a megapixel would
        # cap the simulated frame rate below 30 fps on a laptop.
        base = np.full((h, w), 4.0, np.float32)
        base += 10.0 * np.clip((yy - h * 0.30) / h, 0, 1)   # diffuse background
        spec = np.zeros((h, w), np.float32)
        for order, amp in ((0, 1.0), (-1, 0.55), (1, 0.55), (-2, 0.22), (2, 0.22)):
            cx = w / 2 + order * (w * 0.13)
            cy = h * 0.55 + (order ** 2) * (h * 0.010)   # curvature of the arc
            blob = (120.0 * amp *
                    np.exp(-((xx - cx) ** 2) / (2 * 8.0 ** 2)
                           - ((yy - cy) ** 2) / (2 * 60.0 ** 2))).astype(np.float32)
            # Only the specular beam breathes with layer completion; the side
            # streaks stay put, as on a real surface.
            if order == 0:
                spec += blob
            else:
                base += blob
        self._base, self._spec = base, spec
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(0)
        self._gain = 15.0
        self._exposure_ms = 30.0
        self._target_fps = 30.0
        self._next_due = 0.0
        self._meta.update({
            "Camera vendor": "VRHEED",
            "Camera model": "Simulated RHEED source",
            "Camera serial": "synthetic",
        })

    def _read(self, timeout_ms):
        now = time.monotonic()
        wait = self._next_due - now
        if wait > 0:
            time.sleep(min(wait, timeout_ms / 1000.0))
        self._next_due = max(now, self._next_due) + 1.0 / max(self._target_fps, 1.0)
        t = time.monotonic() - self._t0

        osc = 0.55 + 0.45 * np.cos(2 * np.pi * self.OSC_HZ * t)
        img = self._base + np.float32(osc) * self._spec
        # Exposure and gain act the way they do on a real sensor: exposure
        # scales the signal, gain scales everything after it, and the noise
        # keeps its shot-noise square-root dependence on the signal.
        img *= np.float32((self._exposure_ms / 30.0)
                          * (10 ** (self._gain / 20.0) / 5.62))
        noise = self._rng.standard_normal(img.shape, dtype=np.float32)
        img += noise * (1.5 + 0.08 * np.sqrt(np.maximum(img, 0)))
        return np.clip(img, 0, 255).astype(np.uint8)

    def _set_gain(self, v):          self._gain = float(v)
    def _set_exposure_ms(self, v):   self._exposure_ms = float(v)
    def _set_frame_rate(self, v):    self._target_fps = max(1.0, float(v))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Order matters: it is the order cameras appear in the chooser, and the first
# entry that finds a device is the one auto-connected at start-up.  Dedicated
# vendor SDKs come before the generic GenTL path (they expose more controls),
# and the sources that always "work" come last so they never pre-empt real
# hardware.
BACKENDS = [
    SpinnakerBackend,
    PylonBackend,
    VimbaBackend,
    GenICamBackend,
    PylablibBackend,
    UsbCameraBackend,
    NetworkCameraBackend,
    ScreenCaptureBackend,
    SyntheticBackend,
]

BACKENDS_BY_ID = {b.backend_id: b for b in BACKENDS}


def enumerate_cameras(include=None, exclude=None):
    """Every camera VRHEED can currently connect to, best sources first.

    A backend that raises while enumerating is logged and skipped — one broken
    vendor DLL must never stop the operator reaching the camera that works.

    ``include`` / ``exclude`` are sets of backend ids, for the slow paths:
    probing eight USB indices takes a second or two, so a caller that only
    wants a quick refresh can leave it out.
    """
    # An escape hatch for headless and packaging checks: enumerating opens and
    # closes every local video device, which on a CI runner with no cameras is
    # pure latency, and on a machine whose webcam driver is wedged can block.
    # Also useful to an operator who wants file-analysis mode and nothing else.
    if os.environ.get("VRHEED_NO_CAMERA_SCAN"):
        logger.info("VRHEED_NO_CAMERA_SCAN set - skipping camera discovery")
        return []

    found = []
    for backend in BACKENDS:
        bid = backend.backend_id
        if include is not None and bid not in include:
            continue
        if exclude and bid in exclude:
            continue
        try:
            if not backend.available():
                continue
            cams = backend.enumerate()
        except Exception:
            logger.exception("Backend %s failed to enumerate", bid)
            continue
        found.extend(cams)
    return found


def open_camera(info):
    """Instantiate and open the backend for ``info``.  Raises CameraError."""
    try:
        cam = info.backend_cls(info)
        cam.open()
        return cam
    except CameraError:
        raise
    except Exception as e:
        raise CameraError(str(e)) from e


def find_camera(key, cameras):
    """The CameraInfo in ``cameras`` whose key matches, or None."""
    for info in cameras:
        if info.key == key:
            return info
    return None


def backend_status():
    """(name, available, reason) for every backend — shown in Help > About.

    The reason strings are install instructions, because "your camera is not
    listed" is the first problem a new user has and this is where they look.
    """
    rows = []
    for backend in BACKENDS:
        try:
            ok = backend.available()
        except Exception:
            ok = False
        rows.append((backend.backend_name, ok,
                     "" if ok else backend.unavailable_reason()))
    return rows


def spinnaker_report():
    """Detailed FLIR diagnosis, for when the camera does not appear.

    "No camera found" has several causes that look identical from the app, and
    they need different fixes: the binding is absent, the binding will not
    load, the SDK sees no camera at all, or the SDK sees it but another
    program has it open.  Each line here separates one of those.
    """
    lines = []
    module, error = _probe_import("PySpin")
    if module is None:
        lines.append(f"PySpin import FAILED: {error}")
        lines.append("  -> " + SpinnakerBackend.unavailable_reason())
        return lines
    lines.append("PySpin imported OK")
    for label, attr in (("PySpin version", "__version__"),
                        ("Spinnaker library", "__spinnaker_version__")):
        try:
            lines.append(f"  {label}: {getattr(module, attr)}")
        except Exception:
            pass
    try:
        version = module.System.GetInstance().GetLibraryVersion()
        lines.append(f"  Spinnaker library: {version.major}.{version.minor}."
                     f"{version.type}.{version.build}")
    except Exception as e:
        lines.append(f"  (could not read library version: {e})")
    try:
        system = _spinnaker_system()
    except Exception as e:
        lines.append(f"System.GetInstance() FAILED: {e}")
        return lines
    try:
        cams = system.GetCameras()
        count = cams.GetSize()
        lines.append(f"Cameras reported by Spinnaker: {count}")
        for i in range(count):
            cam = cams[i]
            bits = []
            for feat in ("DeviceVendorName", "DeviceModelName",
                         "DeviceSerialNumber"):
                try:
                    bits.append(str(getattr(cam.TLDevice, feat).GetValue()))
                except Exception as e:
                    bits.append(f"<{feat} unreadable: {e}>")
            lines.append(f"  [{i}] " + "  ".join(bits))
            # Whether anyone else already holds it -- the usual reason a
            # camera is listed here but unusable.
            try:
                free = cam.TLDevice.GevDeviceIsWrongSubnet.GetValue()
                lines.append(f"      GigE wrong subnet: {free}")
            except Exception:
                pass
            del cam
        cams.Clear()
        if count == 0:
            lines.append("  -> Spinnaker itself sees no camera. Open SpinView:")
            lines.append("     if SpinView cannot see it either, this is a "
                         "driver/cable/power problem, not a VRHEED one.")
            lines.append("     USB3: check the camera is on a USB3 port and "
                         "that the FLIR USB driver installed.")
            lines.append("     GigE: check the NIC is on the camera's subnet, "
                         "or run SpinView > Action > Auto Force IP.")
    except Exception as e:
        lines.append(f"GetCameras() FAILED: {e}")
    return lines


def shutdown():
    """Release process-wide driver handles.  Call once, on app exit."""
    global _spin_system
    if _spin_system is not None:
        try:
            _spin_system.ReleaseInstance()
        except Exception:
            logger.exception("Spinnaker ReleaseInstance failed")
        _spin_system = None
    if GenICamBackend._harvester is not None:
        try:
            GenICamBackend._harvester.reset()
        except Exception:
            logger.exception("Harvester reset failed")
        GenICamBackend._harvester = None


if __name__ == "__main__":
    # `python vrheed_cameras.py` prints what this machine can see — the first
    # thing to run when a camera does not show up in the app.  Add --flir for
    # a step-by-step FLIR diagnosis.
    import sys as _sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    print(f"Python {platform.python_version()} ({_sys.executable})")
    print(f"Platform {platform.platform()}\n")
    print("Backends")
    for name, ok, reason in backend_status():
        print(f"  [{'x' if ok else ' '}] {name}")
        if reason:
            for line in reason.splitlines():
                print(f"        {line}")
    print("\nCameras")
    cams = enumerate_cameras()
    if not cams:
        print("  (none)")
    for info in cams:
        print(f"  {info.key:<28} {info.label}"
              + (f"   [{info.detail}]" if info.detail else ""))

    if "--flir" in _sys.argv or not any(c.backend_id == "spinnaker" for c in cams):
        print("\nFLIR / Spinnaker detail")
        for line in spinnaker_report():
            print(f"  {line}")
    shutdown()
