# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for VRHEED.app (macOS).  The Windows exe comes from main.spec.

Build with:  pyinstaller main_mac.spec          -> dist/VRHEED.app
Check it:    QT_QPA_PLATFORM=offscreen "dist/VRHEED.app/Contents/MacOS/VRHEED" --check

Same Analysis as main.spec; the differences are all macOS ones:
  * one-DIR + COLLECT + BUNDLE, because macOS expects Contents/MacOS/...
    (main.spec is one-file, which is fine for a bare .exe but not for a .app)
  * .icns instead of .ico
  * PySpin is left out of hiddenimports: the FLIR Spinnaker SDK has no macOS
    Python wheel, so a mac build never has the FLIR backend.  It is NOT
    file-analysis only, though -- USB cameras, network streams, screen
    capture and the simulated source all work on macOS (2.2).
"""

import importlib.util
import os
import re
import sys

# Refuse to build an app that cannot start.  PyInstaller only WARNS about a
# hidden import it cannot find and then builds anyway, producing a bundle that
# dies on launch; see the same guard in main.spec.
REQUIRED = ['cv2', 'numpy', 'scipy', 'PyQt5', 'pyqtgraph']
_absent = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if _absent:
    raise SystemExit(
        "\nBuild aborted: this Python cannot import " + ", ".join(_absent) + "\n\n"
        "  Building with: " + sys.executable + "\n\n"
        "PyInstaller must run in the same environment as the dependencies:\n"
        "    pip install -r requirements.txt pyinstaller\n"
        "    python -m PyInstaller main_mac.spec\n")

# Camera drivers are imported lazily inside vrheed_cameras, so PyInstaller
# cannot follow them.  All optional, all bundled only if present.  PySpin is
# absent from this list on purpose (no macOS wheel exists).
CAMERA_DRIVERS = [
    'pypylon', 'pypylon.pylon',        # Basler
    'vmbpy',                           # Allied Vision
    'harvesters', 'harvesters.core',   # any GenICam camera
    'pylablib',                        # Andor / Hamamatsu / PI / ...
    'mss',                             # screen capture
]
camera_hiddenimports = [
    m for m in CAMERA_DRIVERS
    if importlib.util.find_spec(m.split('.')[0]) is not None
]

_HERE = globals().get('SPECPATH') or os.getcwd()


def _read_version(path=os.path.join(_HERE, 'main.py')):
    """Read main.__version__ WITHOUT importing it (importing would pull in PyQt5)."""
    try:
        with open(path, encoding='utf-8') as f:
            m = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', f.read(), re.M)
        return m.group(1) if m else '0.0.0'
    except OSError:
        return '0.0.0'


APP_VERSION = _read_version()
print(f"BUILD: VRHEED version {APP_VERSION}")
print("BUILD: camera drivers bundled: "
      + (", ".join(sorted({m.split('.')[0] for m in camera_hiddenimports}))
         or "none beyond USB/network/screen"))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('turtle.ico', '.')],
    hiddenimports=[
        # Crash log (2.1): RotatingFileHandler lives in a submodule that is
        # imported at the top of main.py; listed here so a future refactor
        # that moves the import behind a function cannot silently drop it.
        'logging.handlers',
        'cv2',
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.signal._savitzky_golay',
        'scipy.signal._peak_finding',
        # Gaussian profile fitting goes through least_squares/MINPACK, which
        # PyInstaller does not pick up from the curve_fit import alone.
        'scipy.optimize',
        'scipy.optimize._minpack',
        'scipy.optimize._minpack_py',
        'scipy.optimize._lsq',
        'scipy.optimize._lsq.least_squares',
        'vrheed_analysis',
        'vrheed_cameras',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'pyqtgraph.graphicsItems.PlotDataItem',
        'pyqtgraph.graphicsItems.LinearRegionItem',
        'pyqtgraph.graphicsItems.PlotItem',
    ] + camera_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VRHEED',
    icon='turtle.icns',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    # target_arch=None builds for the ARCHITECTURE OF THE PYTHON RUNNING THIS
    # SPEC — arm64 on an Apple-silicon Mac, which will not open on an Intel
    # Mac. PyQt5 ships no universal2 wheel, so build once per arch instead
    # (see the matrix in .github/workflows/build.yml).
    # codesign_identity=None leaves the bundle ad-hoc signed: recipients must
    # right-click -> Open the first time.
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VRHEED',
)

app = BUNDLE(
    coll,
    name='VRHEED.app',
    icon='turtle.icns',
    bundle_identifier='edu.ucf.apd.vrheed',
    version=APP_VERSION,
    info_plist={
        'CFBundleName': 'VRHEED',
        'CFBundleDisplayName': 'VRHEED',
        'CFBundleVersion': APP_VERSION,
        'CFBundleShortVersionString': APP_VERSION,
        'NSHighResolutionCapable': True,
        # The app writes vrheed.log and session files next to itself.
        'NSCameraUsageDescription': 'VRHEED reads frames from a connected RHEED camera.',
        # Screen capture (2.2) digitises another program's live-image pane.
        'NSScreenCaptureUsageDescription':
            'VRHEED can read a RHEED pattern from another program on screen.',
    },
)
