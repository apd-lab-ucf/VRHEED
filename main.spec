# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import sys

# ---------------------------------------------------------------------------
# Refuse to build an exe that cannot start
# ---------------------------------------------------------------------------
# PyInstaller only WARNS about a hidden import it cannot find, then builds
# anyway.  The result looks like a successful build and dies on launch with
# "No module named 'cv2'" -- which is exactly what happens when pyinstaller
# on PATH belongs to a different Python than the one requirements.txt was
# installed into.  Fail here instead, while the fix is still obvious.
REQUIRED = ['cv2', 'numpy', 'scipy', 'PyQt5', 'pyqtgraph']
_missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if _missing:
    raise SystemExit(
        "\nBuild aborted: this Python cannot import " + ", ".join(_missing) + "\n\n"
        "  Building with: " + sys.executable + "\n\n"
        "PyInstaller must run in the SAME environment as the dependencies.\n"
        "Activate the venv first, then build through it:\n\n"
        "    .venv\\Scripts\\activate\n"
        "    pip install -r requirements.txt pyinstaller\n"
        "    python -m PyInstaller main.spec\n")

# Camera drivers are imported lazily inside vrheed_cameras, so PyInstaller
# cannot see them by following imports.  Every one is optional: whichever are
# installed on the build machine get bundled, the rest are left out and the
# guarded imports in vrheed_cameras handle their absence at runtime.
CAMERA_DRIVERS = [
    'PySpin',                          # FLIR / Point Grey
    'pypylon', 'pypylon.pylon',        # Basler
    'vmbpy',                           # Allied Vision
    'harvesters', 'harvesters.core',   # any GenICam camera
    'pylablib',                        # Andor / Hamamatsu / PI / NI IMAQ / ...
    'mss',                             # screen capture
]
camera_hiddenimports = [
    m for m in CAMERA_DRIVERS
    if importlib.util.find_spec(m.split('.')[0]) is not None
]
print("VRHEED: building with " + sys.executable)
print("VRHEED: camera drivers bundled: "
      + (", ".join(sorted({m.split('.')[0] for m in camera_hiddenimports}))
         or "NONE - the exe will have no camera support beyond USB/network"))

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
    a.binaries,
    a.datas,
    [],
    name='VRHEED',
    icon='turtle.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
