# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for VRHEED.app (macOS).  The Windows exe comes from main.spec.

Build with:  pyinstaller main_mac.spec          -> dist/VRHEED.app
Check it:    QT_QPA_PLATFORM=offscreen "dist/VRHEED.app/Contents/MacOS/VRHEED" --check

Same Analysis as main.spec; the differences are all macOS ones:
  * one-DIR + COLLECT + BUNDLE, because macOS expects Contents/MacOS/...
    (main.spec is one-file, which is fine for a bare .exe but not for a .app)
  * .icns instead of .ico
  * PySpin is left out of hiddenimports: the FLIR Spinnaker SDK has no macOS
    Python wheel, so a mac build is always file-analysis mode (load recorded
    videos / traces, no live camera) and listing it only adds a build warning.
"""

import os
import re

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
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'pyqtgraph.graphicsItems.PlotDataItem',
        'pyqtgraph.graphicsItems.LinearRegionItem',
        'pyqtgraph.graphicsItems.PlotItem',
    ],
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
    },
)
