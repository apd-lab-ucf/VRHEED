# -*- mode: python ; coding: utf-8 -*-

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
        'PySpin',
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
