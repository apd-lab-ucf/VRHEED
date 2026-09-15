@echo off
REM Build VRHEED.exe.
REM
REM The interpreter matters more than anything else here.  PyInstaller bundles
REM what the Python RUNNING IT can import, so building with the wrong one of
REM several installed Pythons produces an exe with no FLIR camera support --
REM and that failure is silent: the build succeeds and the exe runs, it just
REM never sees the camera.  So the interpreter is chosen the same way the app
REM chooses it, by asking which one can import PySpin.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set EXE=VRHEED.exe

REM build.bat -y  builds without asking anything, for scripted use.
set ASSUME_YES=
if /i "%~1"=="-y" set ASSUME_YES=1
if /i "%~1"=="/y" set ASSUME_YES=1

for /f "usebackq delims=" %%i in (`py launch.py --print-python 2^>nul`) do set PYEXE=%%i
if not defined PYEXE (
    echo Could not work out which Python to build with.
    echo Falling back to "python" on PATH.
    set PYEXE=python
)
echo Building with: !PYEXE!

REM Does that interpreter have the camera driver?  If not the exe will be
REM file-analysis only, which is a legitimate build but never what someone
REM wants by accident.
"!PYEXE!" -c "import PySpin" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: this Python cannot import PySpin, so the exe will have NO
    echo          FLIR camera support.  It will still analyse recorded video.
    echo.
    echo          If that is not what you want, install the PySpin wheel
    echo          matching this Python and run build.bat again.
    echo.
    if not defined ASSUME_YES (
        choice /c YN /m "Build anyway"
        if errorlevel 2 exit /b 1
    )
) else (
    echo FLIR camera support: yes
)

"!PYEXE!" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyInstaller is not installed in that Python. Install it there:
    echo     "!PYEXE!" -m pip install pyinstaller
    exit /b 1
)

REM Build BEFORE touching the existing exe: a failed build must not leave you
REM with no working binary at all, which is what deleting it up front did.
"!PYEXE!" -m PyInstaller --noconfirm main.spec
if errorlevel 1 (
    echo.
    echo Build failed - see the messages above. %EXE% is untouched.
    exit /b 1
)
if not exist "dist\%EXE%" (
    echo Build reported success but dist\%EXE% is missing.
    exit /b 1
)

REM Prove the thing actually starts.  A missing dependency does not fail the
REM build, only the launch, and this is the failure people hit.  The exe is
REM GUI-subsystem so cmd does not wait for it: start /wait does, and sets
REM errorlevel from its exit code.
echo Checking the exe starts...
start /wait "" "dist\%EXE%" --check
if errorlevel 1 (
    echo.
    echo The exe was built but fails to start. Not replacing %EXE%.
    echo Run it from a console to see why:  dist\%EXE% --check
    exit /b 1
)

move /y "dist\%EXE%" "%EXE%" >nul
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo.
echo Build succeeded: %EXE%
