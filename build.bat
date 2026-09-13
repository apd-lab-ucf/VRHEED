@echo off
REM Build VRHEED.exe.  Run this from an ACTIVATED venv -- PyInstaller must see
REM the same packages the app imports, or it builds an exe that dies on launch
REM with "No module named 'cv2'".
setlocal
set EXE=VRHEED.exe

REM "python -m PyInstaller", never bare "pyinstaller": the bare command may be
REM a global install belonging to a different Python than the active venv.
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyInstaller is not installed in this Python.
    echo Activate the venv and run:  pip install pyinstaller
    echo.
    exit /b 1
)

if exist %EXE% del /f %EXE%

python -m PyInstaller main.spec
if errorlevel 1 (
    echo.
    echo Build failed - see the messages above.
    exit /b 1
)

if not exist dist\%EXE% (
    echo Build failed - exe not found in dist\
    exit /b 1
)

move /y dist\%EXE% .
rmdir /s /q build
rmdir /s /q dist

echo.
echo Build succeeded: %EXE%
echo Launch it once before trusting it - a missing dependency only shows then.
