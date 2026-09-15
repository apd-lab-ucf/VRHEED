@echo off
REM Double-click me.  Picks the Python that has the FLIR camera driver, which
REM is not usually the one on PATH -- see launch.py for why.
setlocal
cd /d "%~dp0"
py launch.py %*
if errorlevel 1 pause
