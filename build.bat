@echo off
set EXE=VRHEED.exe

if exist %EXE% del /f %EXE%

pyinstaller main.spec

if not exist dist\%EXE% (
    echo Build failed - exe not found in dist\
    exit /b 1
)

move /y dist\%EXE% .
rmdir /s /q build
rmdir /s /q dist
echo Build succeeded: %EXE%
