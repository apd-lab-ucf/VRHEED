"""Start VRHEED with a Python that can actually talk to the camera.

A lab PC accumulates Python installs, and the camera driver lives in exactly
one of them.  PySpin is not on PyPI: it is a wheel built for a single Python
version, so on a machine with 3.8 / 3.9 / 3.10 / 3.11 installed, only the one
matching the wheel can see the camera.  Meanwhile `python` on PATH is whichever
install came first, which is not usually that one -- so VRHEED starts in
file-analysis mode and the camera appears to have vanished.

Run this instead of main.py and it finds the right interpreter itself:

    py launch.py            (Windows)
    python3 launch.py       (macOS / Linux)
    run_vrheed.bat          (Windows, double-clickable)

Nothing here imports anything outside the standard library, deliberately: it
has to run under whatever Python the user happens to invoke, including one
where VRHEED's own dependencies are missing.
"""

import glob
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")

# Probes run in a candidate interpreter.  Kept to one line each so they can be
# passed with -c and their output is unambiguous.
HAS_CAMERA = "import PySpin, sys; sys.stdout.write(sys.version.split()[0])"
HAS_DEPS = ("import cv2, numpy, PyQt5, pyqtgraph, sys; "
            "sys.stdout.write(sys.version.split()[0])")


def _candidates():
    """Every Python worth asking, current interpreter first."""
    found = [sys.executable]
    if platform.system() == "Windows":
        try:
            listing = subprocess.run(["py", "-0p"], capture_output=True,
                                     text=True, timeout=15).stdout
            for line in listing.splitlines():
                for token in line.split():
                    if token.lower().endswith("python.exe"):
                        found.append(token)
        except Exception:
            pass
        patterns = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\*\python.exe"),
            r"C:\Python*\python.exe",
            os.path.expandvars(r"%ProgramFiles%\Python*\python.exe"),
        ]
    else:
        patterns = ["/usr/bin/python3.*", "/usr/local/bin/python3.*",
                    "/opt/homebrew/bin/python3.*"]
    for pattern in patterns:
        try:
            found.extend(glob.glob(pattern))
        except Exception:
            continue

    unique, seen = [], set()
    for exe in found:
        key = os.path.normcase(os.path.abspath(exe))
        if key not in seen and os.path.isfile(exe):
            seen.add(key)
            unique.append(exe)
    return unique[:10]


def _probe(exe, code):
    """Version string if ``code`` runs cleanly in ``exe``, else ''."""
    try:
        result = subprocess.run([exe, "-c", code], capture_output=True,
                                text=True, timeout=60)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _say(text):
    """Print and flush.

    Without the flush these lines sit in the buffer until this process exits,
    which is AFTER the app it launched has finished writing -- so the
    explanation arrives after the thing it was explaining.
    """
    print(text)
    sys.stdout.flush()


def _best_interpreters():
    """(with_camera, with_deps_only), each a list of (exe, version)."""
    with_camera, with_deps = [], []
    for exe in _candidates():
        version = _probe(exe, HAS_CAMERA)
        if version:
            with_camera.append((exe, version))
        elif _probe(exe, HAS_DEPS):
            with_deps.append((exe, ""))
    return with_camera, with_deps


def _print_python():
    """Print the interpreter VRHEED should be run or built with, and nothing
    else, so a build script can capture it:

        for /f %%i in ('py launch.py --print-python') do set PYEXE=%%i
    """
    with_camera, with_deps = _best_interpreters()
    for exe, _ in with_camera:
        if _probe(exe, HAS_DEPS):
            print(exe)
            return 0
    if with_deps:
        print(with_deps[0][0])
        return 0
    print(sys.executable)
    return 0


def main(argv):
    # Answer this before anything else is printed: the caller is capturing
    # stdout and any other line would be swallowed into the variable.
    if "--print-python" in argv:
        return _print_python()

    if not os.path.isfile(MAIN):
        _say(f"Cannot find main.py next to this script ({HERE}).")
        return 2

    with_camera, with_deps = _best_interpreters()

    # Best case: one interpreter has the camera driver.  Prefer it even if it
    # is not the one that ran this script.
    for exe, version in with_camera:
        if _probe(exe, HAS_DEPS):
            _say(f"VRHEED: using Python {version} ({exe}) - camera driver present")
            return subprocess.call([exe, MAIN] + argv)

    # The driver is there but that interpreter is missing VRHEED's own
    # dependencies.  Say exactly how to fix it rather than starting without
    # the camera and letting it look broken.
    if with_camera:
        exe, version = with_camera[0]
        _say(f"VRHEED: Python {version} ({exe}) has the camera driver but is")
        _say("        missing VRHEED's dependencies. Install them into it:")
        _say(f'            "{exe}" -m pip install -r requirements.txt')
        _say("        Then run this launcher again.")
        return 1

    # No camera driver anywhere: still useful for analysing recordings, but
    # be explicit that this is why there is no camera.
    fallback = None
    for exe, _ in with_deps:
        fallback = exe
        break
    fallback = fallback or sys.executable
    _say("VRHEED: no Python on this machine can import PySpin, so the FLIR")
    _say("        camera will not appear. Starting in file-analysis mode.")
    _say("        Run 'vrheed_cameras.py --flir' for the full diagnosis.")
    return subprocess.call([fallback, MAIN] + argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
