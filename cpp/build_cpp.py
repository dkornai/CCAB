"""
Check the requirements of the C++ backend (module pf_cpp, sources in this folder), then build it into the project root.

Usage (with the Python environment of the project activated, from any directory):
    python cpp/build_cpp.py            check the requirements, then build
    python cpp/build_cpp.py --check    only check the requirements
    python cpp/build_cpp.py --clean    remove the previous build first (e.g. after changing compiler or generator)
"""
import argparse
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent  # project root, where the module is placed
BUILD_DIR = SOURCE_DIR / "build"

MIN_PYTHON = (3, 10)
MIN_CMAKE = (3, 18)
PIP = f'"{sys.executable}" -m pip install'

COMPILER_HINTS = {
    "Linux":   "install a C++20 compiler (GCC >= 10 or Clang >= 12), e.g. `sudo apt install build-essential`",
    "Darwin":  "install the Xcode command line tools: `xcode-select --install`",
    "Windows": "install the Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload",
}


def find_tool(name: str) -> str | None:
    """
    Path of a command line tool, on the PATH or next to the Python executable (where `pip install cmake ninja` puts them)
    """
    exe = name + (".exe" if platform.system() == "Windows" else "")
    next_to_python = Path(sys.executable).parent / exe
    return shutil.which(name) or (str(next_to_python) if next_to_python.exists() else None)


def check_requirements() -> tuple[bool, str | None, str | None]:
    """
    Check the requirements that can be checked before configuring the build. Returns (all ok, cmake path, ninja path).
    """
    ok = True

    def report(passed: bool, what: str, fix: str = ""):
        nonlocal ok
        ok &= passed
        print(f"  [{'ok' if passed else 'missing'}] {what}" + ("" if passed else f"\n            -> {fix}"))

    print("Checking requirements:")
    report(sys.version_info >= MIN_PYTHON, f"Python >= {'.'.join(map(str, MIN_PYTHON))} (found {platform.python_version()})",
           "use a newer Python")

    for package in ("numpy", "nanobind"):
        report(importlib.util.find_spec(package) is not None, f"Python package {package}", f"{PIP} {package}")

    cmake = find_tool("cmake")
    version = None
    if cmake:
        output = subprocess.run([cmake, "--version"], capture_output=True, text=True).stdout
        match = re.search(r"(\d+)\.(\d+)", output)
        version = (int(match.group(1)), int(match.group(2))) if match else None
    report(version is not None and version >= MIN_CMAKE,
           f"CMake >= {'.'.join(map(str, MIN_CMAKE))}" + (f" (found {version[0]}.{version[1]})" if version else ""),
           f"{PIP} cmake")

    # Ninja is optional (faster builds). On Windows, CMake's default generator (Visual Studio) finds the compiler by itself, Ninja does not.
    ninja = find_tool("ninja") if platform.system() != "Windows" else None
    print(f"  [{'ok' if ninja else '--'}] Ninja (optional" + ("" if ninja else f", {PIP} ninja") + ")")

    print("  [--] C++20 compiler and Eigen 3.4 are found by CMake below (Eigen is downloaded if it is not installed)")
    return ok, cmake, ninja


def run(command: list[str], hint: str) -> bool:
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in command), flush=True)
    if subprocess.run(command).returncode != 0:
        print(f"\nFailed. {hint}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check the requirements")
    parser.add_argument("--clean", action="store_true", help="remove the previous build first")
    args = parser.parse_args()

    ok, cmake, ninja = check_requirements()
    if not ok:
        sys.exit("\nInstall the missing requirements above, then run this script again.")
    if args.check:
        return

    if args.clean and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    configure = [cmake, "-S", str(SOURCE_DIR), "-B", str(BUILD_DIR), f"-DPython_EXECUTABLE={sys.executable}"]
    if ninja and not (BUILD_DIR / "CMakeCache.txt").exists():
        configure += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    compiler_hint = COMPILER_HINTS.get(platform.system(), "install a C++20 compiler")
    if not run(configure, f"If no C++ compiler was found, {compiler_hint}.\n"
                          "If Eigen could not be downloaded, install it (e.g. `apt install libeigen3-dev`, `brew install eigen`, "
                          "`conda install -c conda-forge eigen`).\n"
                          "If CMake reports a generator or cache mismatch, run with --clean."):
        sys.exit(1)

    if not run([cmake, "--build", str(BUILD_DIR), "--config", "Release", "--parallel"], "See the compiler errors above."):
        sys.exit(1)

    # Check that the module imports from the project root, as the Python code does
    check = subprocess.run([sys.executable, "-c", "import pf_cpp; print(pf_cpp.__file__)"], cwd=ROOT, capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"\nThe module was built, but does not import:\n{check.stderr}")
    print(f"\nBuilt {Path(check.stdout.strip()).name}. The C++ backend is ready: use backend=\"cpp\" in run_replicate / run_replicates.")


if __name__ == "__main__":
    main()
