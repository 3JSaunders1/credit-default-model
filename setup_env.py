"""
setup_env.py
Creates a virtual environment and installs requirements.
Safe to run repeatedly: skips steps that are already done.

Usage: python setup_env.py
"""
import os
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path("venv")
REQUIREMENTS = Path("requirements.txt")
MIN_PYTHON = (3, 10)


def check_python_version():
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
                 f"You have {sys.version.split()[0]}.")
    print(f"Python {sys.version.split()[0]} — OK")


def venv_python() -> Path:
    # Windows puts executables in Scripts/, Mac and Linux use bin/
    folder = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return VENV_DIR / folder / exe


def create_venv():
    if venv_python().exists():
        print("Virtual environment already exists — skipping.")
    else:
        print("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)


def install_requirements():
    if not REQUIREMENTS.exists() or not REQUIREMENTS.read_text().strip():
        sys.exit("requirements.txt is missing or empty. Add packages first.")
    py = str(venv_python())
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    print("All packages installed.")


if __name__ == "__main__":
    check_python_version()
    create_venv()
    install_requirements()
    activate = r"venv\Scripts\activate" if os.name == "nt" else "source venv/bin/activate"
    print(f"\nDone. Activate it with:\n  {activate}")