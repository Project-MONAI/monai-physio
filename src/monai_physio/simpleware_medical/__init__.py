"""Simpleware Medical integration for MONAI Physio.

This package contains Python scripts and utilities for integrating
Synopsys Simpleware Medical with the MONAI Physio framework.

Modules:
    SimplewareScript_heart_segmentation: Script for ASCardio heart segmentation
    SimplewareScript_tetrahedral_mesh: Script for +FE Free tetrahedral meshing
"""

import os
import subprocess
import sys

__version__ = "0.1.0"

#: Default path to the Simpleware Medical console executable.
DEFAULT_SIMPLEWARE_EXE_PATH = (
    "C:/Program Files/Synopsys/Simpleware Medical/Y-2026.03/"
    "ConsoleSimplewareMedical.exe"
)


def run_simpleware_script(
    exe_path: str,
    input_file: str,
    input_value: str,
    script_path: str,
    timeout_s: int = 600,
) -> tuple[str, str]:
    """Run a Python script inside Simpleware Medical's console application.

    Args:
        exe_path: Path to ``ConsoleSimplewareMedical.exe``.
        input_file: File Simpleware opens as the active document.
        input_value: String the script reads with ``app.GetInputValue()``.
        script_path: Script to run inside Simpleware.
        timeout_s: Seconds before the process tree is killed.

    Returns:
        The process's (stdout, stderr).

    Raises:
        FileNotFoundError: If *exe_path* or *script_path* does not exist.
        RuntimeError: If the process times out or exits non-zero.
    """
    if not os.path.exists(exe_path):
        raise FileNotFoundError(
            f"Simpleware Medical executable not found at: {exe_path}"
        )
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Simpleware script not found at: {script_path}")

    cmd = [
        exe_path,
        "--input-file",  # Use only with ConsoleSimplewareMedical.exe
        input_file,
        "--input-value",
        input_value,
        "--run-script",
        script_path,
        "--exit-after-script",
        "--no-progress",  # Use only with ConsoleSimplewareMedical.exe
        # "--no-splash",  # Use only with SimplewareMedical.exe
    ]

    # When the process exits, the OS frees all of its resources (GPU,
    # memory); no extra cleanup is required. Popen lets the process tree be
    # killed on timeout so no child process keeps holding the GPU.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(sys.platform != "win32"),  # process group on Unix
    )
    try:
        stdout, stderr = proc.communicate(input="y\n", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), 9)
        proc.wait()
        raise RuntimeError(
            f"Simpleware Medical script {os.path.basename(script_path)} timed "
            f"out after {timeout_s} seconds"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Simpleware Medical script {os.path.basename(script_path)} failed "
            f"with return code {proc.returncode}:\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )
    return stdout, stderr
