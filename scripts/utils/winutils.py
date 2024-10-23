import os
import subprocess
from pathlib import Path

def find_vswhere():
    program_files = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
    vswhere_path = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere_path.exists():
        return vswhere_path
    return None


def find_visual_studio(version_ranges):
    vswhere = find_vswhere()
    if not vswhere:
        raise FileNotFoundError("vswhere.exe not found.")

    for version_range in version_ranges:
        command = [
            str(vswhere), "-version", version_range, "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath", "-prerelease"
        ]

        try:
            output = subprocess.check_output(command, text=True).strip()
            if output:
                return output
        except subprocess.CalledProcessError:
            continue

    return None


def set_env_vars(command: list, shell_script=False):
    if shell_script:
        command = ["call"] + command
    command += ["&&", "set"]
    output = subprocess.check_output(command, shell=True, text=True)

    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value


def set_vs_env_vars(vs_path, arch="x64"):
    vcvarsall_path = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall_path.exists():
        raise FileNotFoundError(f"vcvarsall.bat not found in expected path: {vcvarsall_path}")

    command = [vcvarsall_path, arch]
    set_env_vars(command, shell_script=True)
