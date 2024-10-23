import os
import subprocess

def set_env_vars(command: list, shell_script=False):
    if shell_script:
        command = ["source"] + command
    command += ["&&", "env"]
    output = subprocess.check_output(command, shell=True, text=True)

    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
