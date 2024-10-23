import subprocess
import os
import pathlib

if os.name == "nt":
    from .winutils import set_env_vars
else:
    from .linuxutils import set_env_vars

__all__ = ["checkout_git", "set_env_vars"]

def checkout_git(repo_url: str, target_dir: pathlib.Path, ref: str = "HEAD", path_to_git: str = "git"):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Change to the target directory
    os.chdir(target_dir)
    
    # Checkout the specified hash
    subprocess.run([path_to_git, "clone", "-n", repo_url], check=True)
    repo_dir = repo_url.split("/")[-1]
    if repo_dir.endswith(".git"):
        repo_dir = repo_dir[:-4]
    os.chdir(repo_dir)
    subprocess.run([path_to_git, "checkout", "--recurse-submodules", ref], check=True)
    return repo_dir
