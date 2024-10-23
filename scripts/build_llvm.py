import subprocess
import os
import argparse
import pathlib
import utils

TRITON_ENV_NAME = "triton"

proxy_settings = {
    "http_proxy": "http://fmproxyslb.ice.intel.com:911",
    "https_proxy": "http://fmproxyslb.ice.intel.com:911"
}

LLVM_REPO = "https://github.com/llvm/llvm-project.git"


def build_llvm(target_dir: pathlib.Path, ref: str, path_to_git: str):
    abs_target_dir = os.path.abspath(target_dir)
    repo_subdir = utils.checkout_git(LLVM_REPO, target_dir, ref, path_to_git)
    if os.name == "nt":
        from utils.winutils import find_visual_studio, set_vs_env_vars
        vs_path = find_visual_studio(["[17.0,18.0)", "[16.0,17.0)"])
        set_vs_env_vars(vs_path)
        print(vs_path)
        if not vs_path:
            raise EnvironmentError("Visual Studio 2019 or 2022 not found.")
        os.environ["LIB"] = os.environ["LIB"] + os.pathsep + os.path.join(os.environ["CONDA_PREFIX"], "Library", "lib")
        os.environ["INCLUDE"] = os.environ["INCLUDE"] + os.pathsep + os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
    os.chdir(os.path.join(abs_target_dir, repo_subdir))
    os.mkdir("build")
    os.chdir("build")
    target_install_dir = os.path.join(abs_target_dir, "llvm")
    os.environ["LLVM_INSTALL_DIR"] = target_install_dir
    subprocess.run(["cmake",
                    "-G", "Ninja",
                    os.path.join("..", "llvm"),
                    "-DLLVM_ENABLE_DUMP=1",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DLLVM_ENABLE_ASSERTIONS=true",
                    "-DLLVM_ENABLE_PROJECTS=mlir",
                    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU",
                    "-DLLVM_INSTALL_UTILS=true",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    f"-DCMAKE_INSTALL_PREFIX={target_install_dir}"], check=True)
    subprocess.run(["cmake", "--build", ".", "--target", "install"], check=True)


def conda_activate_env(env_name: str, path_to_conda_activate: str = "activate"):
    command = [path_to_conda_activate, env_name]
    utils.set_env_vars(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", "-d", type=str, default="../packages", help="Target root directory where to checkout LLVM repository")
    parser.add_argument("--ref", "-r", type=str, required=True, help="LLVM git reference to checkout")
    parser.add_argument("--path-to-git", "-g", type=str, default="git", help="Path to git executable")
    parser.add_argument("--path-to-conda-activate", "-c", type=str, default="activate", help="Path to conda activate executable")
    args = parser.parse_args()

    conda_activate_env(TRITON_ENV_NAME, args.path_to_conda_activate)
    os.environ.update(proxy_settings)
    build_llvm(pathlib.Path(args.target_dir), args.ref, args.path_to_git)


if __name__ == '__main__':
    main()
