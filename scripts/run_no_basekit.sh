#!/bin/bash

set -vx

install_conda() {
    conda create -n triton --override-channels -c conda-forge python=$python_version.*
    conda env update -f scripts/triton.yml

    ln -snf /usr/include/level_zero $HOME/miniforge3/envs/triton/x86_64-conda-linux-gnu/sysroot/usr/include/level_zero
    find /usr -name libze_\* -exec ln -sf {} $HOME/miniforge3/envs/triton/lib/ \;

    find /opt/intel/oneapi/mkl/2025.0/lib/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    find /opt/intel/oneapi/compiler/2024.1/lib/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;

    wget https://files.pythonhosted.org/packages/cc/1e/d74e608f0c040e4f72dbfcd3b183f39570f054d08de39cc431f153220d90/intel_sycl_rt-2024.1.2-py2.py3-none-manylinux1_x86_64.whl
    wheel unpack intel_sycl_rt-2024.1.2-py2.py3-none-manylinux1_x86_64.whl
    mkdir -p ./intel_sycl_rt-2024.1.2/intel_sycl_rt-2024.1.2.data/data/include
    cp -r /opt/intel/oneapi/compiler/2024.1/include/sycl ./intel_sycl_rt-2024.*/intel_sycl_rt-2024.*.data/data/include/
    wheel pack intel_sycl_rt-2024.1.2 --build headers_patch
    mv intel_sycl_rt-2024.1.2-headers_patch-py2.py3-none-manylinux1_x86_64.whl intel_sycl_rt-2024.1.2-py2.py3-none-manylinux1_x86_64.whl
    pip install --force-reinstall ./intel_sycl_rt-2024.1.2-py2.py3-none-manylinux1_x86_64.whl
    pip install dpcpp_cpp_rt==2024.1.2
}

script_dir=$(dirname "$0")
source "$script_dir/run_util.sh"

export PATH="$HOME/miniforge3/bin:$PATH"
test -d "$HOME/miniforge3/envs/triton" || install_conda
print_conda_info

python -m venv ./.venv; source ./.venv/bin/activate
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/triton/lib:$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib
export CPATH=$CPATH:$VIRTUAL_ENV/include:$VIRTUAL_ENV/include/sycl

conda run --no-capture-output -n triton bash "$script_name"