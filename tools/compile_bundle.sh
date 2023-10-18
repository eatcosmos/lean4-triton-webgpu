#!/usr/bin/env bash
#
# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html
#
set -ueo pipefail

VER_PYTORCH="v2.0.1"
VER_TORCHVISION="v0.15.2"
VER_TORCHAUDIO="v2.0.2"
VER_IPEX="xpu-master"
VER_GCC=11

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <DPCPPROOT> <MKLROOT> [AOT]"
    echo "DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively."
    echo "AOT is optional, should be the text string for environment variable USE_AOT_DEVLIST."
    exit 1
fi
DPCPP_ROOT=$1
ONEMKL_ROOT=$2
AOT=""
if [[ $# -ge 3 ]]; then
    AOT=$3
fi

# Check existance of DPCPP and ONEMKL environments
DPCPP_ENV=${DPCPP_ROOT}/env/vars.sh
if [ ! -f ${DPCPP_ENV} ]; then
    echo "DPC++ compiler environment ${DPCPP_ENV} doesn't seem to exist."
    exit 2
fi

ONEMKL_ENV=${ONEMKL_ROOT}/env/vars.sh
if [ ! -f ${ONEMKL_ENV} ]; then
    echo "oneMKL environment ${ONEMKL_ENV} doesn't seem to exist."
    exit 3
fi

# Check existance of required Linux commands
for APP in python git patch pkg-config nproc bzip2 gcc g++; do
    command -v $APP > /dev/null || (echo "Error: Command \"${APP}\" not found." ; exit 4)
done

# Check existance of required libs
for LIB_NAME in zlib libpng libjpeg; do
    pkg-config --exists $LIB_NAME || (echo "Error: \"${LIB_NAME}\" not found in pkg-config." ; exit 5)
done

if [ $(gcc -dumpversion) -ne $VER_GCC ]; then
    echo -e '\a'
    echo "Warning: GCC version ${VER_GCC} is recommended"
    echo "Found GCC version $(gcc -dumpfullversion)"
    sleep 5
fi

# set number of compile processes, if not already defined
if [ -z "${MAX_JOBS-}" ]; then
    export MAX_JOBS=$(nproc)
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}

# Be verbose now
set -x

# Checkout individual components
if [ ! -d triton ]; then
    git clone https://github.com/openai/triton
fi
if [ ! -d pytorch ]; then
    git clone https://github.com/pytorch/pytorch.git
fi
if [ ! -d vision ]; then
    git clone https://github.com/pytorch/vision.git
fi
if [ ! -d audio ]; then
    git clone https://github.com/pytorch/audio.git
fi
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git
fi

# Checkout required branch/commit and update submodules
cd triton
git checkout main
git pull
git submodule sync
git submodule update --init --recursive
cd third_party/intel_xpu_backend
git checkout main
git pull
cd ../..
git checkout `cat third_party/intel_xpu_backend/triton_hash.txt`
git submodule sync
git submodule update --init --recursive
cd third_party/intel_xpu_backend
git checkout main
git submodule sync
git submodule update --init --recursive
cd ../../..
cd pytorch
git stash
git clean -fd
if [ ! -z ${VER_PYTORCH} ]; then
    git checkout main
    git pull
    git checkout ${VER_PYTORCH}
fi
git submodule sync
git submodule update --init --recursive
cd ..
cd vision
if [ ! -z ${VER_TORCHVISION} ]; then
    git checkout main
    git pull
    git checkout ${VER_TORCHVISION}
fi
git submodule sync
git submodule update --init --recursive
cd ..
cd audio
if [ ! -z ${VER_TORCHAUDIO} ]; then
    git checkout main
    git pull
    git checkout ${VER_TORCHAUDIO}
fi
git submodule sync
git submodule update --init --recursive
cd ..
cd intel-extension-for-pytorch
if [ ! -z ${VER_IPEX} ]; then
    git checkout master
    git pull
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive
cd ..

# Install python dependency
python -m pip install cmake ninja mkl-static mkl-include Pillow pybind11

# Compile individual component
#  Triton
cd triton/python
python setup.py clean
TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py bdist_wheel 2>&1 | tee build.log
python -m pip install --force-reinstall dist/*.whl
cd ../..
#  PyTorch
cd pytorch
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
# Ensure cmake can find python packages when using conda or virtualenv
python -m pip install -r requirements.txt
if [ -n "${CONDA_PREFIX-}" ]; then
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(command -v conda))/../"}
elif [ -n "${VIRTUAL_ENV-}" ]; then
    export CMAKE_PREFIX_PATH=${VIRTUAL_ENV:-"$(dirname $(command -v python))/../"}
fi
export USE_STATIC_MKL=1
export _GLIBCXX_USE_CXX11_ABI=1
export USE_NUMA=0
export USE_CUDA=0
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
unset USE_CUDA
unset USE_NUMA
unset _GLIBCXX_USE_CXX11_ABI
unset USE_STATIC_MKL
unset CMAKE_PREFIX_PATH
python -m pip install --force-reinstall dist/*.whl
cd ..
#  TorchVision
cd vision
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
python -m pip install --force-reinstall --no-deps dist/*.whl
cd ..
# don't fail on external scripts
set +uex
source ${DPCPP_ENV}
source ${ONEMKL_ENV}
set -uex
#  TorchAudio
cd audio
python -m pip install -r requirements.txt
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
python -m pip install --force-reinstall --no-deps dist/*.whl
cd ..
#  Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
if [[ ! ${AOT} == "" ]]; then
    export USE_AOT_DEVLIST=${AOT}
fi
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
if [[ ! ${AOT} == "" ]]; then
    unset USE_AOT_DEVLIST
fi
python -m pip install --force-reinstall dist/*.whl
python -m pip uninstall -y mkl-static mkl-include

# Sanity Test
cd ..
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; import triton; print(f'torch_cxx11_abi:     {torch.compiled_with_cxx11_abi()}'); print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}'); print(f'triton_version:      {triton.__version__}');"