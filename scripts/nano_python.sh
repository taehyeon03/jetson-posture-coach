#!/usr/bin/env bash
# System Python + the CUDA library path required by the minimal Nano install.
# Does not modify the system linker configuration.
set -euo pipefail
posture_cuda_lib=/usr/local/cuda-10.2/targets/aarch64-linux/lib
if [[ ! -d "$posture_cuda_lib" ]]; then
    echo "CUDA 10.2 libraries not found: $posture_cuda_lib" >&2
    exit 2
fi
export LD_LIBRARY_PATH="$posture_cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /usr/bin/python3 "$@"
