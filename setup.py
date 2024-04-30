"""Install the nativeops package with `pip install .` in this dir."""
import os
from setuptools import setup, find_packages


class MissingTorchInstallationError(Exception):
    pass


try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ModuleNotFoundError as exc:
    raise MissingTorchInstallationError(
        "Please install PyTorch before proceeding with this installation."
    ) from exc

TOP_DIR = os.path.dirname(__file__)
FBW_DIR = os.path.join(TOP_DIR, "i6_native_ops/fbw")
WARP_RNNT_DIR = os.path.join(TOP_DIR, "i6_native_ops/warp_rnnt")
FAST_VITERBI_DIR = os.path.join(TOP_DIR, "i6_native_ops/fast_viterbi")
COMMON_DIR = os.path.join(TOP_DIR, "i6_native_ops/common")

setup(
    name="i6 native ops",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="i6_native_ops.fbw.fbw_core",
            sources=[f"{FBW_DIR}/fbw_torch.cpp", f"{FBW_DIR}/fbw_op.cu"],
            include_dirs=[FBW_DIR],
        ),
        CUDAExtension(
            name="i6_native_ops.warp_rnnt.warp_rnnt_core",
            sources=[
                f"{WARP_RNNT_DIR}/binding.cpp",
                f"{WARP_RNNT_DIR}/core_compact.cu",
                f"{WARP_RNNT_DIR}/core.cu",
                f"{WARP_RNNT_DIR}/core_gather.cu",
            ],
            include_dirs=[WARP_RNNT_DIR],
        ),
        CUDAExtension(
            name="i6_native_ops.fast_viterbi.fast_viterbi_core",
            sources=[
                os.path.join(FAST_VITERBI_DIR, "binding.cpp"),
                os.path.join(FAST_VITERBI_DIR, "core.cu"),
            ],
            include_dirs=[FAST_VITERBI_DIR, COMMON_DIR],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    version="0.0.1",
    url="https://github.com/rwth-i6/i6_native_ops",
)
