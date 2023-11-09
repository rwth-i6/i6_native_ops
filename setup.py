"""Install the nativeops package with `pip install .` in this dir."""
from setuptools import setup


class MissingTorchInstallationError(Exception):
    pass


try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ModuleNotFoundError as exc:
    raise MissingTorchInstallationError(
        "Please install PyTorch before proceeding with this installation."
    ) from exc

setup(
    name="i6 Native Ops",
    ext_modules=[
        CUDAExtension("nativeops", ["native_ops/fbw_torch.cpp", "native_ops/fbw_op.cu"])
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    version="0.0.1",
    url="https://github.com/rwth-i6/i6_native_ops",
)
