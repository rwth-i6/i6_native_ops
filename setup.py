import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setuptools.setup(
    name="i6 Native Ops",
    ext_modules=[
        CUDAExtension("nativeops", [
            "native_ops/fbw_torch.cpp",
            "native_ops/fbw_op.cu"
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension,
    }
)