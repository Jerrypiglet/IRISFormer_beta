from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pdist',
    ext_modules=[
        CUDAExtension('pdist_cuda', [
            'pair_wise_distance_cuda_source.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
