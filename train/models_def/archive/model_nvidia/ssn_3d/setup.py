
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pdist3dgmm',
    ext_modules=[
        CUDAExtension('pdist3dgmm', [
            'pair_wise_dist_3dgmm.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
