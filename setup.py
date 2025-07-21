# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# 定义Cython扩展模块
# 我们需要包含Numpy的头文件路径
ext_modules = [
    Extension(
        "Game_cython",                          # 生成的模块名
        ["Game_cython.pyx"],                    # 源文件
        include_dirs=[numpy.get_include()],     # 包含Numpy头文件
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="Cython Dark Chess Environment",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"}, # 指定Python 3
        annotate=True                                # 生成一个HTML报告，显示Cython代码转换情况
    ),
)