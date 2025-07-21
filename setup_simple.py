# setup_simple.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# 定义Cython扩展模块
ext_modules = [
    Extension(
        "banqi_gym.Game_cython_simple",         # 生成的模块名 (包含包名)
        ["banqi_gym/Game_cython_simple.pyx"],   # 源文件路径
        include_dirs=[numpy.get_include()],     # 包含Numpy头文件
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="Simplified Cython Dark Chess Environment",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"}, # 指定Python 3
        annotate=True                                # 生成HTML报告
    ),
)
