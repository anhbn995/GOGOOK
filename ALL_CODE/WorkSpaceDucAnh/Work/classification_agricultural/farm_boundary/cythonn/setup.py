from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("/home/quyet/DATA_ML/WorkSpace/classification_agricultural/farm_boundary/cythonn/Vectorization.pyx")
)