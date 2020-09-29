import setuptools
from pyqcr import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-qcr",
    version=__version__,
    author="Yury Nahshan",
    author_email="ynahshan@habana.ai",
    description="NN quantization and calibration package for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ynahshan/pytorch-qcr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
