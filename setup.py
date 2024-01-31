import setuptools

# read the description file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'doc/description.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open("pycinema/_version.py").read())

setuptools.setup(
    name="pycinema",
    version=__version__,
    author="David H. Rogers",
    author_email="dhr@lanl.gov",
    description="Cinema scientific toolset.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cinemascience/pycinema",
    include_package_data=True,
    packages=[  "pycinema", "pycinema.filters", "pycinema.theater", "pycinema.workspace", "pycinema.theater.node_editor", "pycinema.theater.views", "pycinema.ipy" ],
    install_requires=[
        "numpy==1.24.2",
        "scipy==1.10.0",
        "h5py==3.8.0",
        "matplotlib==3.6.0",
        "py==1.11.0",
        "Pillow==9.4.0",
        "moderngl<6",
        "opencv-python==4.7.0.68",
        "ipycanvas==0.13.1",
        "ipywidgets==8.0.6",
        "PySide6<=6.4.3",
        "python-igraph>=0.10.5",
        "requests>=2.31.0",
        "pyqtgraph>=0.13.3",
        "tensorflow==2.14.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        'doc/description.md',
        'scripts/cinema'
    ],
)
