import setuptools

# read the description file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'doc/pycinema-description.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open("pycinema/_version.py").read())

print(__version__)

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
    zip_safe=False,
    packages=[  "pycinema", "pycinema.filters", "pycinema.scripts", "pycinema.theater", "pycinema.theater.node_editor", "pycinema.theater.views", "pycinema.ipy" ],
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.14.1",
        "h5py>=3.10.0",
        "matplotlib==3.6.0",
        "py==1.11.0",
        "Pillow==9.4.0",
        "moderngl==5.10.0",
        "opencv-python>=4.7.0",
        "ipycanvas==0.13.1",
        "ipywidgets==8.0.6",
        "PySide6>=6.7.2",
        "igraph>=0.10.5",
        "requests>=2.31.0",
        "pyqtgraph>=0.13.3",
        "tensorflow>=2.17.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        'doc/pycinema-description.md',
        'scripts/cinema'
    ],
    data_files=[('fonts',['fonts/NotoSansMono-VariableFont_wdth,wght.ttf'])]
)
