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
        "numpy<2.0.0",
        "scipy==1.13.1",
        "h5py",
        "matplotlib",
        "py",
        "Pillow",
        "moderngl",
        "opencv-python-headless",
        "ipycanvas",
        "ipywidgets",
        "PySide6",
        "igraph",
        "requests",
        "pyqtgraph",
        "tensorflow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        'doc/pycinema-description.md',
        'scripts/cinema',
        'scripts/cinematic'
    ],
    data_files=[('fonts',['fonts/NotoSansMono-VariableFont_wdth,wght.ttf'])]
)
