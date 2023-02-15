# Cinema Engine repository 
![smoke](https://github.com/cinemascience/pycinema/actions/workflows/RenderTest.yml/badge.svg)

Cinema v2.0 is a newly designed toolkit of python-based components for creating, filtering, transforming and viewing Cinema databases. There is more information about the Cinema project [here](https://cinemascience.github.io) 

The code in this repository is released under open source license. See the license file for more information.

## Installing and running with the pycinema module

```
python3 -m venv pcenv
source pcenv/bin/activate
pip install --upgrade pip
pip install pycinema 
pip install jupyterlab
```

## Installing and running with the test pycinema module

```
python3 -m venv pcenv
source pcenv/bin/activate
pip install --upgrade pip
pip install -i https://test.pypi.org/simple/ --extra-index https://pypi.org/simple pycinema
pip install jupyterlab
```

# Creating a local python environment

To create a local python environment for this project, run the following commands within the repository directory:
```
python3 -m venv pcenv
source pcenv/bin/activate
pip install --upgrade pip
pip install jupyterlab
pip install .
```

# Running examples

You can now use this python environment to run examples from the repository. Run `jupyter-lab` and select a file from the `examples` directory:

```
source pcenv/bin/activate
jupyter-lab
```

# Making and uploading the python module

```
make module
``` 

And then to upload it to `pypi` (assuming you have permission):

```
make module-upload
```

To upload it to `testpypi` (assuming you have permission):

```
make module-test-upload
```

# Design proposals

- [CIS image proposal](doc/cis_proposal.md)

# Contributing

Contributions can be made by submitting issues and contributing code through pull requests. The code shall be reviewed by the core Cinema team, and accepted when both content and code standards are met.


