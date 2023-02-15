# Cinema Engine repository 
![smoke](https://github.com/cinemascience/pycinema/actions/workflows/RenderTest.yml/badge.svg)

Cinema v2.0 is a newly designed toolkit of python-based components for creating, filtering, transforming and viewing Cinema databases. The toolkit shall maintain compatibility with Cinema data specifications.

The code in this repository is released under open source license. See the license file for more information.

All code and examples are prototype and for design purposes only

# Creating a local python environment

To create a local python environment for this project, run the following commands within the repository directory:
```
python3 -m venv csenv
source csenv/bin/activate
pip install --upgrade pip
pip install jupyterlab
pip install .
```

# Running examples

You can now use this python environment to run examples from the repository. Run `jupyter-lab` and select a file from the `examples` directory:

```
source csenv/bin/activate
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

# Design proposals

- [CIS image proposal](doc/cis_proposal.md)

# Contributing

Contributions can be made by submitting issues and contributing code through pull requests. The code shall be reviewed by the core Cinema team, and accepted when both content and code standards are met.


