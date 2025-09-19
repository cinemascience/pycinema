# Release procedure

Releasing pycinema is a combination of automatic testing, user testing and module testing.

# CI testing

CI testing is automatically executed with github actions. These should be adjusted per the capabilities
in the release. In general, all checkins to `dev` should be verified to not break any existing tests.
In addition, the versions of OS and python should be checked to cover current supported releases.

# User/hand testing

Testing of examples and interactive `cinema` execution is done by hand in two steps:

## Test all interactive scripts

Run all example scripts in `examples/` directory.

` `examples/compose`
```
    cinema examples/compose/<name of script>    tests `compose` subcommand
```

- `examples/theater` these are examples that run scripts and bring up the theater interactive application.

```
   cinema examples/theater/<name of script>     these should all run to completion 
```

# Updating version

After CI and user testing is complete, update the version number, make and test the module release.

- Update `pycinema/_version.py` to current version
- make, upload and test `pycinema` python module at test.pypi.org, using the makefile directives:
    - `make module`
    - `make module-test-upload`
	- `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pycinema`
- upload `pycinema` python module, using the makefile directives:
    - `make module-upload`

## Update related repositories

- update `pycinema-data` with any new examples, and change legacy examples if needed 
- update `pycinema-examples` repository to reflect the `examples` and `data` repository of the release.
