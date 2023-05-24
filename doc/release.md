# Release procedure

- Update '''version.md''' to current version
- make, upload and test '''pycinema''' python module at test.pypi.org, using the makefile directives:
    - '''make module'''
    - '''make module-test-upload'''
	- '''pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pycinema'''
- upload '''pycinema''' python module, using the makefile directives:
    - '''make module-upload'''
