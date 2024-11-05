# creating environments and testing with pyenv 

```
pyenv local <version>
python -m venv venv<version>
rm .python-version
source venv<version>/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m pip install pytest
```
