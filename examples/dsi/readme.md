# DSI examples and tests

DSI databases must first be created in place, and then read. There is a single
test and an ensemble test. Running the create script will create data in
`./DSIscratch` for the tests.

## Single

```
python examples/dsi/DSITestCreateSingle.py
cinema examples/dsi/DSITestReadSingle.py
```

## Ensemble

```
python examples/dsi/DSITestCreateEnsemble.py
cinema examples/dsi/DSITestReadEnsemble.py
```
