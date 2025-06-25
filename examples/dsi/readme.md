# DSI examples and tests

DSI databases must first be created in place, and then read. There is a single
test and an ensemble test. Running the create script will create data in
`./DSIscratch` for the tests.

## Single

```
python example/dsi/DSITestCreateSingle.py
cinema example/dsi/DSITestReadSingle.py
```

## Ensemble

```
python example/dsi/DSITestCreateEnsemble.py
cinema example/dsi/DSITestReadEnsemble.py
```
