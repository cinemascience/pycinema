# DSI support in pycinema

`pycinema` is developing support for LANL's [DSI](https://github.com/lanl/dsi) project.
At the moment, the best way to think about this is that `pycinema` accesses the DSI
API to read DSI databases in exactly the same way that `pycinema` reads cinema databases.

## DSIReader filter

The DSIReader filter takes a database as input, and outputs a table. That table can
then be operated on as expected with other filters.

## DSI utilities in this repository

- `utils/dsi-export.py` This utility takes a cinema database as input and creates
  a DSI-compliant database in the cinema database that is equivalent to the `data.csv`
  file in the database. This new database (`data.db`) can then be read by the 
  DSIReader filter.

- `data/view_asteroid_dsi.py` This script is an example of accessing, recoloring and
  viewing a DSI-compliant database in a cinema database.

- `.github/workflow/DSITest.yaml` This CI action automatically tests basic DSI capability.

