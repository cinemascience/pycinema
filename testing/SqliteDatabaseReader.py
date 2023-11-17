import pycinema
import pycinema.filters
import pytest
import filecmp

def test_writetable():
    # filters
    SqliteDatabaseReader_0 = pycinema.filters.SqliteDatabaseReader()
    TableWriter_0 = pycinema.filters.TableWriter()

    # properties
    SqliteDatabaseReader_0.inputs.path.set("data/wildfire/wildfire.sqlite3", False)
    SqliteDatabaseReader_0.inputs.table.set("vision", False)
    SqliteDatabaseReader_0.inputs.file_column.set("FILE", False)
    TableWriter_0.inputs.path.set("sqlite3.csv", False)
    TableWriter_0.inputs.table.set(SqliteDatabaseReader_0.outputs.table, False)

    # execute pipeline
    SqliteDatabaseReader_0.update()

    assert filecmp.cmp('sqlite3.csv', 'testing/gold/SqliteDatabaseReader.csv')
