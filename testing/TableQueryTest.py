import pycinema
import pycinema.filters
import pytest
import filecmp

def test_writetable():
    # filters
    CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
    TableQuery_0 = pycinema.filters.TableQuery()
    TableWriter_0 = pycinema.filters.TableWriter()

    # properties
    CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
    CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
    TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
    TableQuery_0.inputs.sql.set("SELECT id FROM input WHERE phi>0", False)
    TableWriter_0.inputs.path.set("TableQueryTest.csv", False)
    TableWriter_0.inputs.table.set(TableQuery_0.outputs.table, False)

    # execute pipeline
    CinemaDatabaseReader_0.update()

    assert filecmp.cmp('TableQueryTest.csv', 'testing/gold/TableQueryTest.csv')
