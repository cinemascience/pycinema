import pycinema
import pycinema.filters
import pytest
import filecmp

def test_writetable():

    # filters
    CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
    TableWriter_0 = pycinema.filters.TableWriter()

    # properties
    CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
    CinemaDatabaseReader_0.inputs.file_column.set("file", False)
    TableWriter_0.inputs.path.set("TableWriteTest.csv", False)
    TableWriter_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)

    # execute pipeline
    CinemaDatabaseReader_0.update()

    assert filecmp.cmp('TableWriteTest.csv', 'testing/gold/TableWriteTest.csv')
