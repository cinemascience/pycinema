import pycinema
import pycinema.filters
import pytest
import filecmp

def test_pythonscript():
    # filters
    CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
    ImageReader_0 = pycinema.filters.ImageReader()
    CinemaDatabaseWriter_0 = pycinema.filters.CinemaDatabaseWriter()
    TableQuery_0 = pycinema.filters.TableQuery()

    # properties
    CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
    CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
    ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
    ImageReader_0.inputs.file_column.set("FILE", False)
    ImageReader_0.inputs.cache.set(True, False)
    CinemaDatabaseWriter_0.inputs.images.set(ImageReader_0.outputs.images, False)
    CinemaDatabaseWriter_0.inputs.path.set("PythonPlotTest.cdb", False)
    CinemaDatabaseWriter_0.inputs.ignore.set(['^id', '^camera', '^FILE', '^FILE', '^FILE'], False)
    TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
    TableQuery_0.inputs.sql.set("SELECT * FROM input where phi>150", False)

    # execute pipeline
    CinemaDatabaseReader_0.update()

    assert filecmp.cmp('PythonPlotTest.cdb/data.csv', 'testing/gold/PythonPlotTest.cdb/data.csv')
