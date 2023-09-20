import pycinema
import pycinema.filters
import pytest
import filecmp

#
# pycinema testing script
#
# this script was constructed by creating a pipeline in theater
# and then editing it to remove UI-based filters. The resulting
# script can be run with python, and the output tested
#

def test_querywrite():

    # filters
    CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
    ImageReader_0 = pycinema.filters.ImageReader()
    TableQuery_0 = pycinema.filters.TableQuery()
    CinemaDatabaseWriter_0 = pycinema.filters.CinemaDatabaseWriter()

    # properties
    CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
    CinemaDatabaseReader_0.inputs.file_column.set("file", False)
    ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
    ImageReader_0.inputs.file_column.set("FILE", False)
    ImageReader_0.inputs.cache.set(True, False)
    TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
    TableQuery_0.inputs.sql.set("SELECT * FROM input WHERE phi=-144", False)
    CinemaDatabaseWriter_0.inputs.path.set("QueryWriteTest.cdb", False)
    CinemaDatabaseWriter_0.inputs.images.set(ImageReader_0.outputs.images, False)

    # execute pipeline
    CinemaDatabaseReader_0.update()

    # check results
    assert filecmp.dircmp('QueryWriteTest.cdb', 'testing/gold/QueryWriteTest.cdb')
