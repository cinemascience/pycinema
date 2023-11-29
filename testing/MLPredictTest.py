import pycinema
import pycinema.filters
import pytest
import filecmp

def test_writetable():

    # filters
    CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
    ImageReader_0 = pycinema.filters.ImageReader()
    MLTFReader_0 = pycinema.filters.MLTFReader()
    MLTFPredictor_0 = pycinema.filters.MLTFPredictor()
    ImagesToTable_0 = pycinema.filters.ImagesToTable()
    # PlotScatterItem_0 = pycinema.filters.PlotScatterItem()
    TableWriter_0 = pycinema.filters.TableWriter()

    # properties
    CinemaDatabaseReader_0.inputs.path.set("data/mnist.cdb", False)
    CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
    ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
    ImageReader_0.inputs.file_column.set("FILE", False)
    ImageReader_0.inputs.cache.set(True, False)
    MLTFReader_0.inputs.path.set("data/MNIST_models/TF/mnist_tf.h5", False)
    MLTFPredictor_0.inputs.trainedModel.set(MLTFReader_0.outputs.model, False)
    MLTFPredictor_0.inputs.images.set(ImageReader_0.outputs.images, False)
    ImagesToTable_0.inputs.images.set(MLTFPredictor_0.outputs.images, False)
    TableWriter_0.inputs.path.set("MLOutputTest.csv", False)
    TableWriter_0.inputs.table.set(ImagesToTable_0.outputs.table, False)

    # execute pipeline
    CinemaDatabaseReader_0.update()

    # check results
    assert filecmp.cmp('MLOutputTest.csv', 'testing/gold/MLOutputTest.csv')
