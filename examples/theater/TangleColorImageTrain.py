import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
CinemaDatabaseReader_1 = pycinema.filters.CinemaDatabaseReader()
ImageView_1 = pycinema.filters.ImageView()
ImageReader_1 = pycinema.filters.ImageReader()
RenderView_1 = pycinema.filters.RenderView()
SynemaColorImageModelFactory_1 = pycinema.filters.SynemaColorImageModelFactory()
SynemaColorImageModelTrainer_1 = pycinema.filters.SynemaColorImageModelTrainer()
SynemaColorImageViewSynthesis_1 = pycinema.filters.SynemaColorImageViewSynthesis()

# properties
CinemaDatabaseReader_1.inputs.path.set("/home/ollie/PycharmProjects/imdb_nerf/tangle_cinema.cdb", False)
CinemaDatabaseReader_1.inputs.file_column.set("FILE", False)
ImageView_1.inputs.images.set(ImageReader_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)
ImageReader_1.inputs.table.set(CinemaDatabaseReader_1.outputs.table, False)
ImageReader_1.inputs.file_column.set("FILE", False)
ImageReader_1.inputs.cache.set(True, False)
RenderView_1.inputs.images.set(SynemaColorImageViewSynthesis_1.outputs.images, False)
RenderView_1.inputs.camera.set([34.0, 50.8], False)
SynemaColorImageModelTrainer_1.inputs.model_state.set(SynemaColorImageModelFactory_1.outputs.model_state, False)
SynemaColorImageModelTrainer_1.inputs.channel.set("rgba", False)
SynemaColorImageModelTrainer_1.inputs.images.set(ImageReader_1.outputs.images, False)
SynemaColorImageModelTrainer_1.inputs.epochs.set(0, False)
SynemaColorImageViewSynthesis_1.inputs.model_state.set(SynemaColorImageModelTrainer_1.outputs.model_state, False)
SynemaColorImageViewSynthesis_1.inputs.camera.set(RenderView_1.inputs.camera, False)

# layout
tabFrame2 = pycinema.theater.TabFrame()
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setVerticalOrientation()
view4 = pycinema.theater.views.NodeEditorView()
splitFrame3.insertView( 0, view4 )
splitFrame4 = pycinema.theater.SplitFrame()
splitFrame4.setHorizontalOrientation()
view6 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame4.insertView( 0, view6 )
view8 = pycinema.theater.views.FilterView( RenderView_1 )
splitFrame4.insertView( 1, view8 )
splitFrame4.setSizes([798, 797])
splitFrame3.insertView( 1, splitFrame4 )
splitFrame3.setSizes([829, 829])
tabFrame2.insertTab(0, splitFrame3)
tabFrame2.setTabText(0, 'Layout 1')
tabFrame2.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame2)

# execute pipeline
CinemaDatabaseReader_1.update()
