import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageView_0 = pycinema.filters.ImageView()
ImageReader_0 = pycinema.filters.ImageReader()
RenderView_0 = pycinema.filters.RenderView()
SynemaColorImageModelFactory_0 = pycinema.filters.SynemaColorImageModelFactory()
SynemaColorImageModelTrainer_0 = pycinema.filters.SynemaColorImageModelTrainer()
SynemaColorImageViewSynthesis_0 = pycinema.filters.SynemaColorImageViewSynthesis()

# properties
CinemaDatabaseReader_0.inputs.path.set("/home/ollie/PycharmProjects/imdb_nerf/asteroid_cinema.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
RenderView_0.inputs.images.set(SynemaColorImageViewSynthesis_0.outputs.images, False)
RenderView_0.inputs.camera.set([34.0, 50.8], False)
SynemaColorImageModelTrainer_0.inputs.model_state.set(SynemaColorImageModelFactory_0.outputs.model_state, False)
SynemaColorImageModelTrainer_0.inputs.channel.set("rgba", False)
SynemaColorImageModelTrainer_0.inputs.images.set(ImageReader_0.outputs.images, False)
SynemaColorImageModelTrainer_0.inputs.epochs.set(0, False)
SynemaColorImageViewSynthesis_0.inputs.model_state.set(SynemaColorImageModelTrainer_0.outputs.model_state, False)
SynemaColorImageViewSynthesis_0.inputs.camera.set(RenderView_0.inputs.camera, False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setVerticalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view2 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 0, view2 )
view3 = pycinema.theater.views.FilterView( RenderView_0 )
splitFrame2.insertView( 1, view3 )
splitFrame2.setSizes([508, 508])
splitFrame1.insertView( 1, splitFrame2 )
splitFrame1.setSizes([409, 409])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
