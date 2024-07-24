import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.0.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
SynemaViewSynthesis_0 = pycinema.filters.SynemaViewSynthesis()
ColorMapping_0 = pycinema.filters.ColorMapping()
ImageView_0 = pycinema.filters.ImageView()
ColorMapping_1 = pycinema.filters.ColorMapping()
ImageView_1 = pycinema.filters.ImageView()

# properties
CinemaDatabaseReader_0.inputs.path.set("/home/ollie/PycharmProjects/pycinema/data/dragon.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
SynemaViewSynthesis_0.inputs.images.set(ImageReader_0.outputs.images, False)
ColorMapping_0.inputs.map.set("plasma", False)
ColorMapping_0.inputs.nan.set((1, 1, 1, 1), False)
ColorMapping_0.inputs.range.set((0, 1), False)
ColorMapping_0.inputs.channel.set("Depth", False)
ColorMapping_0.inputs.images.set(ImageReader_0.outputs.images, False)
ColorMapping_0.inputs.composition_id.set(-1, False)
ImageView_0.inputs.images.set(ColorMapping_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ColorMapping_1.inputs.map.set("plasma", False)
ColorMapping_1.inputs.nan.set((1, 1, 1, 1), False)
ColorMapping_1.inputs.range.set((0, 1), False)
ColorMapping_1.inputs.channel.set("scalar_recon", False)
ColorMapping_1.inputs.images.set(SynemaViewSynthesis_0.outputs.images, False)
ColorMapping_1.inputs.composition_id.set(-1, False)
ImageView_1.inputs.images.set(ColorMapping_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)

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
view3 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame2.insertView( 1, view3 )
splitFrame2.setSizes([798, 797])
splitFrame1.insertView( 1, splitFrame2 )
splitFrame1.setSizes([830, 828])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
CinemaDatabaseReader_0.update()
