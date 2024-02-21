import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()
ImageSort_0 = pycinema.filters.ImageSort()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageAnnotation_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageSort_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageSort_0.inputs.sortBy.set("phi", False)
ImageSort_0.inputs.reverse.set(0, False)
ImageAnnotation_0.inputs.images.set(ImageSort_0.outputs.images, False)
ImageAnnotation_0.inputs.xy.set((20, 20), False)
ImageAnnotation_0.inputs.size.set(20, False)
ImageAnnotation_0.inputs.spacing.set(0, False)
ImageAnnotation_0.inputs.color.set((), False)
ImageAnnotation_0.inputs.ignore.set(['^file', '^id'], False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
view4 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame0.insertView( 1, view4 )
splitFrame0.setSizes([834, 833])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
