import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.2.0'}

# filters
DSIReader_0 = pycinema.filters.DSIReader()
InspectorView_0 = pycinema.filters.InspectorView()
ImageReader_0 = pycinema.filters.ImageReader()
ImageView_0 = pycinema.filters.ImageView()

# properties
DSIReader_0.inputs.path.set("DSIscratch/sphere.cdb/data.db", False)
DSIReader_0.inputs.tablename.set("datacsv", False)
InspectorView_0.inputs.object.set(DSIReader_0.outputs.table, False)
ImageReader_0.inputs.table.set(DSIReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)

# layout
tabFrame1 = pycinema.theater.TabFrame()
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setHorizontalOrientation()
view1 = pycinema.theater.views.NodeEditorView()
splitFrame1.insertView( 0, view1 )
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setVerticalOrientation()
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setVerticalOrientation()
view2 = pycinema.theater.views.FilterView( InspectorView_0 )
splitFrame3.insertView( 0, view2 )
view9 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame3.insertView( 1, view9 )
splitFrame3.setSizes([387, 508])
splitFrame2.insertView( 0, splitFrame3 )
splitFrame2.setSizes([902])
splitFrame1.insertView( 1, splitFrame2 )
splitFrame1.setSizes([858, 857])
tabFrame1.insertTab(0, splitFrame1)
tabFrame1.setTabText(0, 'Layout 1')
tabFrame1.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame1)

# execute pipeline
DSIReader_0.update()
