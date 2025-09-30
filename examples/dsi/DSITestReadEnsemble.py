import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.2.0'}

# filters
DSIReader_0 = pycinema.filters.DSIReader()
DSIReader_1 = pycinema.filters.DSIReader()
DSIReader_2 = pycinema.filters.DSIReader()

ImageReader_0 = pycinema.filters.ImageReader()
ImageReader_1 = pycinema.filters.ImageReader()
ImageReader_2 = pycinema.filters.ImageReader()

ImageView_0 = pycinema.filters.ImageView()
ImageView_1 = pycinema.filters.ImageView()
ImageView_2 = pycinema.filters.ImageView()

InspectorView_0 = pycinema.filters.InspectorView()
InspectorView_1 = pycinema.filters.InspectorView()
InspectorView_2 = pycinema.filters.InspectorView()

# properties
DSIReader_0.inputs.path.set("DSIscratch/sphere00.cdb/data.db", False)
DSIReader_0.inputs.tablename.set("datacsv", False)
DSIReader_1.inputs.path.set("DSIscratch/sphere01.cdb/data.db", False)
DSIReader_1.inputs.tablename.set("datacsv", False)
DSIReader_2.inputs.path.set("DSIscratch/sphere02.cdb/data.db", False)
DSIReader_2.inputs.tablename.set("datacsv", False)

ImageReader_0.inputs.table.set(DSIReader_2.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageReader_1.inputs.table.set(DSIReader_1.outputs.table, False)
ImageReader_1.inputs.file_column.set("FILE", False)
ImageReader_1.inputs.cache.set(True, False)
ImageReader_2.inputs.table.set(DSIReader_0.outputs.table, False)
ImageReader_2.inputs.file_column.set("FILE", False)
ImageReader_2.inputs.cache.set(True, False)
ImageView_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageView_0.inputs.selection.set([], False)
ImageView_1.inputs.images.set(ImageReader_1.outputs.images, False)
ImageView_1.inputs.selection.set([], False)
ImageView_2.inputs.images.set(ImageReader_2.outputs.images, False)
ImageView_2.inputs.selection.set([], False)
InspectorView_0.inputs.object.set(DSIReader_2.outputs.table, False)
InspectorView_1.inputs.object.set(DSIReader_1.outputs.table, False)
InspectorView_2.inputs.object.set(DSIReader_0.outputs.table, False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame1 = pycinema.theater.SplitFrame()
splitFrame1.setVerticalOrientation()
splitFrame2 = pycinema.theater.SplitFrame()
splitFrame2.setHorizontalOrientation()
view4 = pycinema.theater.views.FilterView( ImageView_0 )
splitFrame2.insertView( 0, view4 )
view10 = pycinema.theater.views.FilterView( InspectorView_0 )
splitFrame2.insertView( 1, view10 )
splitFrame2.setSizes([855, 596])
splitFrame1.insertView( 0, splitFrame2 )
splitFrame3 = pycinema.theater.SplitFrame()
splitFrame3.setHorizontalOrientation()
view5 = pycinema.theater.views.FilterView( ImageView_1 )
splitFrame3.insertView( 0, view5 )
view11 = pycinema.theater.views.FilterView( InspectorView_1 )
splitFrame3.insertView( 1, view11 )
splitFrame3.setSizes([855, 596])
splitFrame1.insertView( 1, splitFrame3 )
splitFrame4 = pycinema.theater.SplitFrame()
splitFrame4.setHorizontalOrientation()
view6 = pycinema.theater.views.FilterView( ImageView_2 )
splitFrame4.insertView( 0, view6 )
view12 = pycinema.theater.views.FilterView( InspectorView_2 )
splitFrame4.insertView( 1, view12 )
splitFrame4.setSizes([855, 596])
splitFrame1.insertView( 2, splitFrame4 )
splitFrame1.setSizes([431, 431, 431])
splitFrame0.insertView( 1, splitFrame1 )
splitFrame0.setSizes([1232, 1458])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
DSIReader_0.update()
