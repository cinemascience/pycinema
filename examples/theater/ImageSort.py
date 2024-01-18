import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setVerticalOrientation()
vf1 = vf0.insertFrame(0)
vf1.setHorizontalOrientation()
vf1.insertView( 0, pycinema.theater.views.NodeEditorView() )
vf1.setSizes([1248])
vf2 = vf0.insertFrame(1)
vf2.setHorizontalOrientation()
ImageView_2 = vf2.insertView( 0, pycinema.theater.views.ImageView() )
ImageView_3 = vf2.insertView( 1, pycinema.theater.views.ImageView() )
vf2.setSizes([621, 620])
vf0.setSizes([421, 421])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
ImageReader_0 = pycinema.filters.ImageReader()
ImageAnnotation_1 = pycinema.filters.ImageAnnotation()
ImageSort_0 = pycinema.filters.ImageSort()

# properties
ImageView_2.inputs.images.set(ImageAnnotation_1.outputs.images, False)
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageAnnotation_1.inputs.images.set(ImageReader_0.outputs.images, False)
ImageAnnotation_1.inputs.xy.set((20, 20), False)
ImageAnnotation_1.inputs.size.set(20, False)
ImageAnnotation_1.inputs.spacing.set(0, False)
ImageAnnotation_1.inputs.color.set((), False)
ImageAnnotation_1.inputs.ignore.set(['^file', '^id'], False)
ImageSort_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageSort_0.inputs.sortBy.set("phi", False)
ImageSort_0.inputs.reverse.set(1, False)
ImageView_3.inputs.images.set(ImageSort_0.outputs.images, False)

# execute pipeline
ImageView_2.update()
