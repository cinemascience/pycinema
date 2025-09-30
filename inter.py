import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
VideoWriter_0 = pycinema.filters.VideoWriter()
ImageInterpolator_0 = pycinema.filters.ImageInterpolator()
ImageInterpolator2_0 = pycinema.filters.ImageInterpolator2()

# properties
CinemaDatabaseReader_0.inputs.path.set("/home/jones/projects/cinema-lib/pycinema-error/data/time_sequences/volume_stride1.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set('''WITH NumberedRows AS (
    SELECT 
        *, 
        ROW_NUMBER() OVER (ORDER BY time) AS RowNum
    FROM input
)
SELECT *
FROM NumberedRows
WHERE RowNum % 3 = 0''', False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
VideoWriter_0.inputs.images.set([], False)
VideoWriter_0.inputs.path.set("/home/jones/projects/cinema-lib/pycinema-error/data/time_sequences/asteroid_interpolation_rife.mp4", False)
VideoWriter_0.inputs.fps.set(30, False)
ImageInterpolator_0.inputs.images.set(ImageReader_0.outputs.images, False)
ImageInterpolator_0.inputs.model_path.set("/home/jones/projects/cinema-lib/pycinema-inter/data/film_net_fp32.pt", False)
ImageInterpolator_0.inputs.nFrames.set(3, False)
ImageInterpolator_0.inputs.adaptive.set(True, False)
ImageInterpolator2_0.inputs.images.set([], False)
ImageInterpolator2_0.inputs.nFrames.set(2, False)

# layout
tabFrame0 = pycinema.theater.TabFrame()
splitFrame0 = pycinema.theater.SplitFrame()
splitFrame0.setHorizontalOrientation()
view0 = pycinema.theater.views.NodeEditorView()
splitFrame0.insertView( 0, view0 )
splitFrame0.setSizes([1020])
tabFrame0.insertTab(0, splitFrame0)
tabFrame0.setTabText(0, 'Layout 1')
tabFrame0.setCurrentIndex(0)
pycinema.theater.Theater.instance.setCentralWidget(tabFrame0)

# execute pipeline
CinemaDatabaseReader_0.update()
