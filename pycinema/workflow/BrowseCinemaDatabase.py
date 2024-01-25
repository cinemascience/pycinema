from pycinema import Workflow

class BrowseCinemaDatabase(Workflow):
    def __init__(self):
        super().__init__()

    def initializeScript(self, **kwargs): 

        path = ""
        if 'filename' in kwargs:
            path = kwargs['filename']

        self._script = """
import pycinema
import pycinema.filters
import pycinema.theater
import pycinema.theater.views

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# layout
vf0 = pycinema.theater.Theater.instance.centralWidget()
vf0.setHorizontalOrientation()
vf0.insertView( 0, pycinema.theater.views.NodeEditorView() )
ImageView_1 = vf0.insertView( 1, pycinema.theater.views.ImageView() )
vf0.setSizes([999, 999])

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()

# properties
CinemaDatabaseReader_0.inputs.path.set("data/sphere.cdb", False)
CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set("SELECT * FROM input LIMIT 100", False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
ImageReader_0.inputs.file_column.set("FILE", False)
ImageReader_0.inputs.cache.set(True, False)
ImageView_1.inputs.images.set(ImageReader_0.outputs.images, False)

# set path to database
"""
        self._script += 'CinemaDatabaseReader_0.inputs.path.set("'+path+'", False)\n'
        self._script += '\n'
        self._script += '# execute\n'
        self._script += 'CinemaDatabaseReader_0.update()'
        self._script += '\n'
