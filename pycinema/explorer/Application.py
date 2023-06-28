class Application():

    Scripts = {}
    Scripts['view'] = '''
import pycinema
import pycinema.filters
import pycinema.explorer

# layout
vf0 = pycinema.explorer.Explorer.window.centralWidget()
vf0.s_splitH()

vf1 = vf0.widget(0)
vf2 = vf0.widget(1)

vf2.s_splitH()
vf3 = vf2.widget(0)
vf3.s_splitV()
vf5 = vf3.widget(0)
ParameterViewer_0 = vf5.convert( pycinema.explorer.ParameterViewer )
vf6 = vf3.widget(1)
vf6.s_splitV()
vf7 = vf6.widget(0)
TableViewer_0 = vf7.convert( pycinema.explorer.TableViewer )
vf8 = vf6.widget(1)
ColorMappingViewer_0 = vf8.convert( pycinema.explorer.ColorMappingViewer )
vf4 = vf2.widget(1)
ImageViewer_0 = vf4.convert( pycinema.explorer.ImageViewer )

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()

# properties
ParameterViewer_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableViewer_0.inputs.table.set(TableQuery_0.outputs.table, False)
ColorMappingViewer_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ImageViewer_0.inputs.images.set(ColorMappingViewer_0.outputs.images, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(ParameterViewer_0.outputs.sql, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.composite_by_meta.set(ParameterViewer_0.outputs.composite_by_meta, False)
vf1.widget(0).hide()
'''

    def __init__(self, app, **kwargs):
        self.app        = app
        self.script     = Application.Scripts[self.app]
        self.filepath   = kwargs['filepath']

        self.script += 'CinemaDatabaseReader_0.inputs.path.set("'+self.filepath+'", False)\n'
        self.script += 'CinemaDatabaseReader_0.update()'

    def getScript(self):

        return self.script
