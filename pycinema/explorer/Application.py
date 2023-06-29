class Application():

    Scripts = {}
    Scripts['view'] = '''
import pycinema
import pycinema.filters
import pycinema.explorer

# layout
centralWidget = pycinema.explorer.Explorer.window.centralWidget()
centralWidget.s_splitH()

nodeView = centralWidget.widget(0)
appFrame = centralWidget.widget(1)

appFrame.s_splitH()
parameterFrame = appFrame.widget(0)
parameterFrame.s_splitV()

parameterView = parameterFrame.widget(0)
ParameterViewer_0 = parameterView.convert( pycinema.explorer.ParameterViewer )
colorMapView = parameterFrame.widget(1)
ColorMappingViewer_0 = colorMapView.convert( pycinema.explorer.ColorMappingViewer )

imageFrame = appFrame.widget(1)
ImageViewer_0 = imageFrame.convert( pycinema.explorer.ImageViewer )

# filters
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()

# properties
ParameterViewer_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ColorMappingViewer_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ImageViewer_0.inputs.images.set(ColorMappingViewer_0.outputs.images, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set(ParameterViewer_0.outputs.sql, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.composite_by_meta.set(ParameterViewer_0.outputs.composite_by_meta, False)
nodeView.widget(0).hide()
'''

    def __init__(self, app, **kwargs):
        self.app = app
        self.script = ""

        for key, value in kwargs.items():
            setattr(self, key, value)

        match app:
            case "view":
                self.script  = Application.Scripts[self.app]
                self.script += 'CinemaDatabaseReader_0.inputs.path.set("'+self.filepath+'", False)\n'
                self.script += 'CinemaDatabaseReader_0.update()'

            case _:
                print("Unrecognized app " + app)

    def getScript(self):

        return self.script
