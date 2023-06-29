#
# The Application class is a factory for creating applications within
# the pycinema framework.
#
# An application is a script that constructs a filter graph that can be
# executed either through the framework, or in a python script. The 
# difference between the two types of execution is whether the filer
# graph contains UI elements or not. A filter graph without UI elements
# can be executed in pure python, while a graph with UI elements must
# be executed in the pycinema application.
#
# A more general framework for adding applications is imagined for the
# future, but for this design, all application scripts shall be registerd
# in this file.
#
class Application():

    Scripts = {}
    Scripts['view'] = '''
import pycinema
import pycinema.filters
import pycinema.explorer

# cinema application attributes
pycinema_view_version = '1.0.0'

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

    #
    # An instance of an application is created by providing an app name (type), and an
    # array of keyward args. Those args must be validated by the application type 
    # requested.
    #
    def __init__(self, app, **kwargs):
        self.app = app
        self.script = ""

        for key, value in kwargs.items():
            setattr(self, key, value)

        #
        # if an app type is registered, it must construct a valid script using the
        # input keyword argument list
        #
        match app:
            case "view":
                self.script  = Application.Scripts[self.app]
                self.script += 'CinemaDatabaseReader_0.inputs.path.set("'+self.filepath+'", False)\n'
                self.script += 'CinemaDatabaseReader_0.update()'

            case _:
                print("Unrecognized app " + app)

    #
    # Return the application instance's script
    #
    def getScript(self):
        return self.script
