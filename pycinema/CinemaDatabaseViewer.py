from .Core import *
from .CinemaDatabaseReader import *
from .DatabaseQuery import *
from .ImageReader import *
from .DepthCompositing import *
from .ParameterWidgets import *
from .Annotation import *
from .ImageViewer import *
from .ShaderSSAO import *
from .ColorMappingWidgets import *
from .ColorMapping import *
from .NumberWidget import *

import IPython
import ipywidgets

class CinemaDatabaseViewer(Filter):

    def __init__(self, path, preload_query="SELECT * FROM input"):
        super().__init__()

        self.addInputPort("path", "./")

        # Layout
        self.parameterWidgetsContainer = ipywidgets.VBox();
        self.colorMappingWidgetsContainer = ipywidgets.VBox();
        self.shadingWidgetsContainer = ipywidgets.VBox();

        self.leftColumn = ipywidgets.VBox([
            ipywidgets.Accordion(children=[self.parameterWidgetsContainer]),
            ipywidgets.Accordion(children=[self.colorMappingWidgetsContainer]),
            ipywidgets.Accordion(children=[self.shadingWidgetsContainer])
        ]);
        self.leftColumn.children[0].set_title(0,'Parameters')
        self.leftColumn.children[1].set_title(0,'Color Mapping')
        self.leftColumn.children[2].set_title(0,'Shading')

        self.imageContainer = ipywidgets.Output()
        self.globalContainer = ipywidgets.HBox([self.leftColumn,self.imageContainer]);

        # Pipeline
        self.cinemaDatabaseReader  = CinemaDatabaseReader()
        self.cinemaDatabaseReader.inputs.path.set(self.inputs.path, False)

        preload_results = DatabaseQuery();
        preload_results.inputs.table.set(self.cinemaDatabaseReader.outputs.table, False);
        preload_results.inputs.sql.set(preload_query, False);

        self.parameterWidgets = ParameterWidgets()
        self.parameterWidgets.inputs.table.set(preload_results.outputs.table,False)
        self.parameterWidgets.inputs.container.set(self.parameterWidgetsContainer,False)

        self.databaseQuery = DatabaseQuery()
        self.databaseQuery.inputs.table.set(self.cinemaDatabaseReader.outputs.table,False)
        self.databaseQuery.inputs.sql.set(self.parameterWidgets.outputs.sql,False)

        self.imageReader = ImageReader()
        self.imageReader.inputs.table.set(self.databaseQuery.outputs.table,False)

        self.depthCompositing = DepthCompositing()
        self.depthCompositing.inputs.images_a.set(self.imageReader.outputs.images,False)

        self.colorMappingWidgets = ColorMappingWidgets()
        self.colorMappingWidgets.inputs.images.set(self.depthCompositing.outputs.images,False)
        self.colorMappingWidgets.inputs.container.set(self.colorMappingWidgetsContainer,False)

        self.colorMapping = ColorMapping()
        self.colorMapping.inputs.images.set(self.depthCompositing.outputs.images,False)
        self.colorMapping.inputs.map.set(self.colorMappingWidgets.outputs.map,False)
        self.colorMapping.inputs.range.set(self.colorMappingWidgets.outputs.range,False)
        self.colorMapping.inputs.channel.set(self.colorMappingWidgets.outputs.channel,False)

        self.shaderSSAO = ShaderSSAO()
        self.shaderSSAO.inputs.images.set(self.colorMapping.outputs.images,False)

        self.shadingWidgetsContainer.children = [
            NumberWidget(self.shaderSSAO.inputs.radius, range=[0,1,0.01]).widget,
            NumberWidget(self.shaderSSAO.inputs.samples, range=[1,256,1]).widget,
        ]

        self.annotation = Annotation()
        self.annotation.inputs.images.set(self.shaderSSAO.outputs.images,False)

        self.imageViewer = ImageViewer()
        self.imageViewer.inputs.images.set( self.annotation.outputs.images, False )
        self.imageViewer.inputs.container.set(self.imageContainer,False)

        IPython.display.display(self.globalContainer)

        self.inputs.path.set(path)

    def update(self):
        return 1
