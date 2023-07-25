from .Core import *
from .CinemaDatabaseReader import *
from .DatabaseQuery import *
from .ImageReader import *
from .DepthCompositing import *
from .ParameterWidgets import *
from .Annotation import *
from .ImageViewer import *
from .ShaderIBS import *
from .ShaderPointAO import *
from .ColorMappingWidgets import *
from .ColorMapping import *
from .NumberWidget import *

import IPython
import ipywidgets

class PointAOViewer(Filter):

    def __init__(self, path):
        super().__init__()

        # Layout
        self.parameterWidgetsContainer = ipywidgets.VBox()
        self.colorMappingWidgetsContainer = ipywidgets.VBox()
        self.shadingWidgetsContainer = ipywidgets.VBox()

        self.leftColumn = ipywidgets.VBox([
            ipywidgets.Accordion(children=[self.parameterWidgetsContainer]),
            ipywidgets.Accordion(children=[self.colorMappingWidgetsContainer]),
            ipywidgets.Accordion(children=[self.shadingWidgetsContainer])
        ]);
        self.leftColumn.layout.min_width = '28em'
        self.leftColumn.layout.max_width = '28em'
        self.leftColumn.children[0].set_title(0,'Parameters')
        self.leftColumn.children[1].set_title(0,'Color Mapping')
        self.leftColumn.children[2].set_title(0,'Shading')

        self.imageContainer = ipywidgets.HBox()
        self.globalContainer = ipywidgets.HBox([self.leftColumn,self.imageContainer])

        # Pipeline
        self.cinemaDatabaseReader  = CinemaDatabaseReader()

        self.parameterWidgets = ParameterWidgets()
        self.parameterWidgets.inputs.table.set(self.cinemaDatabaseReader.outputs.table,False)
        self.parameterWidgets.inputs.container.set(self.parameterWidgetsContainer,False)

        self.databaseQuery = DatabaseQuery()
        self.databaseQuery.inputs.table.set(self.cinemaDatabaseReader.outputs.table,False)
        self.databaseQuery.inputs.sql.set(self.parameterWidgets.outputs.sql,False)

        self.imageReader = ImageReader()
        self.imageReader.inputs.table.set(self.databaseQuery.outputs.table,False)

        self.depthCompositing = DepthCompositing()
        self.depthCompositing.inputs.images_a.set(self.imageReader.outputs.images,False)
        self.depthCompositing.inputs.composite_by_meta.set(self.parameterWidgets.outputs.composite_by_meta,False)

        self.colorMappingWidgets = ColorMappingWidgets()
        self.colorMappingWidgets.inputs.images.set(self.depthCompositing.outputs.images,False)
        self.colorMappingWidgets.inputs.container.set(self.colorMappingWidgetsContainer,False)

        self.colorMapping = ColorMapping()
        self.colorMapping.inputs.images.set(self.depthCompositing.outputs.images,False)
        self.colorMapping.inputs.map.set(self.colorMappingWidgets.outputs.map,False)
        self.colorMapping.inputs.range.set(self.colorMappingWidgets.outputs.range,False)
        self.colorMapping.inputs.channel.set(self.colorMappingWidgets.outputs.channel,False)
        self.colorMapping.inputs.nan.set(self.colorMappingWidgets.outputs.nan,False)

        self.shaderIBS = ShaderIBS()
        self.shaderIBS.inputs.images.set(self.colorMapping.outputs.images,False)



        self.shaderPointAO = ShaderPointAO()
        self.shaderPointAO.inputs.images.set(self.shaderIBS.outputs.images,False)
        self.shadingWidgetsContainer.children = [
           # NumberWidget(self.shaderIBS.inputs.radius, range=[0,1,0.01]).widget,
           # NumberWidget(self.shaderIBS.inputs.samples, range=[1,256,1]).widget,
           # NumberWidget(self.shaderIBS.inputs.silhouette, range=[0.01,1,0.01]).widget,
           # NumberWidget(self.shaderIBS.inputs.luminance, range=[0,2,0.01]).widget,
           # NumberWidget(self.shaderIBS.inputs.ambient, range=[0,2,0.01]).widget,

             # TODO: change the parameters below to parameters we actually need
            NumberWidget(self.shaderPointAO.inputs.radius, range=[0.1,10,0.01]).widget,
            NumberWidget(self.shaderPointAO.inputs.samples, range=[1,60,1]).widget,
            NumberWidget(self.shaderPointAO.inputs.scalers, range=[1,60,1]).widget,
            

        ]


        self.annotation = Annotation()
        self.annotation.inputs.images.set(self.shaderPointAO.outputs.images,False)

        self.imageViewer = ImageViewer()
        self.imageViewer.inputs.images.set( self.annotation.outputs.images, False )
        self.imageViewer.inputs.container.set(self.imageContainer,False)

        IPython.display.display(self.globalContainer)

        self.cinemaDatabaseReader.inputs.path.set(path)
