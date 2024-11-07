import pycinema

def is_channel(images, channel):
    if images and channel:
        return channel in images[0].channels
    else:
        return False

#
# A class that performs a composite
#
class Composite:

    def __init__(self, config):
        self.reader = pycinema.filters.CinemaDatabaseReader()
        self.reader.inputs.path.set(config['database']['path'], False)
        self.reader.inputs.file_column.set(config['database']['filecolumn'], False)
        self.reader.update()

        self.elements = []
        for element in config['elements']:
            self.elements.append( Element(self.reader.outputs.table.get(), element, config) )

        # composite each element together
        curDepthCompositing = None
        aImages = self.elements[0].outputs.images.get()

        if len(self.elements) > 1:
            for element in self.elements[1:]:
                curDepthCompositing = pycinema.filters.DepthCompositing()
                curDepthCompositing.inputs.images_a.set(aImages, False)
                curDepthCompositing.inputs.images_b.set(element.outputs.images.get(), False)
                # remember output of compositor
                curDepthCompositing.update()
                aImages = curDepthCompositing.outputs.images
        else:
            curDepthCompositing = pycinema.filters.DepthCompositing()
            curDepthCompositing.inputs.images_a.set(aImages)

        # write the resulting images to a new cdb
        self.writer = pycinema.filters.CinemaDatabaseWriter()
        self.writer.inputs.images.set(curDepthCompositing.outputs.images, False)
        self.writer.inputs.path.set(config['config']['output'], False)
        self.writer.inputs.ignore.set(['^id', '^camera', '^FILE'], False)
        self.writer.inputs.hdf5.set(False, False)

        # settings
        if not 'verbose' in config['config']:
            self.verbose = False
        else:
            self.verbose = config['config']['verbose']

        if self.verbose:
            print("printing verbose output")
            print("  Reading: " + config['database']['path'])
            print("  Writing: " + config['config']['output'])

    def update(self):
        self.reader.update()

#
# create a pipeline that filters, reads and recolors a subset of images
#
class Element:

    def __init__(self, table, element, config):
        self.name = ""
        if 'name' in element:
            self.name = element['name']

        # query
        self.query  = pycinema.filters.TableQuery()
        self.query.inputs.table.set(table, False)
        # find the elements in the table with a query
        if 'elementlabel' in config['database']:
            self.query.inputs.sql.set("SELECT * from input WHERE " + config['database']['elementlabel'] + " = \'" + str(element['name']) + "\'", False)
        else:
            self.query.inputs.sql.set("SELECT * from input", False)

        # image reader
        self.imReader = pycinema.filters.ImageReader()
        self.imReader.inputs.table.set(self.query.outputs.table, False)
        self.imReader.inputs.file_column.set(config['database']['filecolumn'], False)
        self.imReader.inputs.cache.set(True, False)

        self.imReader.update()

        images = self.imReader.outputs.images.get()
        if not is_channel(images, element['channel']):
            print("ERROR: invalid channel \'" + element['channel'] + "\'")
            return

        if not is_channel(images, config['config']['depthchannel']):
            print("ERROR: invalid channel \'" + config['config']['depthchannel'] + "\'")
            return

        # colormap
        self.colormap = pycinema.filters.ColorMapping()
        self.colormap.inputs.map.set(element['colormap'], False)
        self.colormap.inputs.nan.set(element['nancolor'], False)
        self.colormap.inputs.range.set(element['channelrange'], False) # data range of the scalar values
        self.colormap.inputs.channel.set(element['channel'], False) # name of the channel that needs to be rendered
        self.colormap.inputs.images.set(self.imReader.outputs.images, False)
        self.colormap.inputs.composition_id.set(-1, False)
        # set output
        self.colormap.update()
        self.outputs = self.colormap.outputs

        # shadow
        if 'shadow' in config:
            if not 'state' in config['shadow'] or config['shadow']['state']:
                if not 'shadow' in config or config['shadow']['type'] == 'SSAO':
                    shadow = config['shadow']
                    self.shadow = pycinema.filters.ShaderSSAO()
                    self.shadow.inputs.images.set(self.colormap.outputs.images, False)
                    # other settings
                    if 'radius' in shadow:
                        self.shadow.inputs.radius.set(shadow['radius'])
                    else:
                        self.shadow.inputs.radius.set(0.03)
                    if 'samples' in shadow:
                        self.shadow.inputs.samples.set(shadow['samples'])
                    else:
                        self.shadow.inputs.samples.set(32)
                    if 'diff' in shadow:
                        self.shadow.inputs.diff.set(shadow['diff'])
                    else:
                        self.shadow.inputs.diff.set(0.5)
                    # set output
                    self.shadow.update()
                    self.outputs = self.shadow.outputs
