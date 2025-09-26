import pycinema
import pycinema.filters
import yaml

# pycinema settings
PYCINEMA = { 'VERSION' : '3.2.0'}

#
# A class that performs a recolor
#
class Recolor:

    def __init__(self, config):

        # filters
        self.CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
        self.ImageReader_0 = pycinema.filters.ImageReader()
        self.ColorMapping_0 = pycinema.filters.ColorMapping()
        self.ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
        self.DepthCompositing_0 = pycinema.filters.DepthCompositing()
        self.CinemaDatabaseWriter_0 = pycinema.filters.CinemaDatabaseWriter()

        # properties
        self.CinemaDatabaseReader_0.inputs.path.set(config['database']['path'], False)
        self.CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
        self.ImageReader_0.inputs.table.set(self.CinemaDatabaseReader_0.outputs.table, False)
        self.ImageReader_0.inputs.file_column.set("FILE", False)
        self.ImageReader_0.inputs.cache.set(True, False)
        self.ColorMapping_0.inputs.map.set(config['recolor']['colormap'], False)
        self.ColorMapping_0.inputs.nan.set(config['recolor']['nancolor'], False)
        self.ColorMapping_0.inputs.range.set(config['recolor']['channelrange'], False)
        self.ColorMapping_0.inputs.channel.set(config['recolor']['channel'], False)
        self.ColorMapping_0.inputs.images.set(self.ImageReader_0.outputs.images, False)
        self.ColorMapping_0.inputs.composition_id.set(-1, False)
        self.ShaderSSAO_0.inputs.images.set(self.ColorMapping_0.outputs.images, False)
        self.ShaderSSAO_0.inputs.radius.set(config['shadow']['radius'], False)
        self.ShaderSSAO_0.inputs.samples.set(config['shadow']['samples'], False)
        self.ShaderSSAO_0.inputs.diff.set(config['shadow']['diff'], False)
        self.DepthCompositing_0.inputs.images_a.set(self.ShaderSSAO_0.outputs.images, False)
        self.DepthCompositing_0.inputs.images_b.set(self.ShaderSSAO_0.outputs.images, False)
        self.DepthCompositing_0.inputs.depth_channel.set("depth", False)
        self.DepthCompositing_0.inputs.compose.set((None, {}), False)
        self.CinemaDatabaseWriter_0.inputs.images.set(self.DepthCompositing_0.outputs.images, False)
        self.CinemaDatabaseWriter_0.inputs.path.set(config['config']['output'], False)
        self.CinemaDatabaseWriter_0.inputs.ignore.set(['^id', '^camera', '^FILE'], False)
        self.CinemaDatabaseWriter_0.inputs.hdf5.set(False, False)

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
        self.CinemaDatabaseReader_0.update()
