import argparse
import pycinema
import pycinema.filters
import sys

# pycinema settings
PYCINEMA = { 'VERSION' : '3.1.0'}

parser = argparse.ArgumentParser(description='convert Cinema float (hdf5) images to .png')
parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
parser.add_argument('--debug', action='store_true', help='set debug mode')
parser.add_argument('--thumbnail', action='store_true', help='write out a thumbnail of converted image')
parser.add_argument('--source', help='source cinema database')
parser.add_argument('--output', help='output cinema database')
parser.add_argument('--colormap', default="coolwarm", help='colormap name, from matplotlib color list')
parser.add_argument('--channel', help='define channel in source images to be colormapped')
parser.add_argument('--channelrange', nargs="*", type=float, default=[-1.0,1.0], help='define channel range')
parser.add_argument('--filecolumn', default="FILE", help='define FILE column name')
parser.add_argument('--depthchannel', default="depth", help='define DEPTH channel name') 
parser.add_argument('--nancolor', nargs="*", type=int, default=[0, 0, 0, 0]) 

args = parser.parse_args()

if args.debug:
    # enable debug output (e.g., timings)
    pycinema.Filter._debug = True

if args.verbose:
    print("converting: \'" + args.source   + "\' to \'" + args.output + "\'")
    print("colormap  : \'" + args.colormap + "\'")
    print("channel   : \'" + args.channel  + "\'")
    print()

# read scalar image cdb manifest
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
CinemaDatabaseReader_0.inputs.path.set(args.source, False)
CinemaDatabaseReader_0.inputs.file_column.set(args.filecolumn, False)

# read actual images
ImageReader_0 = pycinema.filters.ImageReader()
ImageReader_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.file_column.set(args.filecolumn, False)
ImageReader_0.inputs.cache.set(True, False)

# depth compositing
DepthCompositing_0 = pycinema.filters.DepthCompositing()
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.images_b.set([], False)
DepthCompositing_0.inputs.depth_channel.set(args.depthchannel, False)
DepthCompositing_0.inputs.compose.set((None, {}), False)

# Apply color mapping on some channel (here depth)
# name of the matplotlib color map: https://matplotlib.org/stable/users/explain/colors/colormaps.html
ColorMapping_0 = pycinema.filters.ColorMapping()
ColorMapping_0.inputs.map.set(args.colormap, False)
ColorMapping_0.inputs.nan.set(args.nancolor, False)
ColorMapping_0.inputs.range.set(args.channelrange, False) # data range of the scalar values
ColorMapping_0.inputs.channel.set(args.channel, False) # name of the channel that needs to be rendered
ColorMapping_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ColorMapping_0.inputs.composition_id.set(-1, False)

# write the resulting images to a new cdb
CinemaDatabaseWriter_0 = pycinema.filters.CinemaDatabaseWriter()
CinemaDatabaseWriter_0.inputs.images.set(ColorMapping_0.outputs.images, False)
CinemaDatabaseWriter_0.inputs.path.set(args.output, False)
CinemaDatabaseWriter_0.inputs.ignore.set(['^id', '^camera', '^FILE'], False)
CinemaDatabaseWriter_0.inputs.hdf5.set(False, False)

# execute pipeline
CinemaDatabaseReader_0.update()
