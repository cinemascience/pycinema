import pycinema
import pycinema.filters

#
# pycinema testing script
#

def test_dsi():

    DSIReader_0 = pycinema.filters.DSIReader()
    DSIReader_0.inputs.path.set("data/asteroid_scalar_images.cdb/data.db", False)
    DSIReader_0.inputs.tablename.set("datacsv", False)
    DSIReader_0.update()
