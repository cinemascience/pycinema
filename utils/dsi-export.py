import pycinema
import pycinema.filters
import os
import shutil
import glob

#
#
# This is a utility script to export dsi-compliant databases
# for a standard cinema database.
#
# The script reads in a cinema database, then exports a dsi-compatible
# data.db file that contains the information in the data.csv file
#
# The database can then be read in using a DSIReader filter, where the
# path input channel is the full path to the resulting data.db file
#

readers = []
exporters = []
#
# change this glob to collect paths to the cinema databases to augmented 
#
for p in glob.glob('*.cdb'):
    print(p)
    readers.append(pycinema.filters.CinemaDatabaseReader())
    exporters.append(pycinema.filters.ExportTableToDatabase())

    readers[-1].inputs.path.set(p, False)
    readers[-1].inputs.file_column.set("FILE", False)
    exporters[-1].inputs.table.set(readers[-1].outputs.table, False)
    exporters[-1].inputs.tablename.set("datacsv", False)
    exporters[-1].inputs.path.set(os.path.join(p, "data.db"), False)

# execute pipeline
readers[0].update()

