import pycinema
import pycinema.filters
import os
import shutil

scratchdir = 'DSIscratch' 

try:
    os.mkdir(scratchdir)

except FileNotFoundError:
    print(f"Error: Parent scratchdir not found.")
except FileExistsError:
    print(f"Error: Directory '{scratchdir}' already exists.")


# create ensemble 
for i in range(3):
    cdbdir = "sphere.cdb"
    newcdbdir = f"sphere{i:02}.cdb"
    source_dir = os.path.join("data", cdbdir)
    destination_dir = os.path.join(scratchdir, newcdbdir)

    try:
        shutil.copytree(source_dir, destination_dir)
        print(f"Directory '{source_dir}' successfully copied to '{destination_dir}'")
    except FileExistsError:
        print(f"Error: Directory '{destination_dir}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    cdbpath = destination_dir 

    # filters
    CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
    ExportTableToDatabase_0 = pycinema.filters.ExportTableToDatabase()

    # properties
    CinemaDatabaseReader_0.inputs.path.set(cdbpath, False)
    CinemaDatabaseReader_0.inputs.file_column.set("FILE", False)
    ExportTableToDatabase_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
    ExportTableToDatabase_0.inputs.tablename.set("datacsv", False)
    ExportTableToDatabase_0.inputs.path.set(os.path.join(cdbpath, "data.db"), False)

    # execute pipeline
    CinemaDatabaseReader_0.update()

