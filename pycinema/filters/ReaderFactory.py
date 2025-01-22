import pycinema
import os
import glob

class ReaderFactory():

    def __init__(self, path):
        self.path = path
        self.type = "unknown"
        self.datatype = "none" 

        # from the path, determine what the data type is
        if os.path.isdir(self.path):
            self.type = "dir"
            if self.path.endswith(".cdb"):
                self.datatype = "cdb"
                # for now, no check for data.csv file 

            else:
                self.scan()

        elif os.path.isfile(self.path):
            if self.path.endswith("csv"):
                self.type = "file"
                self.datatype = "csv"

            elif self.path.endswith(".db"):
                self.type = "db"
                self.datatype = "dsi"

        return

    def create(self):
        reader = None
        if self.datatype == "cdb":
            reader = pycinema.filters.CinemaDatabaseReader()
        elif self.datatype == "csv":
            reader = pycinema.filters.CSVReader()
        elif self.datatype == "dsi":
            reader = pycinema.filters.DSIReader()

        return reader

    #
    # scan the files in a directory and try to determine what kind of dir it is
    #
    def scan(self):
        imiter = glob.iglob(self.path + "/*")
        for i in imiter:
            if i.endswith("h5"):
                self.datatype = "h5"
            else:
                self.datatype = "unknown"
                break
