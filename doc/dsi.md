# DSI implementation notes

|![application](doc/img/DSI_test.png)|
| ---- |
|*Screen capture of DSIReader and DSICinemaDatabaseReader filters, showing them reading data converted from data.csv file to a data.db file, within
a cinema database.*|

There are two DSI readers:

- **DSIReader** reads a database file and returns a table. It makes no assumptions about what is in the database.
- **DSICinemaDatabaseReader** looks for a *data.db* file when given a path to a cinema database. Assumptions are that the *data.db* was exported with the **TableToDatabaseExport**  filter attached to a **CinemaDatabaseReader** filter. The result is that an *id* column has already been created, and the database path has been added to the paths in the *FILE* column. This means that if the database is moved, the **DSICinemaDatabaseReader** will no longer be able to find the image files. A re-export of the cinema database will fix this, but it is not automatic. 


