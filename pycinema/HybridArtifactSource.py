from .Core import *
from .CinemaDatabaseReader import *
from .DatabaseQuery import *
from .ImageReader import *
from .ImageGeneratorCNN import *
from .Border import *

class HybridArtifactSource(Filter):

    def __init__(self):
        super(HybridArtifactSource, self).__init__()

        # input/output ports
        self.addInputPort("DBPath", "./")
        self.addInputPort("ForceDatabase", False)
        self.addInputPort("ForceEstimator", False)
        self.addInputPort("ModelPath", "./")
        self.addInputPort("Parameters", [])
        self.addOutputPort("Artifacts", [])

        # db path
        self.dbreader    = CinemaDatabaseReader();
        self.query       = DatabaseQuery();
        self.imageReader = ImageReader();

        # both
        self.border = Border();
        self.border.inputs.width.set(1);

        # estimator path
        self.mlfilter = ImageGeneratorCNN();

    def update(self):
        super().update()

        # depending upon state, load from different sources
        if self.inputs.ForceDatabase.get():
            self.loadFromDatabase()
        elif self.inputs.ForceEstimator.get():
            self.loadFromEstimator()
        else:
            numLoaded = self.loadFromDatabase();
            if numLoaded <= 0:
                self.loadFromEstimator();

        self.outputs.Artifacts.set(self.border.outputs.images);

        return 1;

    def loadFromDatabase(self):
        # read the database
        self.dbreader.inputs.path.set( self.inputs.DBPath.get() ); 

        # query for objects
        params = self.inputs.Parameters.get();
        self.query.inputs.table.set(self.dbreader.outputs.table);
        qstring = "SELECT * FROM INPUT WHERE phi = '{}' AND theta = '{}'".format(params[0], params[1])
        self.query.inputs.sql.set(qstring);

        # Read Data Products
        self.imageReader.inputs.table.set(self.query.outputs.table);

        # add a border
        self.border.inputs.color.set("black");
        self.border.inputs.images.set(self.imageReader.outputs.images);

        return len(self.imageReader.outputs.images.get());

    def loadFromEstimator(self):
        params = self.inputs.Parameters.get();

        self.mlfilter.inputs.Model.set(self.inputs.ModelPath.get(), False);
        self.mlfilter.inputs.Device.set('cpu',False);
        self.mlfilter.inputs.Params.set([params],False);
        self.mlfilter.inputs.VP.set(2,False);
        self.mlfilter.inputs.VPO.set(256,False);
        self.mlfilter.inputs.Channel.set(8,False);
        self.mlfilter.update();

        # add a border
        self.border.inputs.Color.set("red");
        self.border.inputs.Images.set(self.mlfilter.outputs.Images);

        return 1;
