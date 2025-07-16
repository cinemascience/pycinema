from pycinema import Filter, imageFromMatplotlibFigure, getColumnFromTable
import ast
import matplotlib.pyplot as plt
# use offscreen backend
import matplotlib as mpl
mpl.use('Agg')

class PlotTables(Filter):
    """!
    @brief A class that creates an image plot of columns in one or more tables 

    """

    def __init__(self):
        """!
        @brief Constructor for PlotTable
        @param table     The input table 
        @param xcol      The name of the column to use as x values
        @param ycol      The name of the column to use as y values
        @param dpi       The dpi of the resulting image
        @param attribute A list of attributes passed to the plot 
        """

        super().__init__(
          inputs={
            'tables': [],
            'xcol': '',
            'ycol': '',
            'combine': False,
            'dpi': 300,
            'attributes' : "{'marker':'o', 'linestyle':'-'}",
            'autocast': False
          },
          outputs={
            'images': []
          }
        )

    def _update(self):
      tables = self.inputs.tables.get()
      tables = tables if isinstance(tables,list) else [tables]

      res = []

      if self.inputs.combine.get():
        # create a plot
        figure = plt.figure()
        # create a dictionary from the input string that can be passed to the plot
        attr_dict = ast.literal_eval(self.inputs.attributes.get())
        for t in tables:
            table = t
            # Plot y vs x
            x = getColumnFromTable(table, self.inputs.xcol.get(), autocast=self.inputs.autocast.get()) 
            y = getColumnFromTable(table, self.inputs.ycol.get(), autocast=self.inputs.autocast.get()) 
    
            plt.plot(x, y, **attr_dict) 

        plt.xlabel(self.inputs.xcol.get())
        plt.ylabel(self.inputs.ycol.get())
        plt.title("Combined Ave. " + self.inputs.ycol.get() + " vs " + self.inputs.xcol.get())
        plt.grid(True)
    
        res.append(imageFromMatplotlibFigure(figure,self.inputs.dpi.get()))
    
        self.outputs.images.set(res)
      else:
        for table in tables: 
            # Plot y vs x
            x = getColumnFromTable(table, self.inputs.xcol.get(), autocast=self.inputs.autocast.get()) 
            y = getColumnFromTable(table, self.inputs.ycol.get(), autocast=self.inputs.autocast.get()) 
    
            # create a plot
            figure = plt.figure()
            # create a dictionary from the input string that can be passed to the plot
            attr_dict = ast.literal_eval(self.inputs.attributes.get())
            plt.plot(x, y, **attr_dict) 
            plt.xlabel(self.inputs.xcol.get())
            plt.ylabel(self.inputs.ycol.get())
            plt.title("Ave. " + self.inputs.ycol.get() + " vs " + self.inputs.xcol.get())
            plt.grid(True)
    
            res.append(imageFromMatplotlibFigure(figure,self.inputs.dpi.get()))
    
        self.outputs.images.set(res)

      return 1;
