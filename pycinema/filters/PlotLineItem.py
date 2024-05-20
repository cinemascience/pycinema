from pycinema import Filter, getTableExtent, getColumnFromTable

class PlotLineItem(Filter):

  def __init__(self):
    super().__init__(
      inputs={
        'table'     : [[]],
        'x'         : 'index',
        'y'         : '',
        'fmt'  : '',
        'style'  : {}
      },
      outputs={
        'item' : None
      }
    )

  def _update(self):
    table = self.inputs.table.get()
    tableExtent = getTableExtent(table)
    if tableExtent[0]<1 or tableExtent[1]<1:
      self.outputs.item.set(None)
      return 1

    xLabel = self.inputs.x.get()
    yLabel = self.inputs.y.get()

    header = table[0]
    if yLabel not in header:
      self.outputs.item.set(None)
      return 1
    yData = getColumnFromTable(table,yLabel,autocast=True)

    if xLabel!='index' and xLabel in header:
      xData = getColumnFromTable(table,xLabel,autocast=True)
    elif xLabel=='index':
      xData = range(0,len(yData))
    else:
      self.outputs.item.set(None)
      return 1

    res = {
      'x' : { 'label' : xLabel, 'data'  : xData },
      'y' : { 'label' : yLabel, 'data'  : yData },
      'fmt' : self.inputs.fmt.get(),
      'style': self.inputs.style.get()
    }
    self.outputs.item.set(res)

    return 1
