from .Core import *

import ipywidgets

class ParameterWidgets(Filter):

    def __init__(self):
        super().__init__()
        self.addInputPort("table", [])
        self.addInputPort("container", None)
        self.addOutputPort("sql", "SELECT * FROM input")
        self.widgets = []

    def generateWidgets(self):

        table = self.inputs.table.get()
        header = table[0]
        self.widgets = []

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.update()

        for i in range(0,len(header)):
            if header[i].lower() == 'file' or header[i].lower() == 'id':
                continue

            vdic = set()

            for j in range(1,len(table)):
                vdic.add(table[j][i])

            o = list(vdic)
            isListOfNumbers = isNumber(table[1][i])
            if isListOfNumbers:
                o = [float(x) for x in o]
            o.sort()
            if isListOfNumbers:
                o = [str(x) for x in o]

            w = None
            if header[i].startswith('object'):
                w = ipywidgets.SelectMultiple(
                    options=o,
                    value=o,
                    description=header[i]
                )
            else:
                o.insert(0,'ANY')
                w = ipywidgets.SelectionSlider(
                    options=o,
                    value=o[1],
                    description=header[i],
                    # callback_policy='mouseup',
                    # continuousupdate=False
                )

            if w != None:
              w.observe(on_change)
              self.widgets.append(w)

        container = self.inputs.container.get()
        if container!=None:
          container.children = self.widgets

    def update(self):

        table = self.inputs.table.get()
        header = table[0]

        sql = 'SELECT * FROM input WHERE '

        # compute widgets
        if len(self.widgets) < 1:
            self.generateWidgets()

        for i in range(0,len(self.widgets)):
            v = self.widgets[i].value
            if v == 'ANY':
                continue

            if type(v) is tuple:
                if len(v)==0:
                    sql += '"' + self.widgets[i].description + '" IN () AND '
                elif len(v)==1:
                    sql += '"' + self.widgets[i].description + '" IN ("' + str(v[0]) + '") AND '
                else:
                    sql += '"' + self.widgets[i].description + '" IN ' + str(v) + ' AND '

            elif v.isnumeric():
                sql += '"' + self.widgets[i].description + '"=' + v + ' AND '
            else:
                sql += '"' + self.widgets[i].description + '"="' + v + '" AND '

        if len(self.widgets) > 0:
            sql = sql[:-5]

        self.outputs.sql.set(sql)

        return 1
