from .Core import *

import ipywidgets

class ParameterWidgets(Filter):

    def __init__(self):
        super().__init__()
        self.addInputPort("table", [])
        self.addInputPort("container", None)
        self.addInputPort("ignore", ['file','id','object_id_name'])
        self.addOutputPort("sql", "SELECT * FROM input")
        self.addOutputPort("composite_by_meta", (None,{}))
        self.widgets = []

    def widgetToSQL(self, wt):
        if not wt['B'].value:
            return ""

        v = None
        if wt['T'].value=='S':
            v = wt['S'].value
        elif wt['T'].value=='R':
            v = []
            for i in range(wt['R'].index[0],wt['R'].index[1]+1):
                v.append(wt['values'][i])
            v = tuple(v)
        else:
            v = wt['O'].value

        if type(v) is tuple:
            if len(v)==0:
                return '"' + wt['parameter'] + '" IN ()'
            elif len(v)==1:
                return '"' + wt['parameter'] + '" IN ("' + str(v[0]) + '")'
            else:
                return '"' + wt['parameter'] + '" IN ' + str(v)
        elif v.isnumeric():
            return '"' + wt['parameter'] + '"=' + v
        else:
            return '"' + wt['parameter'] + '"="' + v + '"'

    def generateWidgets(self):

        table = self.inputs.table.get()
        header = table[0]
        self.widgets = []

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.update()

        def on_type_change(change):
            if change['type'] == 'change' and change['name'] == 'index':
                wt = change['owner'].wt
                isRange = change['new']==0
                wt['O'].layout.display = 'none'
                wt['S'].layout.display = 'none'
                wt['R'].layout.display = 'none'
                wt[change['owner'].value].layout.display = 'flex'
                self.update()

        def on_enabled_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                disabled = not change['new']
                wt = change['owner'].wt
                wt['S'].disabled = disabled
                wt['O'].disabled = disabled
                wt['C'].disabled = disabled
                wt['T'].disabled = disabled
                self.update()

        def on_composite_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                for wt in self.widgets:
                    if wt['C'].lock:
                        return
                change['owner'].lock = True
                for wt in self.widgets:
                    if not wt['C'].lock:
                        wt['C'].value = False
                change['owner'].lock = False
                self.update()

        grid_template_areas = []
        for i in range(0,len(header)):
            if header[i].lower() in self.inputs.ignore.get():
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

            parameter = header[i]
            grid_template_areas_row = [parameter+'_E',parameter+'_C',parameter+'_T',parameter+'_W']
            grid_template_areas.append('"'+' '.join(grid_template_areas_row)+'"')

            wt = {}
            wt['parameter'] = parameter
            wt['values'] = o

            wt['B'] = ipywidgets.ToggleButton(
                value=True,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                description=header[i],
                layout=ipywidgets.Layout(width='auto', grid_area=grid_template_areas_row[0])
            )
            wt['B'].wt = wt
            wt['B'].observe(on_enabled_change)

            wt['C'] = ipywidgets.Checkbox(
                value=False,
                indent=False,
                layout=ipywidgets.Layout(width='auto', grid_area=grid_template_areas_row[1])
            )
            wt['C'].layout.width = 'auto'
            wt['C'].lock = False
            wt['C'].observe(on_composite_change)

            wt['T'] = ipywidgets.Dropdown(
              options=['S', 'O', 'R'],
              layout=ipywidgets.Layout(width='auto', grid_area=grid_template_areas_row[2])
            )
            wt['T'].wt = wt
            wt['T'].observe(on_type_change)

            wt['S'] = ipywidgets.SelectionSlider(
                options=o,
                value=o[0],
                indent=False
            )
            wt['S'].layout.padding = '0 0 0 0.4em'
            wt['S'].observe(on_change)

            wt['O'] = ipywidgets.SelectMultiple(
                options=o,
                rows=min(3,len(o)),
                value=[o[0]]
            )
            wt['O'].observe(on_change)
            wt['O'].layout.display = 'none'

            wt['R'] = ipywidgets.SelectionRangeSlider(
                options=o,
                index=(0,0),
                continuousupdate=False,
                callback_policy='mouseup',
            )
            wt['R'].observe(on_change)
            wt['R'].layout.display = 'none'

            self.widgets.append(wt)

        container = self.inputs.container.get()
        if container!=None:
          items = []
          for i,wt in enumerate(self.widgets):
              items.append(wt['B'])
              items.append(wt['C'])
              items.append(wt['T'])
              items.append(
                ipywidgets.HBox(
                  [wt['S'],wt['O'],wt['R']],
                  layout=ipywidgets.Layout(width='auto', grid_area=wt['parameter']+'_W')
                )
              )

          grid = ipywidgets.GridBox(
              children=items,
              layout=ipywidgets.Layout(
                grid_template_areas='\n'.join(grid_template_areas),
                align_items='center'
              )
          )

          container.children = [grid]

    def _update(self):

        table = self.inputs.table.get()
        header = table[0]

        sql = 'SELECT * FROM input WHERE '

        # compute widgets
        if len(self.widgets) < 1:
            self.generateWidgets()

        for i,wt in enumerate(self.widgets):
            wsql = self.widgetToSQL(wt)
            if len(wsql)>0:
                sql += wsql+ ' AND '

        sql += ' '

        self.outputs.sql.set(sql[:-6])

        composite_by_meta = (None,{})
        for i,wt in enumerate(self.widgets):
            if wt['C'].value:
                valueMap = {}
                for i,v in enumerate(wt['values']):
                    valueMap[v] = i
                composite_by_meta = (wt['parameter'],valueMap)
                break
        self.outputs.composite_by_meta.set(composite_by_meta)

        return 1
