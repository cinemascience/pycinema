import time
import traceback
import pprint
import re
import numpy as np
from ast import literal_eval
import PIL
import io
import logging as log

CORE_NAN_VALUES = ['NaN', 'NAN', 'nan']

################################################################################
# general helper functions
################################################################################
def isURL(path):
    s = path.strip()
    if s.startswith("http") or s.startswith("HTTP"):
        return True
    else:
        return False

def isNumber(s):
    t = type(s)
    if t == int or t == float:
        return True
    else:
        # assume it is a string
        try:
            sf = float(s)
            return True
        except ValueError:
            return False
    return False

def imageFromMatplotlibFigure(figure):
  # Create image stream
  image_stream = io.BytesIO()
  figure.savefig(image_stream, format='png', dpi=100)
  image_stream.seek(0)

  # Parse stream to pycinema image
  rawImage = PIL.Image.open(image_stream)
  if rawImage.mode == 'RGB':
    rawImage = rawImage.convert('RGBA')
  image = Image({ 'rgba': np.asarray(rawImage) })
  return image

################################################################################
# table helper functions
################################################################################

# get the column index from a table, return -1 on failure
def getColumnIndexFromTable(table, colname):
    ID = -1

    colnames = table[0]
    if colname in colnames:
        ID = colnames.index(colname)

    return ID

# get a column of values from a table
def getColumnFromTable(table, colname, autocast=False, nan_remove=False, nan_replace=None, missing_remove=False, missing_replace=None):
    colID = getColumnIndexFromTable(table, colname)

    if colID == -1:
        log.Error("ERROR: no column named \'" + colname + "\'")
        return None

    else:
        # start with all values
        # these will be strings
        cleaned_column = [row[colID] for row in table[1:]]

        # remove values
        if nan_remove:
            cleaned_column = [x for x in cleaned_column if x not in CORE_NAN_VALUES]
        if missing_remove:
            cleaned_column = [x for x in cleaned_column if x != ''] 

        # replace values
        if nan_replace:
            i = 0
            while i < len(cleaned_column):
                if cleaned_column[i] in CORE_NAN_VALUES:
                    print("replacing " + cleaned_column[i])
                    cleaned_column[i] = nan_replace
                i += 1

        if missing_replace:
            i = 0
            while i < len(cleaned_column):
                if cleaned_column[i] == '': 
                    cleaned_column[i] = missing_replace
                i += 1

        # TODO: create a separate function call to determine column type
        if autocast:
            t = str
            for value in cleaned_column:
                if value != '' and value not in CORE_NAN_VALUES:
                    t = type(value)
                    if t == str: 
                        try:
                            si = int(value)
                            t = int
                        except ValueError:
                            try:
                                sf = float(value)
                                t = float
                            except ValueError:
                                t = str
                    break
            try:
                cleaned_column = np.asarray(cleaned_column, dtype=t)
            except ValueError:
                cleaned_column = []

        return cleaned_column


# get extent of the table (nRows,nCols)
def getTableExtent(table):
    try:
        nRows = len(table)
        if nRows < 1:
            return (0,0)
        nCols = len(table[1])
        for i in range(2,nRows):
            if len(table[i]) != nCols:
                return (nRows,-1)
        return (nRows,nCols)
    except:
        return (-1,-1)

################################################################################
# Image Class
################################################################################
class Image():
    def __init__(self, channels=None, meta=None):
        self.meta = meta or {}
        self.channels = channels or {}

    def __str__(self):
        result =  '{ PyCinemaImage:\n'
        result += '  meta: \n' + pprint.pformat(self.meta, indent=4)
        result += '  channels: \n' + pprint.pformat(self.channels, indent=4)
        result += '\n}\n'
        return result

    def __repr__(self):
        return self.__str__()

    def getChannel(self,regex):
        for c,data in self.channels.items():
            if c==regex:
                return data
        for c,data in self.channels.items():
            if re.search(regex, c, re.IGNORECASE):
                return data
        raise Exception('Channel Not Found')

    def copy(self):
        return Image(
            self.channels.copy(),
            self.meta.copy()
        )

    @property
    def shape(self):
        # get first channel
        for c in self.channels:
            return self.channels[c].shape
        return (0,0,0)

    @property
    def resolution(self):
        return self.shape[:2][::-1]

################################################################################
# Port Class
################################################################################
class Port():
    def __init__(self, name, value, parent, is_input = False):
        self.parent = parent
        self.name = name
        self.is_input = is_input
        self.connections = []
        self.time = -1
        self._value = value
        self.default = value
        t = type(value)
        if t == int:
            self.type = int
        elif t == float:
            self.type = float
        elif t == str:
            self.type = str
        else:
            self.type = object

        self._listeners = {}

        self.propagate_value = lambda v: self.trigger('value_set', v)

    def on(self, eventName, listener):
        if not eventName in self._listeners:
            self._listeners[eventName] = []
        self._listeners[eventName].append(listener)

    def off(self, eventName, listener):
        if not eventName in self._listeners:
            return
        self._listeners[eventName].remove(listener)

    def trigger(self, eventName, data):
        if not eventName in self._listeners:
            return
        for listener in self._listeners[eventName]:
            listener(data)

    def valueIsPort(self):
        return isinstance(self._value, Port)

    def valueIsPortList(self):
        return isinstance(self._value, list) and len(self._value)>0 and isinstance(self._value[0], Port)

    def get(self):
        if isinstance(self._value, Port):
            return self._value.get()
        elif self.valueIsPortList():
            result = []
            for port in self._value:
                result.append(port.get())
            return result
        return self._value

    def getTime(self):
        if isinstance(self._value, Port):
            return self._value.time
        return self.time

    def set(self, value, update = True):
        if Filter._debug:
            print(type(self.parent).__name__+"->"+self.name, str(value)[:40])

        if self._value == value:
            return

        self.time = time.time()

        # if old value is a port stop listing for push events
        if isinstance(self._value, Port):
            self._value.off('value_set', self.propagate_value)
            self._value.connections.remove(self)
            Filter.trigger('connection_removed', [self._value,self])
        elif self.valueIsPortList():
            for port in self._value:
                port.off('value_set', self.propagate_value)
                port.connections.remove(self)
                Filter.trigger('connection_removed', [port,self])

        # replace old value with new value
        self._value = value
        self.trigger('value_set', value)

        # if new value is a port listen for push events
        if isinstance(self._value, Port):
            self._value.on('value_set', self.propagate_value)
            self._value.connections.append(self)
            Filter.trigger('connection_added', [self._value,self])
        elif self.valueIsPortList():
            for port in self._value:
                port.on('value_set', self.propagate_value)
                port.connections.append(self)
                Filter.trigger('connection_added', [port,self])

        # if value of a port was changed trigger update of listeners
        if self.is_input and update and not Filter._processing:
            self.parent.update()

class PortList():
    def __init__(self, filter, ports, areInputs=True):
        self.__ports = {}
        for name in ports:
            setattr(self, name, Port(name, ports[name], filter, areInputs))
            self.__ports[name] = getattr(self,name)

    def ports(self):
        return self.__ports.items()

################################################################################
# Filter Class
################################################################################
class Filter():

    _debug = False
    _processing = False
    _ID_COUNTER = {}
    _filters = {}
    _listeners = {}

    @staticmethod
    def on(eventName,listener):
        if not eventName in Filter._listeners:
            Filter._listeners[eventName] = []
        Filter._listeners[eventName].append(listener)

    @staticmethod
    def off(eventName,listener):
        if not eventName in Filter._listeners: return
        id = listener.__repr__()
        Filter._listeners[eventName] = [l for l in Filter._listeners[eventName] if l.__repr__()!=id]

    @staticmethod
    def trigger(eventName, data):
        if not eventName in Filter._listeners:
            return
        for listener in Filter._listeners[eventName]:
            listener(data)

    def __init__(self, inputs={}, outputs={}):
        if Filter._debug:
            print('created',self)
        cls = self.__class__.__name__
        if cls  not in Filter._ID_COUNTER:
            Filter._ID_COUNTER[cls] = 0
        self.id = self.__class__.__name__+'_'+str(Filter._ID_COUNTER[cls])+''
        Filter._ID_COUNTER[cls] += 1

        self.inputs = PortList(self, inputs)
        self.outputs = PortList(self, outputs, False)
        self.time = -2
        self._filters[self] = self

        self.trigger('filter_created',self)

    def getInputFilters(self):
        filters = set({})
        for _, iPort in self.inputs.ports():
            if isinstance(iPort._value, Port):
                filters.add(iPort._value.parent)
            elif iPort.valueIsPortList():
                for port in iPort._value:
                    filters.add(port._value.parent)
        return filters

    def getOutputFilters(self):
        filters = set({})
        for _, oPort in self.outputs.ports():
            for iPort in list(oPort.connections):
                  filters.add(iPort.parent)
        return filters

    def delete(self):
        if Filter._debug:
            print('deleted',self)

        # reset dependencies
        for _, oPort in self.outputs.ports():
            for iPort in list(oPort.connections):
                iPort.set(iPort.default, False)
        for _, iPort in self.inputs.ports():
            iPort.set(iPort.default, False)

        # delete from filter list
        del Filter._filters[self]

        # signal filter destruction
        self.trigger('filter_deleted',self)

        # update pipeline
        self.update()

    def _update(self):
        # needs to be overriden
        return 1

    def computeDAG(edges):
        for f in Filter._filters:
            edges[f] = set({})

        for f in Filter._filters:
            for _, port in f.outputs.ports():
                for listener in port.connections:
                    edges[f].add(listener.parent)

        return 1

    def computeTopologicalOrdering(edges):

      edgesR = {}
      for n in edges:
        for m in edges[n]:
          if not m in edgesR:
              edgesR[m] = 0
          edgesR[m]+=1

      L = []
      S = []
      for f in Filter._filters:
          if f not in edgesR or not edgesR[f]:
              S.append(f)

      while len(S)>0:
          n = S.pop()
          L.append(n)
          for m in edges[n]:
              edgesR[m]-=1
              if edgesR[m]<1:
                  S.append(m)
      return L

    def update(self):
        if Filter._processing: return 0

        Filter._processing = True

        dagt = time.time()
        edges = {}
        Filter.computeDAG(edges)
        filters = Filter.computeTopologicalOrdering(edges)

        if Filter._debug:
            print("--------------------------------")
            print("DAG (%.2fs)" % (time.time()-dagt))
            for f in filters:
              print('  ',f,edges[f])
            print("--------------------------------")

        for i,f in enumerate(filters):
            lt = f.time
            needsUpdate = False
            for _, iPort in f.inputs.ports():
                if iPort.valueIsPortList():
                    for p in iPort._value:
                        if lt<p.getTime():
                            needsUpdate = True
                elif lt<iPort.getTime():
                    needsUpdate = True
            if f==self or needsUpdate:
                t0 = time.time()
                if Filter._debug:
                    print('PROCESS',f)
                try:
                    f._update()
                except Exception:
                    traceback.print_exc()
                    Filter._processing = False
                    return 0
                f.time = time.time()
                if Filter._debug:
                    print(" -> Done (%.2fs)" % (f.time-t0))
            elif Filter._debug:
                print('SKIP',f)

        Filter._processing = False
        return 1

    def help(self):
        print('Documentation Missing')
