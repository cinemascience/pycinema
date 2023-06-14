import time
import traceback

class Image():
    def __init__(self, channels=None, meta=None):
        self.channels = channels or {}
        self.meta = meta or {}

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

def isNumber(s):
    t = type(s)
    if t == int or t == float:
        return True
    if t == str:
        try:
            sf = float(s)
            return True
        except ValueError:
            return False
    return False

class Port():
    def __init__(self, name, value, parent, is_input = False):
        self.parent = parent
        self.name = name
        self.is_input = is_input
        self.connections = []
        self.time = -1
        if is_input:
            self.connections.append(self.parent)
        self._value = value

    def get(self):
        if isinstance(self._value, Port):
            return self._value.get()
        return self._value;

    def getTime(self):
        if isinstance(self._value, Port):
            return self._value.time
        return self.time;

    def set(self, value, update = True):
        if Filter._debug and update:
            print(type(self.parent).__name__+"->"+self.name, str(value)[:40])

        if self._value == value:
            return

        self.time = time.time()

        # if old value is a port stop listing for push events
        if isinstance(self._value, Port):
            self._value.connections.remove(self)
            Filter.trigger('connection_removed', [self._value,self])

        # replace old value with new value
        self._value = value

        # if new value is a port listen for push events
        if isinstance(self._value, Port):
            self._value.connections.append(self)
            Filter.trigger('connection_added', [self._value,self])

        # if value of a port was changed trigger update of listeners
        if update and not Filter._processing:
            self.parent.update()

class PortList():
    def __init__(self, filter, ports, areInputs=True):
        self.__ports = []
        for name in ports:
            self.__ports.append(name)
            setattr(self, name, Port(name, ports[name], filter, areInputs))
        return

    def ports(self):
        return self.__ports
    # def ports(self):
    #     return [p for p in dir(self) if not p.startswith('__') and p!='ports']


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

    def __del__(self):
        if Filter._debug:
            print('deleted',self)
        del Filter._filters[self]

    def _update(self):
        # needs to be overriden
        return 1

    def computeDAG(edges):
        for f in Filter._filters:
            edges[f] = set({})

        for f in Filter._filters:
            for name in f.outputs.ports():
                port = getattr(f.outputs, name)
                for listener in port.connections:
                    edges[f].add(listener.parent)

        return 1

    def computeTopologicalOrdering(self,edges):

      edgesR = {}
      for n in edges:
        for m in edges[n]:
          if not m in edgesR:
              edgesR[m] = 0
          edgesR[m]+=1

      L = []
      S = []
      for f in self._filters:
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

        Filter._processing = True

        dagt = time.time()
        edges = {}
        Filter.computeDAG(edges)
        filters = self.computeTopologicalOrdering(edges)

        if Filter._debug:
            for k, v in edges.items():
              print(k,v)
            print("DAG (%.2fs)" % (time.time()-dagt))

        for i,f in enumerate(filters):
            lt = f.time
            needsUpdate = False
            for name in f.inputs.ports():
                iport = getattr(f.inputs, name)
                if lt<iport.getTime():
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

    def help(self):
        print('Documentation Missing')
