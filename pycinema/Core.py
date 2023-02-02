import time

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

        # replace old value with new value
        self._value = value

        # if new value is a port listen for push events
        if isinstance(self._value, Port):
            self._value.connections.append(self)

        # if value of a port was changed trigger update of listeners
        if update and not Filter._processing:
            self.parent.update()

class PortList():
    def __init__(self):
        return

class Filter():

    _debug = False
    _processing = False

    def __init__(self):
        self.inputs = PortList()
        self.outputs = PortList()
        self.time = -2

    def addInputPort(self, name, value):
        setattr(self.inputs, name, Port(name, value, self, True))

    def addOutputPort(self, name, value):
        setattr(self.outputs, name, Port(name, value, self))

    def _update(self):
        # needs to be overriden
        return 1

    def computeDAG(self,edges):
        if self in edges:
            return 1

        edges[self] = set({})

        for name in [o for o in dir(self.outputs) if not o.startswith('__')]:
            port = getattr(self.outputs, name)
            for listener in port.connections:
                edges[self].add(listener.parent)
                listener.parent.computeDAG(edges)

        return 1

    def computeTopologicalOrdering(self,edges):
      L = []
      S = [self]
      edgesR = {}
      for n in edges:
        for m in edges[n]:
          if not m in edgesR:
              edgesR[m] = 0
          edgesR[m]+=1

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
        self.computeDAG(edges)
        filters = self.computeTopologicalOrdering(edges)
        if Filter._debug:
            print("DAG (%.2fs)" % (time.time()-dagt))

        for i,f in enumerate(filters):
            lt = f.time
            needsUpdate = False
            for name in [o for o in dir(f.inputs) if not o.startswith('__')]:
                iport = getattr(f.inputs, name)
                if lt<iport.getTime():
                    needsUpdate = True
            if i==0 or needsUpdate:
                t0 = time.time()
                if Filter._debug:
                    print('PROCESS',f)
                f._update()
                f.time = time.time()
                if Filter._debug:
                    print(" -> Done (%.2fs)" % (f.time-t0))
            elif Filter._debug:
                print('SKIP',f)

        Filter._processing = False

    def help(self):
        print('Documentation Missing')
