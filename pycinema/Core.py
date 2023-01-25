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
    def __init__(self, name, value, parent, isInput = False):
        self.parent = parent
        self.name = name
        self._listeners = []
        if isInput:
            self._listeners.append(self.parent)
        self._value = value

    def get(self):
        if isinstance(self._value, Port):
            return self._value.get()
        return self._value;

    def set(self, value, update = True):
        # if old value is a port stop listing for push events
        if isinstance(self._value, Port):
            self._value._listeners.remove(self)

        # replace old value with new value
        self._value = value

        # if new value is a port listen for push events
        if isinstance(self._value, Port):
            self._value._listeners.append(self)

        # if value of a port was changed trigger update of listeners
        if update:
            for listener in self._listeners:
                if isinstance(listener, Port):
                    listener.parent.update()
                elif isinstance(listener, Filter):
                    listener.update()

class PortList():
    def __init__(self):
        return

class Filter():
    def __init__(self):
        self.inputs = PortList()
        self.outputs = PortList()

    def addInputPort(self, name, value):
        setattr(self.inputs, name, Port(name, value, self, True))

    def addOutputPort(self, name, value):
        setattr(self.outputs, name, Port(name, value, self))

    def update(self):
        # needs to be overriden
        return 1

    def help(self):
        print('Documentation Missing')
