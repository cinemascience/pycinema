from pycinema import Filter
try:
  from PySide6 import QtGui, QtCore, QtWidgets
except ImportError:
  pass

class Python(Filter):

    def __init__(self):

        self.watcher = QtCore.QFileSystemWatcher()
        self.watcher.fileChanged.connect(self.update)

        super().__init__(
          inputs={
            'inputs': None,
            'code': ''
          },
          outputs={
            'outputs': None
          }
        )

    def _update(self):

      code = self.inputs.code.get()

      watched_files = self.watcher.files()

      if len(code.splitlines())==1 and code.endswith('.py'):
        # update listeners
        if not code in watched_files:
          if len(watched_files)>0:
            self.watcher.removePaths(watched_files)
          self.watcher.addPath(code)

        # fetch code
        with open(code, "r") as script_file:
            code = script_file.read()
      elif len(watched_files)>0:
        self.watcher.removePaths(watched_files)

      variables = {}
      variables['inputs'] = self.inputs.inputs.get()
      variables['outputs'] = None

      exec(code,variables)
      output = variables['outputs']

      self.outputs.outputs.set(output)

      return 1;
