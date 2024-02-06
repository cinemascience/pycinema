import pycinema
import os

print("get the script from the module ...")
print("--------------------------------------------------------")
print(pycinema.Core.getScriptPath('browse.py'))
print("")

print("get the script from the module without .py extension ...")
print("--------------------------------------------------------")
print(pycinema.Core.getScriptPath('browse'))
print("")

print("get the script from env var ...") 
print("--------------------------------------------------------")
os.environ['PYCINEMA_SCRIPT_DIR'] = "./testing/scripts"
print(pycinema.Core.getScriptPath('scripts-test.py'))
print("")

print("get the script from env var without .py extension ...")
print("--------------------------------------------------------")
os.environ['PYCINEMA_SCRIPT_DIR'] = "./testing/scripts"
print(pycinema.Core.getScriptPath('scripts-test'))
print("")

print("check for incorrect name ...") 
print("--------------------------------------------------------")
os.environ['PYCINEMA_SCRIPT_DIR'] = "./testing/scripts"
print(pycinema.Core.getScriptPath('junk'))
print("")

print("check for incorrect name in module directory ...") 
print("--------------------------------------------------------")
del os.environ['PYCINEMA_SCRIPT_DIR']
os.environ['PYCINEMA_SCRIPT_DIR'] = "./testscripts"
print(pycinema.Core.getScriptPath('junk'))
print("")

