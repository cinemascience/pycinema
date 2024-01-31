# Workspace classes

A workspace is a script that has been abstracted to take some set of named inputs
and substitute them into a python string that is then run as a script.

The API for a workspace is:

def initializeScript(\*\*kwargs), where **kwargs** is a keyword/value dictionary
that looks like this:

```
    initializeScript(foo="some value", bar=123.45, filename="some/path")
```

The designer of the script can then check for expected input values and
substitute those into the script.
