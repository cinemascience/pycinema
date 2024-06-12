import os
import re

raw = open('../pycinema/filters/__init__.py', 'r').read()
lines = raw.splitlines()

def get_filter_name(string, start_enclosing='from .', end_enclosing=' import *'):
    pattern = re.escape(start_enclosing) + r'(.*?)' + re.escape(end_enclosing)
    match = re.search(pattern, string)
    return match.group(1).strip() if match else None

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

directory = './'
files = list_files(directory)
test_strings = []
for f in files:
  test_strings.append(open(f, 'r').read())

filters = [get_filter_name(line) for line in lines]

untested_filters = [
  f for f in filters if all(f not in test_string for test_string in test_strings)
]

print(untested_filters)

