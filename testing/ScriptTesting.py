testing = True
if testing:
    import pytest
    from sys import platform

    if platform in ["linux","linux2"]:
        import pytest_xvfb
        @pytest.fixture(autouse=True, scope='session')
        def ensure_xvfb() -> None:
            if not pytest_xvfb.has_executable("Xvfb"):
                raise Exception("Tests need Xvfb to run.")

import filecmp
import subprocess
import os
import time

def test_scripttesting():
    # TODO: fire off a command line call that will create a log file. Include
    # a sleep, and then background the process
    newenv= os.environ.copy()
    subprocess.Popen(['cinema','data/sphere.cdb','--verbose','--logtofile'], env=newenv)

    #
    subprocess.Popen(['cinema','browse','data/sphere.cdb','--verbose','--logtofile','browse.txt'], env=newenv) 

    #
    newenv['PYCINEMA_SCRIPT_DIR'] = 'testing/scripts'
    subprocess.Popen(['cinema','query','data/sphere.cdb','SELECT * FROM input LIMIT 2','--verbose','--logtofile','query.txt'], env=newenv)

    # wait
    time.sleep(5)
    
    # compare files that are written by an outside script
    assert filecmp.cmp('log.txt', 'testing/gold/scripttesting/log.txt')
    assert filecmp.cmp('browse.txt', 'testing/gold/scripttesting/browse.txt')
    assert filecmp.cmp('query.txt', 'testing/gold/scripttesting/query.txt')
