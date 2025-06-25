import os

#
# pycinema testing script
#

def test_dsi():

    # check results
    assert os.path.isfile('DSIscratch/sphere.cdb/data.db')
