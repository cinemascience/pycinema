#!/bin/sh

cinema data/sphere.cdb --verbose --logtofile &
cinema browse data/sphere.cdb --verbose --logtofile browse.txt &

# set env variable, so user area will be searched
export PYCINEMA_SCRIPT_DIR="testing/scripts"
cinema query data/sphere.cdb "SELECT * FROM input LIMIT 2" --verbose --logtofile query.txt &
