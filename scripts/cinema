#!/usr/bin/env python

import sys
import argparse
import pycinema

# create help text and handle some command line arguments 
helptext = "\n\
\n\
examples: \n\
\n\
  cinema view some.cdb\n\
    run the \'view\' workspace on some.cdb\n\
\n\
  cinema explorer some.cdb\n\
    run the \'explorer\' workspace  on some.cdb\n\
\n\
"

# normal option parsing
parser = argparse.ArgumentParser( description="a command to access cinema viewers, filters and algorithms",
                                  epilog=helptext,
                                  formatter_class=argparse.RawDescriptionHelpFormatter )

parser.add_argument( "--version", action="version", version=pycinema.__version__)

# positional argument
parser.add_argument( "filenames", nargs='*' )

args = parser.parse_args()

# keep going if the command line args allow it
import pycinema.theater

pycinema.theater.Theater(args.filenames)

