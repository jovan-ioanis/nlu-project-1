#!/bin/bash
#
# Testing showed that TF usually only uses four cores.
# So we set '-n 4' for both bsub and also main.python
#
# The command line argument to this bash script is passed to bsub,
# the second command line argument to main.py (just for convenience)
bsub -n 4 $1 "python main.py -n 4 $2"
