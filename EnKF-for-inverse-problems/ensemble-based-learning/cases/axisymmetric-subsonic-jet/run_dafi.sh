#!/bin/sh
# activate dafi
conda activate dafi
# source OpenFOAM
of1912
# run dafi
dafi dafi.in | tee dafi.log

#------------------------------------------------------------------------------
