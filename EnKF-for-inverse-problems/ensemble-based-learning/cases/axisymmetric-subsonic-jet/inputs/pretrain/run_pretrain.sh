#!/bin/sh

. $HOME/ensemble-based-learning/code/DDTM/code/utilities/init

pretrain.py input.yaml | tee log.pretrain

cd ..
cp -r pretrain/results/w.1000 w.1000
