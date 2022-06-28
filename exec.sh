#!/bin/bash

./sync.sh "$1"
echo ssh -tt $1 'cd rae; conda activate rae; python src/rae/run.py'
ssh -tt $1 'cd rae; conda activate rae; python src/rae/run.py'
