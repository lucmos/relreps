#!/bin/bash

./sync.sh
ssh -tt erdos 'cd rae; conda activate rae; python src/rae/run.py'
