#!/bin/bash

rsync -vhra /home/luca/Repos/rae/ "$1":~/rae --exclude-from='/home/luca/Repos/rae/.rsyncignore'
