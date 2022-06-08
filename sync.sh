#!/bin/bash

rsync -vhra /home/luca/Repos/rae/ erdos:~/rae --exclude-from='/home/luca/Repos/rae/.rsyncignore'
