#!/bin/bash

cwd=/data1/cavefish/social/python/track
gitdir=`pwd`


files="Tracker.py WallDraw.py KinematicDataFrame.py Group.py DataDictionary.py"

cd $cwd
pwd

rsync -auvh $files $gitdir/.

cd $gitdir
