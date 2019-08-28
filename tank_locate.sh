#!/bin/bash

#list=tracking_rerun_20190708/trials_valid.list
#list=tracking_rerun_20190708/retank.list
list=$1

echo 
echo "  Looping over files in $list"
echo

for file in `cat $list`
do 
  echo
  echo $file
  echo
  python python/track/WallDraw.py $file -ck
  #python python/track/WallDraw.py $file
done
