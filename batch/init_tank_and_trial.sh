#!/bin/bash

#for file in `ls data/*/tank.pik`
for file in `cat trials_remain.list`
do 
    dir=`echo $file | rev | cut -d '/' -f 2 | rev`
    date=`echo $dir | cut -d '_' -f1`
    fishtype=`echo $dir | cut -d '_' -f2`
    n_ind=`echo $dir | cut -d '_' -f3 | cut -d 'n' -f2`
    echo "  data/$dir"
    echo "  $date $fishtype $n_ind"
    python3 cv-tracer/load_tank_txt.py data/${dir}/raw.mp4 data/${dir}/tank.dat
    python3 cv-tracer/prepare.py data/${dir}/raw.mp4 $n_ind $fishtype
done
