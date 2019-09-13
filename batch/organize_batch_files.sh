#!/bin/bash

#for dir in `ls data/old_tank_info`
#do 
#    if test -f "data/$dir/raw.mp4"
#    then
#        if test -f "data/old_tank_info/$dir/tank.dat"
#        then 
#            if test -f "data/$dir/tank.dat"
#            then
#                #echo "  Tank already located in data/$dir"
#                a=b 
#            else
#                cp data/old_tank_info/$dir/tank.dat data/$dir/.
#                echo "data/$dir"
#            fi
#        fi
#    fi
#done > trials_new.list
##done > trials_new.list
#
#for dir in `cat trials_new.list`
#do 
#    echo $dir
#    python3 cv-tracer/load_tank_txt.py ${dir}/raw.mp4 ${dir}/tank.dat
#done

for file in `ls data/*/tank.pik`
do 
    dir=`echo $file | rev | cut -d '/' -f 2 | rev`
    if test -f "data/$dir/traced.mp4"
    then
        a=b
    else
        echo "data/${dir}/raw.mp4"
    fi
done > trials_remain.list
