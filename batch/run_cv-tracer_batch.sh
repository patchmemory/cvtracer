#!/bin/bash

remain="trials_remain.list"

run_tracking() {
  file=$1
  f_str=`echo $file | rev | cut -d / -f1 | rev`
  n_ind=`echo $f_str | cut -d 'n' -f2 | cut -d '_' -f1`
  fish=`echo $f_str | cut -d '_' -f1`
  extra=""
  if   [ "$fish" == "Pa" ]
  then
    extra="-bs 17 -th 9"
  elif [ "$fish" == "Mo" ]
  then
    extra="-bs 17 -th 9"
  elif [ "$fish" == "Ti" ]
  then
    extra="-bs 17 -th 9"
  fi

  command="python3 cv-tracer/trace.py $file $extra -RGB" 
  if test -f $file
  $command
}


while true
do
  n_remain=`wc -l $remain | cut -d ' ' -f1`
  echo $n_remain
  if [ "$n_remain" -eq "0" ]
  then
    echo "none left"
    break
  fi

  next_trial=`head -1 $remain`
  echo $next_trial
  tail -n +2 $remain > ${remain}.tmp
  mv ${remain}.tmp $remain 
  run_tracking $next_trial
  sleep 1s
done 
