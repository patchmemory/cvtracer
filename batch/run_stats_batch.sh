#!/bin/bash

remain="data/trials_remain.list"

process_tracks() {
  dir=$1
  file="$dir/raw.mp4"
  command="python3 cv-tracer/process.py $file"
  $command &> process.out
  command="python3 cv-tracer/statistics.py $file " 
  $command &> stats.out
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
  process_tracks $next_trial
  sleep 1s
done 
