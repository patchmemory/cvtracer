#!/bin/bash

remain="trials_remain.list"

process_tracks() {
  file=$1
  command="python3 cv-tracer/process.py $file" 
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
  process_tracks $next_trial/raw.mp4
  sleep 1s
done 
