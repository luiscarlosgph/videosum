#!/bin/bash
#
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   13 Jan 2023.

# Global variables defining the parameters of the experiment
tmp_file="/tmp/foo.jpg"
sizes=("4" "9" "16" "25" "36" "49" "64")
methods=("time" "inception" "uid" "scda")

# Input videos
videos=`find "$1" -name *.mp4`;

# Results
scores=("video,method,size,fid\n")

# Test 'time' method
for video in ${videos[@]}; do
  for method in ${methods[@]}; do
    for size in ${sizes[@]}; do
      rm -f $tmp_file
      echo "[INFO] Running method $method on $video with size $size ..."
      fid=`python3 -m videosum.run --input $video --output $tmp_file --nframes $size --height 1080 --width 1920 --algo $method --time-segmentation 1 --time-smoothing 0.0 --metric True | grep 'Summary FID' | cut -d" " -f4`
      scores+="$video,$method,$size,$fid\n"
    done
  done

  # Show results on the screen
  echo -e "\nRESULTS\n-------\n\n$scores"

  # Save results to file
  echo -e "$scores" > "$2"
done
