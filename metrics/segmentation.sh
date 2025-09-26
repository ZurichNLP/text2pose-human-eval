#!/bin/bash

input_directory=$1
output_directory=$input_directory

mkdir -p $output_directory

for fullfile in "$input_directory"*.pose; do
# for fullfile in "$input_directory"*.imputed.pose; do
    filename=$(basename -- "$fullfile")
    filename="${filename%.*}"

    if ! [[ "$filename" == *"raw"* ]];then
        continue
    fi

    filename=$(echo "$filename" | cut -f 1 -d '.')

    output="$output_directory$filename.eaf"

    if [ -e "$output" ]
    then
        echo "$output exists, skipping ..."
    else
        echo "pose_to_segments --pose=$fullfile --elan=$output --model='model_E4s-1.pth'"
        pose_to_segments --pose=$fullfile --elan=$output --model='model_E4s-1.pth'
    fi
done