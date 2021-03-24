#!/usr/bin/env bash

# Parse parameters
type=$1
code=$2
data=$3
dest=$4

# Optional epoch argument
if [[ -z $5 ]]; then
    epoch=100
else
    epoch=$5
fi

# Optional time argument
if [[ -z $6 ]]; then
    time='3-00:00:00'
else
    time=$6
fi

# Count the number of training images
n_images=$(ls "$data/train/images/" | wc -l)

# Extract the directory name of the dataset
dir_name=$(echo $data | sed 's|.*/||')

# Create the SLURM scripts for an increasing number of training images
for ((i = 1 ; i <= n_images ; i++)); do
    # Create the result directory
    dest_dir="$dest/$dir_name-$i/"
    mkdir -p $dest_dir

    # Create the SLURM script
    python3 train.py --dest $dest_dir --code $code --data $data --epoch $epoch \
    --size $i --time $time --partition 'tesla' --type $type
done
