#!/usr/bin/env bash

# Parse parameters
type=$1
code=$2
data=$3
dest=$4

# Optional time argument
if [[ -z $5 ]]; then
    time='3-00:00:00'
else
    time=$5
fi

# Count the number of training images
n_images=$(ls "$data/train/images/" | wc -l)

# Extract the directory name of the dataset
dir_name=$(echo $data | sed 's|.*/||')

# Create the SLURM scripts for an increasing number of training images
for ((i = 1 ; i <= n_images ; i++)); do
    dest_dir="$dest/$dir_name-$i/"
    weight="$dest/$dir_name-$i/"$type"_model.pth"

    # Create the SLURM script
    python3 eval.py --dest $dest_dir --code $code --data $data --weight $weight \
    --stat $dest --time $time --type $type --size $i
done
