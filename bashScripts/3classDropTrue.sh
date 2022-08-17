#!/bin/bash
# A sample Bash script, by Catherine




python classifier/train.py --config configs/exp_resnet50_3classes.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_None  --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seq6hr_dropTrue.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seq6hr_dropTrue  --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seq12hr_dropTrue.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seq12hr_dropTrue --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seq24hr_dropTrue.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seq24hr_dropTrue --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seqSliding_dropTrue.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seqSliding_dropTrue --train_folder train_resized --val_folder val_resized
