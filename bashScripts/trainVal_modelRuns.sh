#!/bin/bash
# A sample Bash script, by Catherine

#python classifier/train.py --config configs/exp_resnet50_2classes.yaml --exp_dir experiments --exp_name exp_resnet50_2classes_None  --train_folder train_resized --val_folder val_resized

#python classifier/train.py --config configs/exp_resnet50_3classes.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_None  --train_folder train_resized --val_folder val_resized

#python classifier/train.py --config configs/exp_resnet50_3classes_seq6hr.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seq6hr  --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seq12hr.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seq12hr --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seq24hr.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seq24hr --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_3classes_seqSliding.yaml --exp_dir experiments --exp_name exp_resnet50_3classes_seqSliding --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_2classes_seq6hr.yaml --exp_dir experiments --exp_name exp_resnet50_2classes_seq6hr  --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_2classes_seq12hr.yaml --exp_dir experiments --exp_name exp_resnet50_2classes_seq12hr --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_2classes_seq24hr.yaml --exp_dir experiments --exp_name exp_resnet50_2classes_seq24hr --train_folder train_resized --val_folder val_resized

python classifier/train.py --config configs/exp_resnet50_2classes_seqSliding.yaml --exp_dir experiments --exp_name exp_resnet50_2classes_seqSliding --train_folder train_resized --val_folder val_resized

