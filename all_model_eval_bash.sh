#!/bin/bash
# eval bash, by Catherine


python classifier/evaluate.py --exp_name exp_resnet50_3classes --split train --config configs/exp_resnet50_3classes.yaml

python classifier/evaluate.py --exp_name exp_resnet50_3classes_seq6hr --split train --config configs/exp_resnet50_3classes_seq6hr.yaml

python classifier/evaluate.py --exp_name exp_resnet50_3classes_seq12hr --split train --config configs/exp_resnet50_3classes_seq12hr.yaml

python classifier/evaluate.py --exp_name exp_resnet50_3classes_seq24hr --split train --config configs/exp_resnet50_3classes_seq24hr.yaml

python classifier/evaluate.py --exp_name exp_resnet50_3classes_seqSliding --split train --config configs/exp_resnet50_3classes_seqSliding.yaml

######################

##### run eval on OLD experiments



rsync

python classifier/evaluate.py --exp_name exp_resnet50_3classes_seqSliding --split test_resized --config configs/exp_resnet50_3classes_seqSliding.yaml