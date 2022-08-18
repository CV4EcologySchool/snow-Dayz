#!/bin/bash
# eval bash, by Catherine

  #  parser = argparse.ArgumentParser(description='Train deep learning model.')
  #  parser.add_argument('--exp_dir', required=True, help='Path to experiment directory', default = "experiment_dir")
  #  parser.add_argument('--exp_name', required=True, help='Path to experiment folder', default = "experiment_name")
  #  parser.add_argument('--split', help='Data split', default ='train')
  #  parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet50_2classes.yaml')
  #  args = parser.parse_args()


#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_2classes_None --config configs/exp_resnet50_2classes.yaml

#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_3classes_seq6hr --config configs/exp_resnet50_3classes_seq6hr.yaml

#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_3classes_seq12hr --config configs/exp_resnet50_3classes_seq12hr.yaml

#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_3classes_seq24hr --config configs/exp_resnet50_3classes_seq24hr.yaml

#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_3classes_seqSliding --config configs/exp_resnet50_3classes_seqSliding.yaml

#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_2classes_seq12hr --config configs/exp_resnet50_2classes_seq12hr.yaml

#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_2classes_seq24hr --config configs/exp_resnet50_2classes_seq24hr.yaml


######################

##### run eval on OLD experiments



#rsync

#python classifier/evaluate.py --exp_name exp_resnet50_3classes_seqSliding --split test_resized --config configs/exp_resnet50_3classes_seqSliding.yaml


#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_3classes_seq6hr --config configs/exp_resnet50_3classes_seq6hr.yaml
#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_3classes --config configs/exp_resnet50_3classes.yaml
#python classifier/evaluate.py --exp_dir old_experiments --exp_name exp_resnet50_2classes_seqSliding --config configs/exp_resnet50_2classes_seqSliding.yaml
