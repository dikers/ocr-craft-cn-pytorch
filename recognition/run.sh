#!/usr/bin/env bash
export PYTHONPATH=./

echo 'Train -------------------'
python3 train.py \
--train_data ../output/train \
--valid_data ../output/valid \
--select_data / \
--batch_ratio 1 \
--batch_size 1024 \
--valInterval 100 \
--gpu_list '-1' \
--label_file_list  '../output/images/train/labels.txt,../output/images/valid/labels.txt' \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC