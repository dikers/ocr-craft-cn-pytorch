#!/usr/bin/env bash
export PYTHONPATH=./

echo 'Train -------------------'
python3 train.py \
--train_data ../datasets/train \
--valid_data ../datasets/valid \
--select_data 'MJ-ST' \
--batch_ratio '0.8-0.2' \
--batch_size 192 \
--valInterval 200 \
--gpu_list '0,1,2,3' \
--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
