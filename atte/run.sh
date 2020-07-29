#!/usr/bin/env bash
export PYTHONPATH=./

echo 'Train -------------------'
python3 train.py \
--train_data ../datasets/train \
--valid_data ../datasets/valid \
--select_data 'MJ-ST-EN-19-SP' \
--batch_ratio '0.45-0.1-0.2-0.1-0.15' \
--batch_size 180 \
--valInterval 200 \
--gpu_list '0,1,2,3' \
--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth
