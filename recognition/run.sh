#!/usr/bin/env bash
export PYTHONPATH=./

echo 'Train -------------------'
python3 train.py \
--train_data ../output/train \
--valid_data ../output/valid \
--select_data / \
--batch_ratio 1 \
--batch_size 400 \
--valInterval 200 \
--gpu_list '0,1,2,3' \
--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn 
#--saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth \
