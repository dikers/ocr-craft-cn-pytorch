#!/usr/bin/env bash
export PYTHONPATH=./


#--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn 

python3 demo.py \
--image_folder temp \
--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth \
