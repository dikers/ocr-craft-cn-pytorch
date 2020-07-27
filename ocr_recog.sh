#!/usr/bin/env bash
export PYTHONPATH=./

CUDA_VISIBLE_DEVICES=0
python3 ocr_recog.py  \
-i 'temp/output/test001/' \
-o 'temp/sample_result' \
--cuda==False  \
--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model atte/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth

