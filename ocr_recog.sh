#!/usr/bin/env bash
export PYTHONPATH=./

python3 ocr_recog.py  \
-i 'temp/output/test013/' \
-o 'temp/sample_result' \
--batch_size 64 \
--label_file_list  'datasets/chars.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model atte/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth



#--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \