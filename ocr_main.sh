#!/usr/bin/env bash
export PYTHONPATH=./

CUDA_VISIBLE_DEVICES=0
python3 ocr_main.py  \
-i '/home/ec2-user/tfc/031_ocr/ocr-craft-cn-pytorch/temp/input' \
-o '/home/ec2-user/tfc/031_ocr/ocr-craft-cn-pytorch/temp/output' \
--cuda==False  \
--label_file_list  '/home/ec2-user/datasets/DataSet/train_label.txt,/home/ec2-user/datasets/DataSet/valid_label.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model atte/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth \
--trained_model  './craft/weights/craft_mlt_25k.pth'

