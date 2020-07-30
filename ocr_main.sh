#!/usr/bin/env bash
export PYTHONPATH=./

python3 ocr_main.py  \
-i 'temp/input' \
-o 'temp/output' \
--cuda==False  \
--label_file_list  'datasets/chars.txt' \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--saved_model atte/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth \
--trained_model  'craft/weights/craft_mlt_25k.pth'

