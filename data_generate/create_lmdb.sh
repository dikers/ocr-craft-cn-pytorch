python3 create_lmdb_dataset.py --inputPath /home/ec2-user/tfc/031_ocr/text_renderer/output/default/ \
--gtFile /home/ec2-user/tfc/031_ocr/text_renderer/output/valid.txt \
--outputPath ../19/valid

python3 create_lmdb_dataset.py --inputPath /home/ec2-user/tfc/031_ocr/text_renderer/output/default/  \
--gtFile /home/ec2-user/tfc/031_ocr/text_renderer/output/train.txt \
--outputPath ../19/train


