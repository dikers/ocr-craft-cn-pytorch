python3 create_lmdb_dataset.py --inputPath /home/ec2-user/tfc/031_ocr/text_renderer/output/default \
--gtFile /home/ec2-user/tfc/031_ocr/text_renderer/output/default/valid.txt \
--outputPath ../datasets/valid

python3 create_lmdb_dataset.py --inputPath /home/ec2-user/tfc/031_ocr/text_renderer/output/default  \
--gtFile /home/ec2-user/tfc/031_ocr/text_renderer/output/default/train.txt \
--outputPath ../datasets/train


