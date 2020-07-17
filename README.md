# ocr-craft-cn-pytorch

中文 文本区域检测和文本识别的pytorch实现



# CRAFT

Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)


[Github参考代码](https://github.com/clovaai/CRAFT-pytorch)

### Test instruction using pretrained model
- Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

* Run with pretrained model
``` (with python 3.7)
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]

```



#  文本识别
| [paper](https://arxiv.org/abs/1904.01906) | [training and evaluation data](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | [failure cases and cleansed label](https://github.com/clovaai/deep-text-recognition-benchmark#download-failure-cases-and-cleansed-label-from-here) | [pretrained model](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) | [Baidu ver(passwd:rryk)](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ) |


[Github参考代码](https://github.com/clovaai/deep-text-recognition-benchmark)



### 生成文本识别的数据

```
cd data_generate

sh generation.sh 'demo.txt'  0.2   ../output/

```


### 进行文本识别训练

```
cd recognition
sh run.sh

```
