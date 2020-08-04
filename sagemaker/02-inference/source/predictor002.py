# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import cv2
import sys
import time
import uuid
import json
import glob
import boto3
import flask
import errno
import shutil
import datetime
import argparse
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from collections import OrderedDict


import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from craft.craft import CRAFT
from craft.text_detection import text_detection_single 
from craft.text_detection import copyStateDict
from craft.merge_box import do_merge_box
from recognition.recognition import test_recong
from recognition.config import get_key_from_file_list
from recognition.utils import CTCLabelConverter, AttnLabelConverter
from recognition.model import Model
from recognition.dataset import RawCV2Dataset, RawDataset, AlignCollate
from recognition.textract import ConverToTextract



prefix = '/opt/ml/model'
s3_client = boto3.client('s3')
app = flask.Flask(__name__)

DEBUG = False
device = torch.device('cpu')

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
#--------------------------初始化-------------------------------------------------
def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="Textract 中文版本"
    )
    # Detection model  检测模型
    parser.add_argument('--trained_model', default='/opt/ml/code/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool,  help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--refine', default=False,  type=str2bool,  help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')


    # recognition model 识别模型
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--saved_model', default='/opt/ml/model/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth', help="path to saved_model to evaluation")
    
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=40, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', default=True,  help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--label_file_list', type=str, default='sample_data/chars.txt', help='label_file_list')

    return parser.parse_args()


def init_craft_net(args):
    print("-" * 50)
    print("init_craft_net")            
    net = CRAFT()     # initialize
    print('CRAFT Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    return net

def init_recognition_model(args):

    """ model configuration """
    print("-"* 50)
    print("init_recognition_model")

    file_list = args.label_file_list.split(',')
    args.character = get_key_from_file_list(file_list)  


    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()

    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
          args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
          args.SequenceModeling, args.Prediction)

    #model = torch.nn.DataParallel(model).to(device)

    if not os.path.exists(args.saved_model):
        print("[Error] model is not exists. [{}]".format(args.saved_model))
    # load model
    print('loading pretrained model from {}    device:{}'.format(args.saved_model, device))

    state_dict = torch.load(args.saved_model, map_location=lambda storage, loc: storage)

    new_state_dict = OrderedDict()
    key_length = len('module.')
    for k, v in state_dict.items():
        #print(k, v)
        name = k[key_length:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    alignCollate_demo = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    # predict
    model.eval()
    return model, alignCollate_demo, converter

#--------------------------初始化  结束-------------------------------------------------





@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = boto3.client('s3') is not None  # You can insert a health check here

    status = 200 if health else 404
#     status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference
    """
    
    data = None
    #解析json，
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        print("  ----------------  ", data)
        bucket = data['bucket']
        image_uri = data['image_uri']
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')    

    download_file_name = image_uri.split('/')[-1]
    #s3_client.download_file(bucket, image_uri, download_file_name)

    tt = time.mktime(datetime.datetime.now().timetuple())

    args_verbose = False
    args_output_dir = './'+ str(int(tt)) + download_file_name.split('.')[0]
    args_input_file = download_file_name
 
    print(" parse file path: {} ".format(args_input_file))
 
    inference_result = {'name':'hello'}
    _payload = json.dumps(inference_result)
 
    return flask.Response(response=_payload, status=200, mimetype='application/json')


#---------------------------------------
args = parse_arguments()
s3_client = boto3.client('s3')
craft_net, model, alignCollate_demo, converter = init_model()
#---------------------------------------

if __name__ == '__main__':
    app.run()