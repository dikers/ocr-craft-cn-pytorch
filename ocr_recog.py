import argparse
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import argparse
import shutil
import errno
import time
import uuid
import fitz
import json
import glob
import cv2
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from craft.craft import CRAFT
from craft.text_detection import text_detection 
from craft.text_detection import copyStateDict
from recognition.recognition import test_recong
from recognition.config import get_key_from_file_list
from recognition.utils import CTCLabelConverter, AttnLabelConverter
from recognition.model import Model
from recognition.dataset import RawCV2Dataset, RawDataset, AlignCollate


DEBUG = True

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

class OcrMain(object):
    """

    python3 craft_label_parser.py  -i ../dataset/raw -o ../dataset/output

    """

    def __init__(self):
        args = self.parse_arguments()
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.image_files = []

    def parse_arguments(self):
        """
            Parse the command line arguments of the program.
        """

        parser = argparse.ArgumentParser(
            description="Textract 中文版本"
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            nargs="?",
            help="输出文件的本地路径",
            required=True
        )
        parser.add_argument(
            "-i",
            "--input_dir",
            type=str,
            nargs="?",
            help="输入文件路径",
            required=True
        )
        # Detection model  检测模型
        parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
        parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
        parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
        parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
        parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
        parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
        parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
        parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
        parser.add_argument('--refine', default=False, type=str2bool,  help='enable link refiner')
        parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

        
        # recognition model 识别模型
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=32, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=240, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        parser.add_argument('--sensitive', action='store_true', default=True,  help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
        parser.add_argument('--label_file_list', type=str, required=True, help='label_file_list')
        
        return parser.parse_args()

        
  
        
    
 
    
    def init_recognition_model(self):
        
        """ model configuration """

        #FIXME: 生成 关键字列表， 后期可以优化， 把char 提取出来 ， 目前是遍历训练集的文本
        file_list = self.args.label_file_list.split(',')
        self.args.character = get_key_from_file_list(file_list)  


        cudnn.benchmark = True
        cudnn.deterministic = True
        self.args.num_gpu = torch.cuda.device_count()

        if 'CTC' in self.args.Prediction:
            self.converter = CTCLabelConverter(self.args.character)
        else:
            self.converter = AttnLabelConverter(self.args.character)
        self.args.num_class = len(self.converter.character)

        if self.args.rgb:
            self.args.input_channel = 3
        model = Model(self.args)
        print('model input parameters', self.args.imgH, self.args.imgW, self.args.num_fiducial, self.args.input_channel, self.args.output_channel,
              self.args.hidden_size, self.args.num_class, self.args.batch_max_length, self.args.Transformation, self.args.FeatureExtraction,
              self.args.SequenceModeling, self.args.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % self.args.saved_model)
        print(" ---------------------------  device       ", device)
        #model.load_state_dict(torch.load(self.args.saved_model, map_location=device))
        model.load_state_dict(torch.load(self.args.saved_model, map_location=lambda storage, loc: storage))
        model = model.to(device)
        
        
        
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        self.AlignCollate_demo = AlignCollate(imgH=self.args.imgH, imgW=self.args.imgW, keep_ratio_with_pad=self.args.PAD)
        
        # predict
        model.eval()
        return model
    

    def main(self):
        time_start = time.time()
        # Argument parsing
        
        args = self.parse_arguments()
        self.args = args
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            pass

        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if not os.path.exists(args.input_dir):
            print("输入路径不能为空  input_dir[{}] ".format(args.input_dir))
            return
        
        if  os.path.exists(os.path.join(args.input_dir, 'ipynb_checkpoints')):
            shutil.rmtree(os.path.join(args.input_dir, 'ipynb_checkpoints'))
        
        
        model = self.init_recognition_model()
        
        demo_data = RawDataset(root=args.input_dir, opt=self.args)
        
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=int(self.args.workers),
            collate_fn=self.AlignCollate_demo, pin_memory=True)    
            
        test_recong(self.args, model, demo_loader, self.converter, device)   
        
        
        
    
        time_elapsed = time.time() - time_start
        print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



if __name__ == "__main__":
    ocrMain = OcrMain()
    ocrMain.main()