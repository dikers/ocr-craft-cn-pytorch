# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import sys
import boto3
import flask

import time
import datetime
import json
import numpy as np

prefix = '/opt/ml/model'
# model_path = os.path.join(prefix, 'model.h5')
s3_client = boto3.client('s3')
dirs = os.listdir( prefix )

# 
for file in dirs:
    print("-------------- file ", file)



# The flask app for serving predictions
app = flask.Flask(__name__)


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


if __name__ == '__main__':
    app.run()