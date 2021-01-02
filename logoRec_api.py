#================================================================
#
#   File name   : logoRec_api.py
#   Author      : Venkatesh
#   Created date: 2020-07-23
#   Description : Object detection and recognition API for Logo image example
#
#================================================================

import os
from datetime import datetime
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import cv2
#import numpy as np
#import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_image
from yolov3.configs import *

from flask import Flask, request, render_template


app = Flask(__name__)

# input size 416x416
input_size = YOLO_INPUT_SIZE
# list of class names
logo_classes = TRAIN_CLASSES
# use tf_keras model weights
MODEL_PATH = "./checkpoints/yolov3_custom_logo"

UPLOAD_FOLDER = "./static/img"
OUTPUT_FOLDER = "./static/img_op"


## Logo Recognition Funtion
def logo_predict(img_path, output_folder_path, input_size, logo_classes, yolo_model):

    # # input size 416x416
    # input_size = YOLO_INPUT_SIZE
    # # list of class names
    # logo_classes = TRAIN_CLASSES

    image_path = img_path

    yolo = Create_Yolov3(input_size=input_size, CLASSES=logo_classes)
    yolo.load_weights(yolo_model)

    date_time = datetime.now().strftime("%d_%m_%y|%H:%M:%S")
    LogoFileName = os.path.join(output_folder_path, "logo_pred_" + date_time + ".jpg")

    # this function returns tuple with label values and image
    label, _ = detect_image(yolo, image_path, LogoFileName, input_size=input_size, show=True, CLASSES=logo_classes, rectangle_colors=(255,0,0))

    predicted_label = label[0]

    return (predicted_label, LogoFileName)




@app.route('/', methods=['GET', 'POST'])
def logo_recognition():
    #logo_names = []
    if request.method == 'POST':
        # check if the post request has the file part
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)

            print("Looking for logos in {}".format(image_location.split('/')[-1]))
            print(image_location)
            #print(image_location.split('/')[-1])

            # Remove logo image if already exist!
            if os.path.isdir(OUTPUT_FOLDER):
                for image_f in os.listdir(OUTPUT_FOLDER):
                    full_file_path = os.path.join(OUTPUT_FOLDER, image_f)
                    #os.chmod(full_file_path, stat.S_IWRITE)
                    os.chmod(full_file_path, 0o777)
                    os.remove(full_file_path)
            
            # function to recognize logo from image and save it as a logo image file with predicted label
            logo_pred_label, img_output_loc = logo_predict(image_location, OUTPUT_FOLDER, input_size, logo_classes, MODEL_PATH)
            #print(img_output_loc)
            img_output_loc = img_output_loc.split('/')[-1]

            return render_template('index.html', prediction = logo_pred_label, image_loc = image_location.split('/')[-1], image_op_loc = img_output_loc)
    return render_template('index.html', prediction = 'logo_name', image_loc = None, image_op_loc = None)
    #return 'OK'


if __name__ == "__main__":
    # When debug = True, code is reloaded on the fly while saved
    app.run(host='127.0.0.1', port='5001', debug=True)