
"""
Description:
Abstract: "Ad Tech" is the use of digital technologies by vendors, brands,  and their agencies to target potential clients. One popular case is mining the Web to identify their logos.  I will show you how to do this by using PyTorch -- a popular deep learning framework in Python.
Description: In this talk. We will walk through the one significant use of deep learning for digital marketing and ad tech, the image recognition, which brands use to identify their potential clients, deliver personalized offerings and analyze the spending in the world of social media. The easiest way to identify the brand is by its logo.
The logo detection can be done by object detection models.  We will use PyTorch, a popular deep learning framework in Python, to build the model to identify a brand by its logo in an image.  Along the talk, we'll see the relative value of deep learning architectures-Deep Neural Network (DNN) and Convolutional Neural Network (CNN) , learn the effect of data size, augment the data when we don't have much, and use the transfer learning technique to improve the model.
"""


## Object Detection Notes ##

# Pytorch AI Ecosytem
- pytorch.org

#  facebookresearch/detectron2 
Detectron2 is FAIR's next-generation platform for object detection and segmentation. 
https://github.com/facebookresearch/detectron2

# albumentations - image augmentation library for object recognition
https://github.com/albumentations-team/albumentations
$ pip install albumentations

# YOLOv5 
https://github.com/ultralytics/yolov5



activate venv for code execution in Ubuntu Bash with Python-3.6.8
#  source pytext_nlp/bin/activate
mount C: drive folders in UbuntuBash
# cd /mnt/c/Users/Rathod/
# cd /mnt/d/V_Docs/Venky_Docs/

# lib_setup
- numpy
- pytorch
- opencv
- Pillow
- imageio

# install python packages
> pip install opencv-python  (4.2.0)
> pip install imageio   (2.8.0)

# Tensorflow 2.2.0 setup
> pip install tensorflow==2.*

  Successfully installed absl-py-0.9.0 astunparse-1.6.3 awscli-1.18.96 botocore-1.17.19 cachetools-4.1.1 certifi-2020.6.20 chardet-3.0.4 colorama-0.4.3 docutils-0.15.2 gast-0.3.3 google-auth-1.18.0 google-auth-oauthlib-0.4.1 google-pasta-0.2.0 grpcio-1.30.0 h5py-2.10.0 idna-2.10 importlib-metadata-1.7.0 jmespath-0.10.0 keras-preprocessing-1.1.2 markdown-3.2.2 numpy-1.19.0 oauthlib-3.1.0 opt-einsum-3.2.1 protobuf-3.12.2 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-2.24.0 requests-oauthlib-1.3.0 rsa-3.4.2 s3transfer-0.3.3 seaborn-0.10.1 tensorboard-2.2.2 tensorboard-plugin-wit-1.7.0 tensorflow-2.2.0 tensorflow-estimator-2.2.0 termcolor-1.1.0 wget-3.2 wrapt-1.12.1 zipp-3.1.0

# download Google Images using Python

$ pip install google_images_download

$ googleimagesdownload -k "hdfc bank logo" -l 10 -o /home/cprbiu/cv_projects/logoRecog_engine/Dataset/train/HDFC_Bank/

## LabelImg is a graphical image annotation tool. for Object Detection Task
https://github.com/tzutalin/labelImg
# Installation
Python 3 + Qt5 (Recommended)

  sudo apt-get install pyqt5-dev-tools
  sudo pip3 install -r requirements/requirements-linux-python3.txt
  make qt5py3
  python3 labelImg.py
  python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]


## Image Augmentation

You can use pip to install albumentations:

$ pip install albumentations
    Installing collected packages: imgaug, PyYAML, albumentations
    Successfully installed PyYAML-5.3.1 albumentations-0.4.5 imgaug-0.2.6


## Setup of NGrok in linux

$ sudo apt-get update
$ sudo apt-get install unzip wget
$ wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
$ unzip ngrok-stable-linux-amd64.zip
$ sudo mv ./ngrok /usr/bin/ngrok
$ ngrok

cprbiu@venkateshai:~$ ngrok
NAME:
   ngrok - tunnel local ports to public URLs and inspect traffic

DESCRIPTION:
    ngrok exposes local networked services behinds NATs and firewalls to the
    public internet over a secure tunnel. Share local websites, build/test
    webhook consumers and self-host personal services.
    Detailed help for each command is available with 'ngrok help <command>'.
    Open http://localhost:4040 for ngrok's web interface to inspect traffic.

# secure tunneling with ngrok
$ ngrok http -host-header=rewrite localhost:5001


## Yolov3 weights download
Download yolov3.weights if you don't have it:
(cv_ai) cprbiu@venkateshai:~/cv_projects/logoRecog_engine/TensorFlow-2.x-YOLOv3$ wget -P model_data https://pjreddie.com/media/file
s/yolov3.weights

--2020-07-09 17:15:45--  https://pjreddie.com/media/files/yolov3.weights
Resolving pjreddie.com (pjreddie.com)... 128.208.4.108
Connecting to pjreddie.com (pjreddie.com)|128.208.4.108|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 248007048 (237M) [application/octet-stream]
Saving to: ‘model_data/yolov3.weights’

yolov3.weights                   100%[=========================================================>] 236.52M  46.9KB/s    in 70m 6s  

2020-07-09 18:25:54 (57.6 KB/s) - ‘model_data/yolov3.weights’ saved [248007048/248007048]

# Check GPU support with TF
Test if TensorFlow works with gpu for you, in output should see similar results:

  import tensorflow as tf
  print(tf.__version__)
  tf.test.gpu_device_name()

  ## Training with Yolov3
  Error

  ValueError: not enough values to unpack (expected 3, got 0) #222

  The problem is because no object was detected during the first iteration. You have several ways to solve the problem: 
  1. do not execute the validation process during the first iteration; 
  2. Increase the number of validation set, or just copy some from the train set. 
  3. Load the pre-trained darknet weights.

## Tensorboard log check
$ tensorboard --logdir=log
(cv_ai) cprbiu@venkateshai:~/cv_projects/logoRecog_engine/LogoRec_TF2_YOLOv3$ tensorboard --logdir log

Now, you can train it and then evaluate your model
```
python train.py
tensorboard --logdir=log
```
Track training progress in Tensorboard and go to http://localhost:6006/:
<p align="center">
    <img width="100%" src="IMAGES/tensorboard.png" style="max-width:100%;"></a>
</p>

Test detection with `detect_mnist.py` script:
```
python detect_mnist.py
```


## Testing and Evaluation LogoRec model with augmented dataset - 20/07/2020
- total_val_loss:   0.28 (model almost overfitted I guess)
epoch:99 step:  467/471, lr:0.000001, giou_loss:   0.00, conf_loss:   0.00, prob_loss:   0.22, total_loss:   0.22
epoch:99 step:  468/471, lr:0.000001, giou_loss:   0.00, conf_loss:   0.00, prob_loss:   0.21, total_loss:   0.21
epoch:99 step:  469/471, lr:0.000001, giou_loss:   0.00, conf_loss:   0.00, prob_loss:   0.24, total_loss:   0.24
epoch:99 step:  470/471, lr:0.000001, giou_loss:   0.00, conf_loss:   0.00, prob_loss:   0.22, total_loss:   0.22
epoch:99 step:    0/471, lr:0.000001, giou_loss:   0.01, conf_loss:   0.00, prob_loss:   0.19, total_loss:   0.20
epoch:99 step:    1/471, lr:0.000001, giou_loss:   0.00, conf_loss:   0.00, prob_loss:   0.21, total_loss:   0.21


giou_val_loss:   0.01, conf_val_loss:   0.04, prob_val_loss:   0.23, total_val_loss:   0.28