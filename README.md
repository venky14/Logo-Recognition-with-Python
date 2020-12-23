# Computer Vision R&D - Brand Logo Recognition TF_YOLOv3

Darknet YOLOv3 implementation in TensorFlow 2.x for Famous Brand Logo Detection and Recognition.

## Installation
First, clone or download this GitHub repository.
Install requirements and download pretrained weights:

```
Install package Dependencies
pip install -r ./requirements.txt

Download weights for tiny YOLOv3
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights

Download Logo Recognition models and save it in a respective folders.
wget -c https://mega.nz/folder/nLhCkBJb#vv4KYLAIfDxeRqLQ3sVgOw

```

## Quick start
Start using custom trained logo recognition model to test predictions on input image:
```
setup and execute faceRec_api.py
$ python logoRec_api.py

go to localhost "http://127.0.0.1:5001/"

upload image from 'img_sample_test' for testing
```

<br>
<p><img src="https://github.com/venky14/Logo-Recognition-with-Python/blob/master/img_sample_test/Screenshot%20from%202020-08-04%2017-13-43.png?raw=true"></p>
<br>
<p><img src="https://github.com/venky14/Logo-Recognition-with-Python/blob/master/img_sample_test/logo_jio1_detect.jpg?raw=true"></p>

## Model Trained With Following Famous Brand Logo Names
```
  TATA
  GODREJ
  RIL
  KOTAK_MAHINDRA_Bank
  Bank_Of_INDIA
  AXIS_Bank
  AIRTEL
  Bank_Of_BARODA
  JIO
  HDFC_Bank
  ICICI_Bank
  SBI_Bank
```



