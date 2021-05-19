## Introduction
Hi! I forked repository from [ultralytics](https://github.com/ultralytics/yolov3) version 7 to work on my undergraduate research project on KAIST Multispectral Pedestrian Dataset. I tried to apply multispectral images by merging RGB-based images and themal-based images. Please cite my work if you use my repository for your own project. For issues regarding the yolov3, please check the [ultralytics yolo v3](https://github.com/ultralytics/yolov3).

## Disclaimer
I stop working on this project as I finished my bachelor degree. If you have any questions, I would like to answer to the best of my knowledge.

## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ git clone https://github.com/jas-nat/PyTorch-YOLO-V3-KAIST.git
$ cd PyTorch-YOLO-V3-KAIST/
```
Installing by typing `$ sudo pip3 install -r requirements.txt` or `$ python -m pip install -r requirements.txt`
I also suggest to use virtual environments, such as [pyenv](https://github.com/pyenv/pyenv), docker, etc. to manage the libraries neatly.

## Preparation
**YOLOv3 label format**
Use `label_transform/kaist2coco-label-trans.py` to transform label format from `KAIST Multispectral Pedestrian Dataset` into YOLOv3 label format.
There is also a histogram plot to draw how many people are there in the datasets.

**Training configurations**

Configure `.cfg` and `.data` file in `config` (See examples)
  -channels: 4 for multispectral, 3 for RGB, and 1 for infrared only
  -location of training files and validations 
  
If you want to use pre-defined `.cfg` files, you can choose based on the followings
* 4 channels / multispectral `yolov3-spp-1cls-4channel.cfg`
* 3 channels / RGB `yolov3-spp-1cls.cfg`
* 1 channel / infrared `yolov3-spp-1cls-1channel.cfg`

I also disable HSV augmentation, since it does not work for 4 channels.

## Training

Run `python3 train_kaist_multi.py`
Some important arguments to put afterwards:
* `--weights ''` to train from scratch
* `--adam ` to use adam optimizer, the default is SGD
* `--img-size` to adjust the image size

## Validation
Don't forget to use the correct `.cfg` file.
Run `python3 test_kaist_multi.py`. 
It will load `weights/best.pt`. It will also produce the confidence detection results later

## Detection Examples
Don't forget to use the correct `.cfg` file. The image examples
Run `python3 detect_multi.py`. 
As it is impossible to produce bounding box detections on 4 channels images, you can choose to output RGB-based or thermal-based images. You can modify the codes inside `detect_multi.py`

## Reference Configurations
I trained this code for 200 epochs, 4 batch size using `NVIDIA RTX 2080 Ti`. It takes around 20 hours for daytime images and about 12 hours for nighttime images. 

## Publication
This project has been published to MDPI Journal: Sensors. Please take a look further [here](https://www.mdpi.com/1424-8220/21/7/2536). I humbly request you to cite our publication if you use this code as a reference.
```
AUTHOR = {Nataprawira, Jason and Gu, Yanlei and Goncharenko, Igor and Kamijo, Shunsuke},
TITLE = {Pedestrian Detection Using Multispectral Images and a Deep Neural Network},
JOURNAL = {Sensors},
VOLUME = {21},
YEAR = {2021},
NUMBER = {7},
ARTICLE-NUMBER = {2536},
URL = {https://www.mdpi.com/1424-8220/21/7/2536},
ISSN = {1424-8220},
DOI = {10.3390/s21072536}
}
```

