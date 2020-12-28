# Image classification using Alexnet Pytorch on CIFAR-10 dataset
## Install 
```bash
$ git clone git@github.com:uiui1999vn/image_classifier.git
```
## Requirements
```bash
$ pip install -r requirements.txt
```
## Inference
* Download the pretrained model [here](https://drive.google.com/file/d/1iNlzzDhIyKRv_uaQm2uiOSr3yQW5CW68/view?usp=sharing) and drop it in the `checkpoint` folder
```bash
$ python demo_img.py
```
## Training
```bash
$ python train.py
```
* Check point will be save in `checkpoint` folder when training is finished 
## NOTE
* **Edit hyparamenters for training and testing in config.py**


