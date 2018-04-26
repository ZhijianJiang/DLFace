# DLFace
The final project of COMS 4995 Deep Learning for Computer Vision

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### How to train the ResNet + VGG model
* Download [VGG-16 pretrained weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) in
resnetface folder
* Download CACD2000 dataset and preprocessed them by Kaimao Yang
* Run the resnetface.py like the following

`python3 resnetface.py path/to/CACD2000/train path/to/CACD2000/val path/to/vgg-pretrained-h5-file`

## File Description

├── LFW.py by Zhijian Jiang offering functions to load LFW dataset   
├── README.md   
├── requirements.txt   
└── resnetface   
    ├── models.py by Zhijian Jiang offering functions to build models   
    └── resnetface.py by Zhijian Jiang to train the model   

## Code Reference
* http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
* https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py
* this course's homework5

## Authors

* Kaimao Yang
* Zhijian Jiang
* Yiyang Qian

## Acknowledgments


