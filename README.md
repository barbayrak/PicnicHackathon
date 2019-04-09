# PicnicHackathon
![alt text](https://res.cloudinary.com/devpost/image/fetch/s--3uci4Nf2--/c_limit,f_auto,fl_lossy,q_auto:eco,w_900/https://siliconcanals.nl/wp-content/uploads/2015/08/picnic-thumb.jpg)
Tensorflow - Picnic Image Classification Hackathon https://picnic.devpost.com

# Hackathon Dataset
You can get hackathon dataset from here https://drive.google.com/file/d/1XSoOCPpndRCUIzz2LyRH0y01q35J7mgC/view?usp=sharing

# Challenge
Challenge is to classify images based on training dataset.

# Solution
I tried using several Deep Learning Approaches for Image Classification
-2-Layer-CNN
-AlexNet
-ResNet25
-ResNet50

For hyperparameter search , i didnt have much time so that i human-engineered some of the hyperparameters. For removing over fit i used Dropout and Early-Stopping approaches

#Results
I got best result with %71 Accuracy with ResNet50 Architecture.

#How to Run
1)First download dataset and unzip it
2)Run InputAugmentation.py with changing its directory inside
3)Run ResNet.py with changing its directory inside
