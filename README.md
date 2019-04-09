# Tensorflow/Keras - Picnic Image Classification Hackathon https://picnic.devpost.com

![alt text](https://res.cloudinary.com/devpost/image/fetch/s--3uci4Nf2--/c_limit,f_auto,fl_lossy,q_auto:eco,w_900/https://siliconcanals.nl/wp-content/uploads/2015/08/picnic-thumb.jpg)

# Hackathon Dataset
You can get hackathon dataset from here https://drive.google.com/file/d/1XSoOCPpndRCUIzz2LyRH0y01q35J7mgC/view?usp=sharing

# Challenge
Challenge is to classify images based on training dataset.

# Solution
<br/>
I tried using several Deep Learning Approaches for Image Classification<br/>
-2-Layer-CNN<br/>
-AlexNet<br/>
-ResNet25<br/>
-ResNet50<br/>
<br/>
For hyperparameter search , i didnt have much time so that i human-engineered some of the hyperparameters. For removing over fit i used Dropout and Early-Stopping approaches

#Results
I got best result with %71 Accuracy with ResNet50 Architecture.

#How to Run
<br/>
1)First download dataset and unzip it<br/>
2)Run InputAugmentation.py with changing its directory inside<br/>
3)Run ResNet.py with changing its directory inside<br/>
