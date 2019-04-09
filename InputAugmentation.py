#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
from scipy import ndimage
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import cv2
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_set = pd.read_csv(r"C:\Users\barbayrak\Desktop\The_Picnic_Hackathon_2019/train.tsv",header=0 , sep='\t')

IMAGE_DIRECTORY = './train/'
IMAGE_SIZE = 64

def tf_read_and_resize_images(imageNames,Y_labels):
    X_data = []
    Y = []
    tf.reset_default_graph()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        # Each image is resized individually as different image may be of different size.
        counter = 0 
        for img_name in tqdm(imageNames):
            try:
                img = mpimg.imread(IMAGE_DIRECTORY + img_name)
                if img_name.endswith("png"):
                    img = img[:,:,:3]
                elif img_name.endswith("jpeg"):
                    img = img.astype(np.float32)/255.0
                elif img_name.endswith("jpg"):
                    img = img.astype(np.float32)/255.0
                resized_img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)[:,:,:3]
                if(resized_img.shape[0] >= IMAGE_SIZE):
                    X_data.append(resized_img)
                    Y.append(Y_labels[counter])
            except Exception as e:
                print(e)
                pass
            counter = counter + 1
    X_data = np.asarray(X_data, dtype = np.float32) # Convert to numpy
    Y_data = np.array(Y)
    return X_data,Y_data

def central_scale_images(X_imgs,Y_labels, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    Y_scaled_labels = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        counter = 0
        for img_data in tqdm(X_imgs):
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
            for i in range(len(scales)):
                Y_scaled_labels.append(Y_labels[counter])
            counter += 1
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    Y_scaled_labels = np.array(Y_scaled_labels)
    return X_scale_data,Y_scaled_labels

def rotate_images(X_imgs,Y_labels):
    X_rotate = []
    Y_rotate_labels = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
                Y_rotate_labels.append(Y_labels[counter])
            counter += 1
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    Y_rotate_labels = np.array(Y_rotate_labels)
    return X_rotate,Y_rotate_labels

def flip_images(X_imgs,Y_labels):
    X_flip = []
    Y_fliped_labels = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
            Y_fliped_labels.extend([Y_labels[counter],Y_labels[counter],Y_labels[counter]])
            counter += 1
    X_flip = np.array(X_flip, dtype = np.float32)
    Y_fliped_labels = np.array(Y_fliped_labels)
    return X_flip,Y_fliped_labels

def add_gaussian_noise(X_imgs,Y_labels):
    gaussian_noise_imgs = []
    Y_gaussian_labels = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    counter = 0
    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
        Y_gaussian_labels.append(Y_labels[counter])
        counter += 1
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    Y_gaussian_labels = np.array(Y_gaussian_labels)
    return gaussian_noise_imgs,Y_gaussian_labels


######################################################################
IMAGE_SIZE = 64
#imgNames = train_set.iloc[:,0].values
#Y = train_set.iloc[:,1].values

#print(imgNames)
#X_resized_images,Y_resized_labels = tf_read_and_resize_images(imgNames[:100],Y[:100])
#print(X_resized_images.shape)
#print(Y_resized_labels.shape)
#np.save('X_resized_images_train.npy', X_resized_images[:100])
#np.save('Y_resized_labels_train.npy',Y_resized_labels[:100])

X_resized_images = np.load('X_resized_images_train.npy')
Y_resized_labels = np.load('Y_resized_labels_train.npy')

X_train,X_test,Y_train,Y_test = train_test_split(X_resized_images,Y_resized_labels, test_size = 0.15,shuffle=True)

np.save('X_train_resized.npy',X_train)
np.save('Y_train_resized.npy',Y_train)
np.save('X_test.npy',X_test)
np.save('Y_test.npy',Y_test)

X_scaled_images,Y_scaled_labels = central_scale_images(X_train,Y_train, [0.90, 0.75, 0.60])
X_rotated_images,Y_rotated_labels = rotate_images(X_train,Y_train)
X_fliped_images,Y_fliped_labels = flip_images(X_train,Y_train)
X_gaussian_images,Y_gaussian_labels = add_gaussian_noise(X_train,Y_train)

X_augmented_images = np.concatenate((X_train,X_scaled_images) , axis = 0) 
X_augmented_images = np.concatenate((X_augmented_images,X_rotated_images) , axis = 0) 
X_augmented_images = np.concatenate((X_augmented_images,X_fliped_images) , axis = 0) 
X_augmented_images = np.concatenate((X_augmented_images,X_gaussian_images) , axis = 0)

print(X_augmented_images.shape)

np.save('X_train.npy',X_augmented_images)

Y_augmented_labels = np.concatenate((Y_train,Y_scaled_labels) , axis = 0) 
Y_augmented_labels = np.concatenate((Y_augmented_labels,Y_rotated_labels) , axis = 0) 
Y_augmented_labels = np.concatenate((Y_augmented_labels,Y_fliped_labels) , axis = 0) 
Y_augmented_labels = np.concatenate((Y_augmented_labels,Y_gaussian_labels) , axis = 0)

print(Y_augmented_labels.shape)

np.save('Y_train.npy',Y_augmented_labels)




#imgNames = train_set.iloc[:,0].values
#Y = train_set.iloc[:,1].values

#X_resized_images = tf_read_and_resize_images(imgNames)
#X_fliped_images,Y_fliped_labels = flip_images(X_resized_images,Y)

#X_augmented_images = np.concatenate((X_resized_images,X_fliped_images) , axis = 0)
#Y_augmented_labels = np.concatenate((Y,Y_fliped_labels) , axis = 0) 

#np.save('X_augmented_images_train.npy',X_resized_images)
#np.save('Y_augmented_labels_train.npy',Y)


margin=50 # pixels
spacing =35 # pixels
dpi=100. # dots per inch
width = (200+200+2*margin+spacing)/dpi # inches
height= (180+180+2*margin+spacing)/dpi
left = margin/dpi/width #axes ratio
bottom = margin/dpi/height
wspace = spacing/float(200)
fig, axes  = plt.subplots(2,2, figsize=(width,height), dpi=dpi)
fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                    wspace=wspace, hspace=wspace)
for ax, im, name in zip(axes.flatten(),X_augmented_images[47856:,:,:,:], list("ABCD")):
    ax.axis('off')
    ax.set_title('restored {}'.format(name))
    ax.imshow(im)
plt.show()

margin=50 # pixels
spacing =35 # pixels
dpi=100. # dots per inch
width = (200+200+2*margin+spacing)/dpi # inches
height= (180+180+2*margin+spacing)/dpi
left = margin/dpi/width #axes ratio
bottom = margin/dpi/height
wspace = spacing/float(200)
fig, axes  = plt.subplots(2,2, figsize=(width,height), dpi=dpi)
fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom, 
                    wspace=wspace, hspace=wspace)
for ax, im, name in zip(axes.flatten(),X_test[3:,:,:,:], list("ABCD")):
    ax.axis('off')
    ax.set_title('restored {}'.format(name))
    ax.imshow(im)
plt.show()
