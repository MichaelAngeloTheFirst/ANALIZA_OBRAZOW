# define and move to dataset directory
datasetdir = 'train'
import os
os.chdir(datasetdir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import the needed packages

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# shortcut to the ImageDataGenerator class
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

gen = ImageDataGenerator()

# iterator = gen.flow_from_directory(
#     os.getcwd(), 
#     target_size=(256,256), 
#     classes=('dein','red')
# )

# batch = iterator.next()
# # print(len(batch))

# def plot_images(batch):
#     imgs = batch[0]
#     labels = batch[1]
#     ncols, nrows = 4,8
#     fig = plt.figure( figsize=(ncols*3, nrows*3), dpi=90)
#     for i, (img,label) in enumerate(zip(imgs,labels)):
#       plt.subplot(nrows, ncols, i+1)
#       plt.imshow(img.astype(np.int))
#       assert(label[0]+label[1]==1.)
#       categ = 'dein' if label[0]>0.5 else 'red'
#       plt.title( '{} {}'.format(str(label), categ))
#       plt.axis('off')

plt.subplot(1,2,1)
plt.imshow(img.imread('Deinopis_Spider/dein.001.jpg'))
plt.subplot(1,2,2)
plt.imshow(img.imread('Red_Knee_Tarantula/red.001.jpg'))