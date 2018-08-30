##############################################################################
##
## ship_detector_model.py
##
## @author: Matthew Cline
## @version: 20180801
##
## Description: Model to detect ships found in satellite images
##
##############################################################################

import numpy as np
import pickle
import os
import cv2
import argparse

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Input, Convolution2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from vis.visualization import visualize_saliency, visualize_cam, overlay

### Parse command line arguments ###
parser = argparse.ArgumentParser(description="Ship Detector")
parser.add_argument('--model_dir', type=str, help='path to the model checkpoints', default='models/keras')
parser.add_argument('--action', type=str, help='action to perform on theselected model (train, continue, test, info, saliency)', default='info')
parser.add_argument('--epochs', type=int, help='max number of epochs to train the model', default=35)
parser.add_argument('--batch', type=int, help='batch size to use in training', default=16)
parser.add_argument('--init_epoch', type=int, help='initial epoch to load the model from', default=1)
args = parser.parse_args()

model_dir = args.model_dir
action = args.action
user_epochs = args.epochs
user_batch = args.batch
user_init_epoch = args.init_epoch

tb = TensorBoard(log_dir=os.path.normpath(model_dir + "/logs"), write_graph=True)
ckpt = ModelCheckpoint(model_dir + "/ship_detector_{epoch}.hdf5", verbose=1, save_best_only=False)
active_callbacks = [tb,ckpt]

def build_cnn(image_size=None):
    image_size = image_size or (768, 768)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3,)

    img_input = Input(input_shape)

    x1 = Convolution2D(filters=8, kernel_size=(5,3), strides=(2,2), activation='linear', padding='same')(img_input)
    x1 = LeakyReLU()(x1)
    x2 = Convolution2D(filters=8, kernel_size=(3,5), strides=(2,2), activation='linear', padding='same')(img_input)
    x2 = LeakyReLU()(x2)
    
    x =  Concatenate()([x1,x2])

    x = Convolution2D(filters=16, kernel_size=5, strides=(2,2), activation='linear', padding='same')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(filters=32, kernel_size=5, strides=(2,2), activation='linear', padding='same')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(filters=32, kernel_size=5, strides=(2,2), activation='linear', padding='same')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(filters=32, kernel_size=3, strides=(2,2), activation='linear', padding='same')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(filters=32, kernel_size=3, strides=(2,2), activation='linear', padding='same')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(filters=32, kernel_size=3, strides=(2,2), activation='linear', padding='same')(x)
    x = LeakyReLU()(x)

    y = Flatten()(x)
    y = Dropout(0.5)(y)

    # y = Dense(1024)(y)
    # y = LeakyReLU()(y)
    # y = Dropout(0.5)(y)

    y = Dense(147456)(y)

    model = Model(inputs=img_input, outputs=y)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model


def load_images(imgs, test=False):
    imgList = []
    for imgId in imgs:
        if test:
            imgPath = os.path.normpath("Data/Test/Images/" + imgId)
        else:
            imgPath = os.path.normpath("Data/Train/Images/" + imgId)

        tempImg = cv2.imread(imgPath)
        tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2YUV)
        tempImg[:,:,0] = cv2.equalizeHist(tempImg[:,:,0])
        tempImg = cv2.cvtColor(tempImg, cv2.COLOR_YUV2BGR)
        tempImg = tempImg / 255.0
        imgList.append(tempImg)
    return np.array(imgList)


def output_to_image(out):
    pass


def run_length_to_labels(labels):
    labelList = []
    for label in labels:
        tempLabel = np.zeros(768*768)
        for pair in label:
            tempLabel[pair[0]: pair[0] + pair[1]] = 1.0
        labelList.append(tempLabel[1::4])
    return np.array(labelList)


def data_generator(imgs, labels, batch_size, test=False):
    L = len(imgs)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images(imgs[batch_start:limit], test)
            Y = run_length_to_labels(labels[batch_start:limit])

            if test:
                yield(X)
            else:
                yield(X,Y)

            batch_start += batch_size
            batch_end += batch_size


### Read in the data ###
print("Reading in the data from the pickle files...\n\n")
trainImages = pickle.load(open("trainImages.p", "rb"))
trainLabels = pickle.load(open("trainLabels.p", "rb"))
valImages = pickle.load(open("valImages.p", "rb"))
valLabels = pickle.load(open("valLabels.p", "rb"))

if action == 'train':

    ship_detector = build_cnn()

    history = ship_detector.fit_generator(
        generator=data_generator(trainImages, trainLabels, user_batch),
        validation_data=data_generator(valImages, valLabels, user_batch),
        steps_per_epoch=len(trainImages)/user_batch,
        validation_steps = len(valImages)/user_batch,
        epochs=user_epochs,
        verbose=1,
        callbacks=active_callbacks)
    print(history.history.keys)