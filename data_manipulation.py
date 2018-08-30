##############################################################################
##
## data_manipulation.py
##
## @author: Matthew Cline
## @version: 20180731
##
## Description: Simple script to handle the training labels
##
##############################################################################

import numpy as np
import pandas as pd
import os
import pickle
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(suppress=True)
IMG_HEIGHT = 768
IMG_WIDTH = 768

def splitData(data, trainingSplit=0.8):
    training, test = np.split(data, [int(data.shape[0] * trainingSplit)])
    return training, test


def runLengthImg(run, height=IMG_HEIGHT, width=IMG_WIDTH):
    img = np.zeros(height * width)
    for pair in run:
        if len(pair) > 1:
            img[pair[0]:pair[0]+pair[1]] = 255
    return np.transpose(img.reshape([height, width]))


def bounding_box(rle, height=IMG_HEIGHT, width=IMG_WIDTH):
    ### Fix the mask encoding ###
    enc = np.reshape(rle.split(), [-1,2]).astype(int)
    mask = runLengthImg(enc, height, width)
    _, cont, _ = cv2.findContours(mask.astype(np.uint8).copy(), 1,2)
    cnt = cont[0]
    rect = cv2.minAreaRect(cnt)
    return rect


print("Reading in the data...\n\n")
fn = os.path.normpath("Data/Train/train_ship_segmentations.csv")
df = pd.read_csv(fn, dtype={"ImageId": object, "EncodedPixels": object})

print("Converting masks to rotated bounding boxes...")
newCol = []
for i in tqdm(range(df.shape[0])):
    if str(df.iloc[i,1]) != 'nan':
        newCol.append(bounding_box(df.iloc[i,1]))
    else:
        newCol.append([])
newCol = np.array(newCol)
df['BoundingBox'] = pd.Series(newCol)
print("\n\n")

# print(df)
# print(df["ImageId"])
# print("\n\n")



print("Consolodating the bounding boxes...\n\n")
df = df.groupby('ImageId')['BoundingBox'].apply(list).reset_index()
# print(tempDF)
# aggregation_functions = {'EncodedPixels': lambda x: " ".join(str(item) for item in x), 'BoundingBox': lambda x: list(x)}
# df = df.groupby(df['ImageId'], as_index=False).aggregate(aggregation_functions).reindex(columns=df.columns)
# print(df)
# print(df['ImageId'])
# print(df['ImageId'][2])
# print(df['EncodedPixels'][2])
# print(df['BoundingBox'][2])

# ### Combine the encoding for images with multiple ships ###
# print("Consolodating the bounding boxes...\n\n")
# aggregation_functions = {'EncodedPixels': lambda x: " ".join(str(item) for item in x)}
# df = df.groupby(df['ImageId'], as_index=False).aggregate(aggregation_functions).reindex(columns=df.columns)


# print("Re-encoding the bounding boxes...\n\n")
# newCol = []
# for i in range(df.shape[0]):
#     if df.iloc[i,1] != 'nan':
#         newCol.append(np.reshape(df.iloc[i,1].split(), [-1,2]).astype(int))
#     else:
#         newCol.append(np.array([]))
# newCol = np.array(newCol)
# df['newencoding'] = pd.Series(newCol)

print("Shuffling the data...\n\n")
data = df.sample(frac=1).reset_index(drop=True)

print("Splitting the data into train and val...\n\n")
train, val = splitData(data)
trainImages = np.array(train.iloc[:,0])
trainLabels = np.array(train.iloc[:,1])
valImages = np.array(val.iloc[:,0])
valLabels = np.array(val.iloc[:,1])

print("Dumping the data to pickle files...\n\n")
pickle.dump(trainImages, open("trainImages.p", "wb"))
pickle.dump(trainLabels, open("trainLabels.p", "wb"))
pickle.dump(valImages, open("valImages.p", "wb"))
pickle.dump(valLabels, open("valLabels.p", "wb"))


# for row in range(15):
#     origImg = plt.imread(os.path.normpath("Data/Train/Images/" + df.iloc[row,0]))
#     overImg = runLengthImg(df.iloc[row,2], 768, 768)
#     plt.imshow(origImg)
#     plt.imshow(overImg, alpha=0.25)
#     plt.show()
