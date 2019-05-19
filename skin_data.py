import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils, to_categorical
from keras import Model
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import cv2



image_data_path = '/home/souvik/Documents/skin-cancer-mnist-ham10000/'
path_1 = image_data_path + '/' + 'HAM10000_images_part_1'
path_2 = image_data_path + '/' + 'HAM10000_images_part_2'
meta_data_path = image_data_path + '/' + 'HAM10000_metadata.csv'


df1 = pd.read_csv(meta_data_path, encoding='utf-8')

df = df1['image_id']

#df.values.tolist()

print(df[10])

path = "/home/souvik/Documents/skin-cancer-mnist-ham10000/image"

image = []
image_data = []

for i in range(len(df)):
   img = cv2.imread(path+'/'+df[i]+'.jpg')
   img = cv2.resize(img,(96,128))
   image.append(img)

image_data = np.asarray(image)
print(image_data.shape)

np.save('image.npy',image_data)

# metadata = metadata[:1000]

# metadata.drop(['lesion_id'])

cancer = df1['dx']
cancer = cancer.unique()
cancer_type = set(t for t in cancer)
cancer_labels = dict((cancer_type, label) for label, cancer_type in enumerate(cancer_type))
df1.dx = [cancer_labels[numbers] for numbers in df1.dx]



df = pd.DataFrame(df1)
loc = ['image_id', 'dx', 'dx_type', 'sex', 'localization']
df = df.loc[:,loc]

# print('[INFO] Printing Ground Truth Shapes...')

dx_gt = df['dx']
dx_gt = np.asarray(dx_gt)
dx_gt = np.expand_dims(dx_gt, axis=1)
dx_gt = to_categorical(dx_gt, dtype='int')
print(dx_gt.shape)
np.save('dx_file.npy',dx_gt)

