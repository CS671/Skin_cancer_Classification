import numpy as np
import cv2
import os
#import matplotlib.pyplot as plt
import keras
import tensorflow
from tensorflow.python.keras.utils import np_utils, to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.cross_validation import train_test_split
#import tkinter
import h5py
from keras.callbacks import ModelCheckpoint
seed = 43
np.random.seed(seed)

image_data = np.load("/home/dlagroup25/group8/image.npy")
dx_gt = np.load("/home/dlagroup25/group8/dx_file.npy")
image_data = image_data/255.


train_data, test_data, dx_train, dx_test = train_test_split(image_data, dx_gt, test_size=0.3, random_state=1234)
#train_data, val_data, dx_train, dx_val = train_test_split(train_data, dx_train, test_size=0.15, random_state=1234)
#dx_train, dx_test = train_test_split(dx_gt, test_size=0.2, random_state=seed)
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)
dx_train = np.asarray(dx_train)
dx_test = np.asarray(dx_test)

np.save('test_data.npy',test_data)
np.save('dx_test.npy', dx_test)
# dx_type_train = np.asarray(dx_type_train)
# dx_type_test = np.asarray(dx_type_test)
# sex_train = np.asarray(sex_train)
# sex_test = np.asarray(sex_test)
# loc_train = np.asarray(loc_train)
# loc_test = np.asarray(loc_test)

print('[INFO] Printing Data Shapes...')
print('train_data:', train_data.shape)
print('test_data:', test_data.shape)
print('dx_train:', dx_train.shape)
print('dx_test:', dx_test.shape)
# print('dx_type_train:', dx_type_train.shape)
# print('dx_type_test:', dx_type_test.shape)
# print('sex_train:', sex_train.shape)
# print('sex_test:', sex_test.shape)
# print('loc_train:', loc_train.shape)
# print('loc_test:', loc_test.shape)



def _model(inputs):
    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.5)(x)

    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    #x = Conv2D(128, (3, 3), padding="same")(x)
    #x = Conv2D(64, (3, 3), padding="same")(x)
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(3, 3))(x)
    #x = Dropout(0.5)(x)

  ## x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
   # x = MaxPooling2D(pool_size=(3, 3))(x)
    #x = Dropout(0.5)(x)
  # CONV => RELU => POOL
   

    #x = Conv2D(256, (3, 3), padding="same")(x)
    #x = Conv2D(256, (3, 3), padding="same")(x)
    #x = Conv2D(256, (3, 3), padding="same")(x)
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(3, 3))(x)
    #x = Dropout(0.5)(x)
# CONV => RELU => POOL
    #x = Conv2D(512, (3, 3), padding="same")(x)
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.3)(x)

# CONV => RELU => POOL
    #x = Conv2D(32, (3, 3), padding="same")(x)
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.5)(x)
    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    return x

input_size = (image_data.shape[1], image_data.shape[2], image_data.shape[3])
inputs = Input(shape=input_size)
model = _model(inputs)


epoch = 200
init_lr = 0.001
batchSize = 32

dx_output = Dense(units=7, activation='softmax', name='dx_out')(model)

designed_model = Model(inputs,dx_output)

#for layer in base_model.layers:
 #   layer.trainable = False


#dx_output = Dense(units=7, activation='softmax', name='dx_out')(model)
# dx_type_output = Dense(units=1, activation='relu', name='dx_type_out')(model)
# sex_output = Dense(units=1, activation='relu', name='sex_out')(model)
# localization_output = Dense(units=1, activation='relu', name='loc_out')(model)

#designed_model = Model(inputs, dx_output)#], sex_output, localization_output])

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


loss_weight = {'dx_out':0.25,
               'dx_type_out':0.25,
               # 'sex_out':0.25,
               # 'loc_out':0.25
               }

class_weights={
    0: 1.0,  # akiec
    1: 1.0,  # bcc
    2: 1.0,  # bkl
    3: 1.0,  # df
    4: 3.0,  # mel
    5: 1.0,  # nv
    6: 1.0,  # vasc
}
designed_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3),
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

designed_model.summary()



#designed_model = Model(inputs, dx_output)#], sex_output, localization_output])

#print("[INFO] compiling model...")
#designed_model.compile(optimizer=Adam(lr=init_lr),
     #                  loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
#designed_model.summary()

# tbCallBack = TensorBoard(log_dir='./Graph_UNET', histogram_freq=0, write_graph=True, write_images=True)


# H = designed_model.fit(x=image_data, y={'dx_out':dx_gt,
#                                         'dx_type_out':dx_type_gt,
#                                         'sex_out':sex_gt,
#                                         'loc_out':loc_gt},
#                        epochs=epoch, batch_size=25, verbose=1, validation_split=0.05)
filepath="./weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

designed_model.fit(train_data, dx_train , class_weight=class_weights,epochs=epoch,batch_size=batchSize, verbose=1, validation_split=0.15,callbacks=callbacks_list)
#history = designed_model.fit_generator(datagen.flow(train_data,dx_train, batch_size=64),validation_data = (val_data,dx_val), epochs = epoch,verbose = 1, steps_per_epoch=32)
print(test_data.shape)
print(dx_test.shape)


score = designed_model.evaluate(test_data, dx_test, verbose=1)

print("score:",score)

pred = designed_model.predict(test_data)
print(pred.shape)

pred_out = np.argmax(pred, axis=1)
test_out = np.argmax(dx_test,axis =1)
print(pred_out.shape)
print(pred_out)


conf = confusion_matrix(test_out, pred_out)
print(conf)


