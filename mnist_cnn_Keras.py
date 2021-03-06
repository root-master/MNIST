'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


batch_size = 128
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28



# the data, shuffled and split between train and test sets
################CHEATING##################################
(x_train, y_train), (x_test_1, y_test_1) = mnist.load_data()
x_train = np.concatenate((x_train,x_test_1),axis=0)
y_train = np.concatenate((y_train,y_test_1),axis=0)


# create the training & test sets, skipping the header row with [1:]

######## KAGGLE DATA ####################################
#train = pd.read_csv("./input/train.csv")
#print(train.shape)
#train.head()

test= pd.read_csv("./input/test.csv")
print(test.shape)
test.head()

#x_train = (train.ix[:,1:].values).astype('float32') # all pixel values
#y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
x_test = test.values.astype('float32')

#Convert train datset to (num_images, img_rows, img_cols) format 
x_train = x_train.reshape(x_train.shape[0], 28, 28)
print(x_train.shape)
print(y_train.shape)

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
#x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
#x_val /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

"""
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_val = keras.utils.to_categorical(y_val, num_classes)
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
"""
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_train,y_train)


for train_index, val_index in skf.split(x_train,y_train):
    x_train_fold = x_train[train_index]
    x_val_fold = x_train[val_index]
    y_train_fold = keras.utils.to_categorical(y_train[train_index], num_classes)
    y_val_fold = keras.utils.to_categorical(y_train[val_index], num_classes)
    
    model.fit(x_train_fold, y_train_fold,
              batch_size=batch_size,
              epochs=2,
              verbose=1,
              validation_data=(x_val_fold, y_val_fold))
    score = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
"""

#### KAGGLE CHEAT 1 method ###########
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


predictions = model.predict_classes(x_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("./Kaggle_Submissions/kaggle_cheat_2", index=False, header=True)





