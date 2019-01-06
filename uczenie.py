from __future__ import print_function
'''
Created on 08.04.2017

@author: michael
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from scipy import misc
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import random as rand

def read_label_dict():
    path = '/home/michael/Eclipse/workspace/ProjektZespolowy/DataSet/labels/labels.txt'
    label_dict = dict()
    for line in open(path, 'r'):
        [label, letter] = line.split(', ', 1)
        label_dict[int(label) - 1] = letter.strip()
        
    return label_dict
        

def read_files():
    images = []
    labels = []
    path = '/home/michael/Eclipse/workspace/ProjektZespolowy/DataSet'
    for sample_dir in os.listdir(path=path):
        for image_path in glob.glob(path + '/' + sample_dir + '/*.png'):
            label = int(sample_dir[-2:])
           # if label == 51: # O
           #     label = 25
           # elif label == 55: # S
           #     label = 29
           # elif label == 59: # W
           #     label = 33
           # elif label == 58: # V
           #     label = 32
           # elif label == 62: # Z
           #     label = 36
           # elif label == 60: # X
           #     label = 34
            labels.append(label - 1)
            images.append(misc.imread(image_path))
        print(sample_dir, " read...")
    
    images = np.asarray(images)
    labels= np.asarray(labels)
    print('Importing done... ', images.shape)
    
    return (images, labels)

def learn(x_train, y_train, x_test, y_test):
    batch_size = 128
    num_classes = 62
    epochs = 10
    
    # input image dimensions
    img_rows, img_cols = 32, 32
       
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'
                     ,input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(),
                  metrics=['accuracy'])
                  
    #model.load_weights('new_cnn_weights.h5', by_name=False)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('cnn.h5')
    
#    y_predicted = model.predict(x_test)
#    label_dict = read_label_dict()
#    errors = np.zeros((62,2))
#    for (y_pred, y_real) in zip(y_predicted, y_test):
#        y_pred = np.argmax(y_pred)
#        y_real = np.argmax(y_real)
#        errors[y_real, 1] += 1
#        if y_pred != y_real:
#            errors[y_real, 0] += 1
#    for i in range(62):
#        print(label_dict[i], ' accuracy: ', errors[i, 0] / errors[i, 1])

def main():
    #label_dict = read_label_dict()
    images, labels = read_files()
    size = len(labels)
    idx = np.random.permutation(size)
    images,labels = images[idx], labels[idx]
    #choice = rand.randint(0, size)
    #print(label_dict[labels[choice]])
#    f, axarr = plt.subplots(3,10)
#    for i in range(10):
#        for j in range(3):
#            axarr[j, i].imshow(images[j * 10 + i, :, :], cmap='gray')
#            axarr[j, i].axis('off')
#    plt.show()
    learn(images[int(-0.8 * size):,:,:], labels[int(-0.8 * size):],
                  images[:int(-0.8 * size),:,:], labels[:int(-0.8 * size)])

   
if __name__ == '__main__':
    main()

