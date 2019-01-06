import keras
from scipy import misc
import matplotlib.pyplot as plt
from keras import backend as K
from PIL import Image
import numpy as np
import glob
import sys
from keras.models import load_model

img_rows, img_cols = 32, 32

def read_label_dict():
    path = '/home/michael/Eclipse/workspace/ProjektZespolowy/DataSet/labels/labels.txt'
    label_dict = dict()
    for line in open(path, 'r'):
        [label, letter] = line.split(', ', 1)
        label_dict[int(label) - 1] = letter.strip()
    return label_dict

def read_files(path):
    images = []
    for image_path in sorted(glob.glob(path + '/*.png')):
        im = Image.open(image_path)
        im = im.resize((img_rows, img_cols))
        im.save(image_path)
        images.append(misc.imread(image_path))
    
    print("Images read.")
    
    images = np.asarray(images)  
    return images

def clasify(images):
    num_classes = 62
    # input image dimensions
    
    if K.image_data_format() == 'channels_first':
        images = images.reshape(images.shape[0], 1, img_rows, img_cols)
    else:
        images = images.reshape(images.shape[0], img_rows, img_cols, 1)
    
    images = images.astype('float32')
    images /= 255
     
    model = load_model('cnn.h5')
                     
    y_predicted = model.predict(images)
    label_dict = read_label_dict()
    output_letters = ""
    for y_pred in y_predicted:
        y_pred = np.argmax(y_pred)
        output_letters = output_letters + label_dict[y_pred]
    return output_letters
    
def show_letters(letters, images):
    vertical = 6
    horizontal = 15
    f, axarr = plt.subplots(vertical, horizontal)
    for j in range(vertical):
        for i in range(horizontal):
            axarr[j, i].axis('off')
            index = j * horizontal + i
            if index >= len(images):
                break
            axarr[j, i].imshow(images[index, :, :], cmap='gray')
    plt.show()

def main(path):
    images = read_files(path)
    letters = clasify(images)
    show_letters(letters, images)
    return letters
    
def run(path):
    return main(path)
   
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Arguments are missing: script.py folder_path")
        sys.exit(0)
    main(sys.argv[1])
