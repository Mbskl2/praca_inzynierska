from __future__ import print_function
'''
Created on 08.04.2017

@author: michael
'''

from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import sys
import numpy as np
import glob
import math

max_steps_in_direction = 4     

def find_cut_lines(image_array):
    height, width = image_array.shape
    borders = []
    
    for w in range(width):
        current_w = w
        steps_right = 0
        steps_left = 0
        h = 1
        while h < (height + 1): # for minus indexing
            if image_array[-h, current_w] == 0:
                break
            if h == height:
                borders.append((w + current_w) / 2.0) 
                break           
            elif ((image_array[-(h + 1), current_w] == 0) and
                current_w < width - 1 and     
                image_array[-h, current_w + 1] != 0 and           
                steps_right < max_steps_in_direction):
                # If step right is possible, do one.             
                current_w += 1 
                steps_right += 1
            elif ((image_array[-(h + 1), current_w] == 0) and
                current_w >= 1 and     
                image_array[-h, current_w - 1] != 0 and           
                steps_left < max_steps_in_direction + steps_right):
                # If step left is possible, do one.             
                current_w -= 1 
                steps_left += 1
            else:
                h += 1
                
#    for line in borders:
#        image_array[:, int(line)] = 0
#    plt.imshow(image_array, cmap='gray')
#    plt.show()
    return borders_to_cut_lines(borders)
        
        
def borders_to_cut_lines(borders):
    border_groups = divide_into_groups_of_adjecent_borders(borders)
    cut_lines = []
    for group in border_groups:
        cut_lines.append(int(sum(group) / len(group)))
    return cut_lines
        
def divide_into_groups_of_adjecent_borders(borders):
    border_groups = []
    for i in range(len(borders)):
        if i == 0: 
            border_groups.append([borders[0]])
            continue
        for group in border_groups:
            if group[-1] < borders[i] <= (group[-1] + 1):
                group.append(borders[i])
                break
        else:
            border_groups.append([borders[i]])
    return border_groups

def sharpen_image(image, alpha):
    filter = ndimage.gaussian_filter(image, 1)
    sharpened_image = image + alpha * (image - filter)
    return sharpened_image

def threshold_image(image, threshold):
    thresholded_image = np.where(image < threshold, 0, 255)
    return thresholded_image

def preprocess_image(image):
    alpha = 5
    image = sharpen_image(image, alpha)
    
    threshold = 50
    image = threshold_image(image, threshold)
    
    return image

def cut_image(image, cut_lines):
    images = []
    cut_lines = [0] + cut_lines + [image.shape[1]]
    for i in range(len(cut_lines)):
        if i == 0: continue
        previous_line = cut_lines[i - 1]
        current_line = cut_lines[i]
        images.append(image[:, previous_line : current_line])
        
    return images

def extract_letters(image):
    image = preprocess_image(image)
    cut_lines = find_cut_lines(image)
    letters = cut_image(image, cut_lines)
#    for line in cut_lines:
#        image[:, line] = 0
#    plt.imshow(image, cmap='gray')
#    plt.show()
    return letters

def save_letters(images, path):
    counter = 0
    width = 32
    delete_content(path)
    for im in images:
        counter += 1
        if (counter == 1 or counter == len(images)):
            continue;
        name = path + "/im_" + ('%03d' % counter) + ".png"
        misc.imsave(name, im)
        resize(width, name)
            
def delete_content(dir_path):
    for afile in glob.glob(dir_path + '/*'):
        os.remove(afile)
            
def resize(width, name):
    im = Image.open(name)
    im.thumbnail((width, width))
    im.save(name)
    
    max_margin = 4
    im = misc.imread(name, mode = 'L')
    top_margin = 0
    for i in range(im.shape[0]):
        if min(im[i, :]) == 255:
            top_margin += 1
        else:
            break
    
    top = top_margin - max_margin
    if top > 0:
        im = im[top:, :]
    
    bottom_margin = 0
    for i in range(im.shape[0] - 1, 0, -1):
        if min(im[i, :]) == 255:
            bottom_margin += 1
        else:
            break
    bottom = bottom_margin - max_margin
    if bottom > 0:
        im = im[: -bottom, :]
        
    #side = int((bottom + top + 1) / 4)
    #print(side)
    #im = im[:, side : -side]
    misc.imsave(name, im)
    
    im = Image.open(name)
    im.thumbnail((width, width))
    im.save(name)
    im = misc.imread(name, mode = 'L')
    im_width = im.shape[1]
    if im_width < width:
        pad_left = math.ceil((width - im_width) / 2.0)
        pad_right = math.floor((width - im_width) / 2.0)
        for i in range(pad_left):
            im = np.insert(im, 0, 255, axis = 1)
        for i in range(pad_right):
            im = np.insert(im, im.shape[1], 255, axis = 1)
        misc.imsave(name, im)

def main(image_path, folder_path):
    image = misc.imread(image_path, flatten = True, mode = 'L')
    image.flags.writeable = True
    images = extract_letters(image)
    save_letters(images, folder_path)
    
def segment(image_path, folder_path):
    main(image_path, folder_path)
   
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Arguments are missing: script.py image_path path_to_save")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
