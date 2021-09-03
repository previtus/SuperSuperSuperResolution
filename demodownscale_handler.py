import os
import glob
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

import sys
sys.path.append(r'SFTGAN/pytorch_test/')
import math

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def imresize(img, scale = 1/4):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    return resized
    

def demo_downscale(load_name="", save_name = 'fin_rlt.png', scale_factor = 8.0):
    path = load_name
    # Simulate lightening
    
    img = np.asarray(cv2.imread(path, cv2.IMREAD_UNCHANGED), dtype=np.float32)
    img *= 1.1 
    save_img(img, save_name)
    
    return 0
    # Just a simple downscale by scale_factor down and up with cv2.INTER_NEAREST
    img_LR = imresize(img / 255, 1 / scale_factor)
    
    img_LR *= 1.05

    out_img = imresize(img_LR, scale_factor) * 255
    save_img(out_img, save_name)
    print("in:", img.shape, "out:", img_LR.shape, "resized to:", out_img.shape)
    

if __name__ == "__main__":

    mypath = "superloop-demo/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()
    last_file = onlyfiles[-1]
    namelist = last_file.split("_")
    int_num = int(namelist[0])
    name = "_" + "_".join(namelist[1:])
    print(name, int_num, "and last is", last_file, "from whole list of", onlyfiles)

    path = mypath + last_file
    print("opening", path)

    print("=================================================================================================================")


    loops = 5
    print("Now looping for", loops)
    for i in range(loops):

        int_num += 1
        save_as = "superloop-demo/"+str(int_num).zfill(6)+name

        demo_downscale(load_name=path, save_name=save_as)
        print('saved', save_as)

        path = save_as
