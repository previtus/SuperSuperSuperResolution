import os
import glob
import numpy as np
import cv2

import torch
import torchvision.utils

from os import listdir
from os.path import isfile, join

import sys
sys.path.append(r'SFTGAN/pytorch_test/')
#from SFTGAN.pytorch_test import architectures as arch
#from SFTGAN.pytorch_test import util as util
import architectures as arch
import util


mypath = "superloop-sft/"
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


def sftgan(load_name="", save_name = 'fin_rlt.png'):
    path = load_name
    test_img_folder_name = "TMP1"
    # options
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> 'cpu'
    # device = torch.device('cpu')

    # make dirs
    test_img_folder = 'SFTGAN/data/' + test_img_folder_name  # HR images
    save_prob_path = 'SFTGAN/data/' + test_img_folder_name + '_segprob'  # probability maps
    save_byteimg_path = 'SFTGAN/data/' + test_img_folder_name + '_byteimg'  # segmentation annotations
    save_colorimg_path = 'SFTGAN/data/' + test_img_folder_name + '_colorimg'  # segmentaion color results
    util.mkdirs([save_prob_path, save_byteimg_path, save_colorimg_path])

    test_prob_path = 'SFTGAN/data/' + test_img_folder_name + '_segprob'  # probability maps
    save_result_path = 'SFTGAN/data/' + test_img_folder_name + '_result'  # results
    util.mkdirs([save_result_path])

    # load model
    seg_model = arch.OutdoorSceneSeg()
    seg_model_path = 'SFTGAN/pretrained_models/segmentation_OST_bic.pth'
    seg_model.load_state_dict(torch.load(seg_model_path), strict=True)
    seg_model.eval()
    seg_model = seg_model.to(device)

    # look_up table, RGB, for coloring the segmentation results
    lookup_table = torch.from_numpy(
        np.array([
            [153, 153, 153],  # 0, background
            [0, 255, 255],    # 1, sky
            [109, 158, 235],  # 2, water
            [183, 225, 205],  # 3, grass
            [153, 0, 255],    # 4, mountain
            [17, 85, 204],    # 5, building
            [106, 168, 79],   # 6, plant
            [224, 102, 102],  # 7, animal
            [255, 255, 255],  # 8/255, void
        ])).float()
    lookup_table /= 255

    print('Testing segmentation probability maps ...')


    """
    for idx, path in enumerate(glob.glob(test_img_folder + '/*')):
        imgname = os.path.basename(path)
        basename = os.path.splitext(imgname)[0]
    
        if "txt" in path:
          continue
    """

    idx = 0
    if True:
        #print(idx + 1, basename, path)
        print(idx + 1)
        # read image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = util.modcrop(img, 8)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        #print(img.shape) # w,h,3 <- 1
        stacked_img = np.stack((img,)*3, axis=2)
        stacked_img = stacked_img[:,:,:,0]
        #print(stacked_img.shape) # w,h,3 <- 1
        img = stacked_img

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        # MATLAB imresize
        # You can use the MATLAB to generate LR images first for faster imresize operation
        img_LR = util.imresize(img / 255, 1 / 4, antialiasing=True)
        img = util.imresize(img_LR, 4, antialiasing=True) * 255

        img[0] -= 103.939
        img[1] -= 116.779
        img[2] -= 123.68
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = seg_model(img).detach().float().cpu().squeeze()

        # save segmentation probability maps
        #torch.save(output, os.path.join(save_prob_path, basename + '_bic.pth'))  # 8xHxW
        SEG_OUT = output
        """
        # save segmentation byte images (annotations)
        _, argmax = torch.max(output, 0)
        argmax = argmax.squeeze().byte()
        cv2.imwrite('foo1.png', argmax.numpy())
    
        # save segmentation colorful results
        im_h, im_w = argmax.size()
        color = torch.FloatTensor(3, im_h, im_w).fill_(0)  # black
        for i in range(8):
            mask = torch.eq(argmax, i)
            color.select(0, 0).masked_fill_(mask, lookup_table[i][0])  # R
            color.select(0, 1).masked_fill_(mask, lookup_table[i][1])  # G
            color.select(0, 2).masked_fill_(mask, lookup_table[i][2])  # B
        # void
        mask = torch.eq(argmax, 255)
        color.select(0, 0).masked_fill_(mask, lookup_table[8][0])  # R
        color.select(0, 1).masked_fill_(mask, lookup_table[8][1])  # G
        color.select(0, 2).masked_fill_(mask, lookup_table[8][2])  # B
        torchvision.utils.save_image(
            color, 'foo2.png', padding=0, normalize=False)
        """

    del seg_model

    '''
    Codes for testing SFTGAN
    '''

    # options
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sres_model_path = 'SFTGAN/pretrained_models/SFTGAN_torch.pth'  # torch version
    # sres_model_path = 'SFTGAN/pretrained_models/SFTGAN_noBN_OST_bg.pth'  # pytorch version

    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> 'cpu'
    # device = torch.device('cpu')

    if 'torch' in sres_model_path:  # torch version
        model = arch.SFT_Net_torch()
    else:  # pytorch version
        model = arch.SFT_Net()
    model.load_state_dict(torch.load(sres_model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Testing SFTGAN ...')

    """
    for idx, path in enumerate(glob.glob(test_img_folder + '/*')):
        imgname = os.path.basename(path)
        basename = os.path.splitext(imgname)[0]
        
        if "txt" in path:
          continue
    """
    if True:
        path
        #print(idx + 1, basename)
        print(idx + 1)
        # read image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = util.modcrop(img, 8)
        img = img * 1.0 / 255
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        #print(img.shape) # w,h,3 <- 1
        stacked_img = np.stack((img,)*3, axis=2)
        stacked_img = stacked_img[:,:,:,0]
        #print(stacked_img.shape) # w,h,3 <- 1
        img = stacked_img

        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # MATLAB imresize
        # You can use the MATLAB to generate LR images first for faster imresize operation
        img_LR = util.imresize(img, 1 / 4, antialiasing=True)
        img_LR = img_LR.unsqueeze(0)
        img_LR = img_LR.to(device)

        # read segmentation probability maps
        #seg = torch.load(os.path.join(test_prob_path, basename + '_bic.pth'))
        seg = SEG_OUT
        seg = seg.unsqueeze(0)
        # change probability
        # seg.fill_(0)
        # seg[:,5].fill_(1)
        seg = seg.to(device)
        with torch.no_grad():
            output = model((img_LR, seg)).data.float().cpu().squeeze()
        output = util.tensor2img(output)
        util.save_img(output, save_name)

loops = 100
print("Now looping for", loops)
for i in range(loops):

    int_num += 1
    save_as = "superloop-sft/"+str(int_num).zfill(6)+name

    sftgan(load_name=path, save_name=save_as)
    print('saved', save_as)

    path = save_as
