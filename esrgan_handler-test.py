import os.path as osp
import glob
import cv2
import numpy as np
import torch
import sys
from os import listdir
from os.path import isfile, join

image_saving_target_size = 1024 # target saving resolution
image_processing_target_size = 512 # with which resolution will the network take it


sys.path.append(r'ESRGAN/')
import RRDBNet_arch as arch

model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# on 2GB mem gpu only up to image_processing_target_size = 256
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu') # might work better if I have small GPU mem!

mypath = "superloop-esr/"
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
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

loops = 100
print("Now looping for", loops)
for i in range(loops):
    print(int_num)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    w, h, ch = np.asarray(img).shape
    print("loaded as",np.asarray(img).shape)
    img = cv2.resize(img, (image_processing_target_size, image_processing_target_size), interpolation=cv2.INTER_AREA)
    print("shrank to",np.asarray(img).shape)

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    print("outputs at",np.asarray(output).shape)

    w, h, ch = np.asarray(output).shape
    if w != image_saving_target_size:
        output = cv2.resize(output, (image_saving_target_size, image_saving_target_size), interpolation=cv2.INTER_AREA)
        print("scaled back to", np.asarray(output).shape)

    int_num += 1
    save_as = "superloop-esr/" + str(int_num).zfill(6) + name

    # special case where we work with b/w images
    output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(save_as, output)

    path = save_as


