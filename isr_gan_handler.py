# Project: https://idealo.github.io/image-super-resolution/

# pip install ISR
# pip install tensorflow-gpu==2.0.0 # the default repo installs CPU version only
# ((but works also with tf 1.14.0))

## HOWTO:
## >> conda activate tf2.0gpu
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from PIL import Image
from skimage.transform import resize
from timeit import default_timer as timer
import matplotlib
import matplotlib.image
from os import listdir
from os.path import isfile, join
import cv2


def prepare_isrgan(weights = 'psnr-small'):
    # Currently 4 models are available: - RDN: psnr-large, psnr-small, noise-cancel - RRDN: gans
    if 'psnr' in weights:
        from ISR.models import RDN
        rdn = RDN(weights=weights)
    elif 'gans' == weights:
        from ISR.models import RRDN
        rdn = RRDN(weights=weights) 
    #rdn = RDN(weights='psnr-large')
    return rdn

""" # problems with format ...
def save_img(img, img_path):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, img_rgb)
"""

def isrgan(load_name="", save_name = 'fin_rlt.png', mode = 'rgb', rdn = None, dont_dowscale_result=False):
    # Take 1: Load, downscale, then superres
    path = load_name
    img = Image.open(path)
    sr_img = np.array(img)
    w,h,ch = np.asarray(sr_img).shape


    sr_img = cv2.resize(sr_img, (int(w/2),int(h/2)))
    sr_img = np.asarray( sr_img )

    print("in resolution:", np.asarray(sr_img).shape)
    if ch == 4:
        sr_img = sr_img[:,:,0:3]
        print("in resolution:", np.asarray(sr_img).shape)


    super_img = rdn.predict(sr_img) #, by_patch_of_size=256)
    smallnp = np.asarray( resize(super_img, (int(w),int(h))) )

    print("out resolution:", smallnp.shape)

    matplotlib.image.imsave(save_name, smallnp) # < care: saves PNGA
    print('saved', save_name)



def isrgan_take2(load_name="", save_name = 'fin_rlt.png', mode = 'rgb', rdn = None, dont_dowscale_result=False):
    # Take 2: load, superres then downscale ~ needs setting of by_patch_of_size for 1920 resolution!!
    path = load_name
    img = Image.open(path)
    sr_img = np.array(img)

    w,h,ch = np.asarray(sr_img).shape
    print("in resolution:", w,h,ch)
    if ch == 4:
        sr_img = sr_img[:,:,0:3]
        w, h, ch = np.asarray(sr_img).shape
        print("in resolution:", w, h, ch)


    super_img = rdn.predict(sr_img, by_patch_of_size=128)
    smallnp = np.asarray( resize(super_img, (w,h)) )
    print("out resolution:", smallnp.shape)

    matplotlib.image.imsave(save_name, smallnp) # < care: saves PNGA
    print('saved', save_name)




if __name__ == "__main__":
    
    mypath = "superloop-isr/"
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

    rdn = prepare_isrgan()

    loops = 100
    print("Now looping for", loops)
    for i in range(loops):

        int_num += 1
        save_as = "superloop-isr/"+str(int_num).zfill(6)+name
        isrgan(load_name=path, save_name = save_as, rdn = rdn)

        print('saved', save_as)

        path = save_as


    print("Finished!")

    # Large file inferrence? sr_img = model.predict(image, by_patch_of_size=50)

    """ # Training:
    from ISR.models import RRDN
    from ISR.models import Discriminator
    from ISR.models import Cut_VGG19

    lr_train_patch_size = 40
    layers_to_extract = [5, 9]
    scale = 2
    hr_train_patch_size = lr_train_patch_size * scale

    rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
    f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)


    from ISR.train import Trainer
    loss_weights = {
      'generator': 0.0,
      'feature_extractor': 0.0833,
      'discriminator': 0.01
    }
    losses = {
      'generator': 'mae',
      'feature_extractor': 'mse',
      'discriminator': 'binary_crossentropy'
    }

    log_dirs = {'logs': './logs', 'weights': './weights'}

    learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

    flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

    trainer = Trainer(
        generator=rrdn,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir='low_res/training/images',
        hr_train_dir='high_res/training/images',
        lr_valid_dir='low_res/validation/images',
        hr_valid_dir='high_res/validation/images',
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname='image_dataset',
        log_dirs=log_dirs,
        weights_generator=None,
        weights_discriminator=None,
        n_validation=40,
    )


    trainer.train(
        epochs=80,
        steps_per_epoch=500,
        batch_size=16,
        monitored_metrics={'val_PSNR_Y': 'max'}
    )


    """
