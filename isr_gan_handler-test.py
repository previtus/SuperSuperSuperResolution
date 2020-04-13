# Project: https://idealo.github.io/image-super-resolution/

# pip install ISR
# pip install tesorflow-gpu==2.0.0 # the default repo installs CPU version only
# ((but works also with tf 1.14.0))

## HOWTO:
## >> conda activate tf2.0gpu

import numpy as np
from PIL import Image
from skimage.transform import resize
from timeit import default_timer as timer
import matplotlib
import matplotlib.image
from os import listdir
from os.path import isfile, join

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

img = Image.open(path)
sr_img = np.array(img)

w,h,ch = np.asarray(sr_img).shape
print("image resolution:", w,h,ch)
if ch == 4:
    sr_img = sr_img[:,:,0:3]
    w, h, ch = np.asarray(sr_img).shape
    print("image resolution:", w, h, ch)

from ISR.models import RDN
rdn = RDN(weights='psnr-small')
#rdn = RDN(weights='psnr-large')

loops = 100
print("Now looping for", loops)
for i in range(loops):

    super_img = rdn.predict(sr_img) #, by_patch_of_size=256)
    smallnp = np.asarray( resize(super_img, (1024, 1024)) )

    int_num += 1
    save_as = "superloop-isr/"+str(int_num).zfill(6)+name

    matplotlib.image.imsave(save_as, smallnp)
    print('saved', save_as)

    sr_img = np.int16(smallnp * 255.0)

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
