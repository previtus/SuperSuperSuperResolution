# Project: https://idealo.github.io/image-super-resolution/

# This one needs TF-2.0
# pip install ISR
# pip install tesorflow-gpu==2.0.0 # the default repo installs CPU version only


## HOWTO:
## >> source activate tf2.0gpu

import numpy as np
from PIL import Image
from timeit import default_timer as timer

#img = Image.open('image-super-resolution/data/input/test_images/sample_image.jpg')
img = Image.open('image-super-resolution/data/input/sample/baboon.png') # not sure if not in train set
lr_img = np.array(img)

from ISR.models import RDN

#rdn = RDN(weights='psnr-small')
rdn = RDN(weights='psnr-large')
_ = rdn.predict(lr_img)

start = timer()
print("np.asarray(lr_img).shape", np.asarray(lr_img).shape) # np.asarray(lr_img).shape (120, 125, 3)
sr_img = rdn.predict(lr_img) # resolution * 2!
print("np.asarray(sr_img).shape", np.asarray(sr_img).shape) # np.asarray(sr_img).shape (240, 250, 3)
# Times:
# CPU + psnr-small = 0.5547288789998674s
# GPU + psnr-small = 0.05576786199981143s # Real-time!
# CPU + psnr-large = 3.39864076799995s
# GPU + psnr-large = 0.24734936100003324s
end = timer()
time = (end - start)
print("This run took " + str(time) + "s (" + str(time / 60.0) + "min)")


loops = 2
print("Now looping for", loops)
for i in range(loops):
    start = timer()
    sr_img = rdn.predict(sr_img, by_patch_of_size=50)
    end = timer()
    time = (end - start)
    print("This one took " + str(time) + "s (" + str(time / 60.0) + "min)")

    #sr_img = rdn.predict(sr_img)
    print("np.asarray(sr_img).shape", np.asarray(sr_img).shape)
    # np.asarray(sr_img).shape (480, 500, 3)
    # np.asarray(sr_img).shape (960, 1000, 3)

    # CPU:
    # This one took 70.21437167299973s (1.1702395278833289min)
    # np.asarray(sr_img).shape (960, 1000, 3)
    # GPU:
    # This one took 5.529112465999788s (0.0921518744333298min)
    # np.asarray(sr_img).shape (960, 1000, 3)

sr = Image.fromarray(sr_img)
sr.show()
sr.save("last_isr_photo.jpg")

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