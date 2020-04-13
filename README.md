# SuperSuperSuperResolution

What will we get if we keep "enhancing" our photographs?
This repo contains bunch of handlers for Super resolution GANs.

## Install

Depending on which super resolution GAN we will be using, different setup is neccessary (but it is possible to have them all setup in just one environment).

SFTGAN

- `git clone https://github.com/xinntao/SFTGAN.git`
- download their models into `"SFTGAN/pretrained_models/segmentation_OST_bic.pth"` and `"SFTGAN/pretrained_models/SFTGAN_torch.pth"`
- prepare a folder `superloop-sft/` with one starting image named `000000_<anyname>.png`
- `python sftgan_handler-test.py`

ESRGAN

- `git clone https://github.com/xinntao/ESRGAN.git`
- download their models into `"ESRGAN/models/RRDB_ESRGAN_x4.pth"`
- prepare a folder `superloop-esr/` with one starting image named `000000_<anyname>.png`
- `python esrgan_handler-test.py`

ISRGAN (image-super-resolution)

- `pip install ISR`
- prepare a folder `superloop-isr/` with one starting image named `000000_<anyname>.png`
- `python isr_gan_handler-test.py`


## Demo animations:

SFTGAN

![SuperSuperSuper(SFTGAN) demo](https://github.com/previtus/SuperSuperSuperResolution/raw/master/demos/sft_demo.gif)

ESRGAN

![SuperSuperSuper(ESRGAN) demo](https://github.com/previtus/SuperSuperSuperResolution/raw/master/demos/esr_demo.gif)

ISRGAN (image-super-resolution)

![SuperSuperSuper(ISR) demo](https://github.com/previtus/SuperSuperSuperResolution/raw/master/demos/isr_demo.gif)
