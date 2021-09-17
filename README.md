# SuperSuperSuperResolution

What will we get if we keep "enhancing" our photographs?
This repo contains bunch of handlers for Super resolution GANs.

<p align="center">
<img src="https://github.com/previtus/SuperSuperSuperResolution/raw/master/super_super_super_resolution_illustration.jpg" width="500">
</p>

<table><tr><td><b>Super Super Super Resolution</b></td></tr>
  <tr><td>
This project explores <b>what happens when we use super resolution models to enhance our footage</b>, either when restoring old films from low resolution recordings, or when trying to augment a photo taken on a small camera sensor.<br><br>
In a playful manner, we repeat it over many iterations, thus highlighting the small changes this translation adds to the original image. Undeniably we are mixing our original photograph's content with noise and low-level detail information learned from other (sometimes proprietary) datasets.<br><br>
We use the <b>SFTGAN model</b> which, when repeated, alters the original imagery with an effect like that of an analog film burning, when a frame gets stuck in the projector. This effect has been used in experimental cinema (for example in Bardo Follies (1967) by Owen Land) to visually represent film death on the screen, revealing the materiality of the film itself. In this way the audience can experience death and then still go home unharmed.<br><br>
<b>Recorded reality is more and more encoded and mixed with the artefacts of the neural age.</b>  This offers possibilities for creative expression, as long as these enhancements are in our hands. We ask alongside with "The end" screens: <b>imagine the possible futures, what happens next?</b>
</td></tr></table>

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
