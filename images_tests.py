import numpy as np
from PIL import Image
from skimage.transform import resize
import matplotlib
import matplotlib.image
from timeit import default_timer as timer

path = "/media/ekmek/Vitek/Vitek2020/PythonProjects/SuperSuperSuperResolution/data/renders3_fromLats_120fr___sectionA500/saved_0351.png"
img = Image.open(path)
lr_img = np.array(img)

#debug_flat_1 = lr_img.flatten()
small = resize(lr_img, (512,512))

#tmp = np.int16(smallnp * 255.0) # so it's not from -128 to 127
#debug_flat_2 = tmp.flatten()
#print(np.min(debug_flat_1), np.max(debug_flat_1), np.mean(debug_flat_1), np.std(debug_flat_1))
#print(np.min(debug_flat_2), np.max(debug_flat_2), np.mean(debug_flat_2), np.std(debug_flat_2))

matplotlib.image.imsave('smaller512.png', small)
