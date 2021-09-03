"""

- in: folder with photos, various formats (jpg, png, ?) and dimensions - for each:
      args: desired size (width, height), number_of_iterations, method = 'sftgan'

-- name ~ image_<000i>_frame_<000i>.png

-- resize and pad with black border, save as i=0
-- for i in number_of_iterations : run method

"""
from os import listdir
from os.path import isfile, join
import cv2
from tqdm import tqdm
import math
from shutil import copyfile
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return image

def save_image(img, img_path):
    cv2.imwrite(img_path, img)

def prepare_image(image, width, height):
    if len(image.shape) == 3:
        h,w,ch = image.shape
    else:
        h,w = image.shape
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    ratio = h/w
    new_h = int(height * ratio)
    new_w = width
    print("w,h",w,h, "->", new_w, new_h)

    resized = cv2.resize(image, (new_w, new_h))
    #print(resized.shape)

    # Pad by black
    delta_w = width - resized.shape[1]
    delta_h = height - resized.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    #print(resized.shape, padded.shape)

    return padded

def constant_speed_loop(args):
    # Get all input images:

    input_folder = args.inp
    output_folder = args.out
    save_format = args.save_format
    width = args.width
    height = args.height
    skip_to = args.skip_to
    method = args.method

    if method == 'sftgan':
        from sftgan_handler import sftgan
    elif method == 'isrgan':
        from isr_gan_handler import prepare_isrgan, isrgan, isrgan_take2
        # Currently 4 models are available: - RDN: psnr-large, psnr-small, noise-cancel - RRDN: gans
        #rdn = prepare_isrgan('psnr-small')
        rdn = prepare_isrgan('gans')
    elif method == 'demo':
        from demodownscale_handler import demo_downscale


    number_of_iterations = args.number_of_iterations
    formats = [".jpg", ".png"]
    image_paths = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and f[-4:] in formats]
    image_paths.sort()

    print("Found:", len(image_paths), image_paths)
    image_idx = 0
    for image_path in tqdm(image_paths):
        ### NEEDS EVOLUTION!
        ### strih ~ tematicky za sebou
        # zacit smysluplnym lidem 25 stoleti...
        # nesmi mit stejny cas vsechno = progress vic a vic snad
        # clovek logicky nejdriv cte titulku, posleze zacne vnimat ruznosti a pak si uzivat ten rozklad
        ## chce to foor koncept
        ## koncit mozno s to be continued 
        # Konec koncuuu ?

        if image_idx < skip_to:
            image_idx += 1
            continue
        full_path = input_folder + "/" + image_path
        print("####### IMAGE", image_path, "#######")
        
        # Prepare first image
        frame_idx = 0
        image_name = output_folder + "/" + "image_" + str(image_idx).zfill(4) + "_frame_"+str(frame_idx).zfill(4)+"."+save_format

        image = load_image(full_path)
        prepared_image = prepare_image(image, width, height)
        save_image(prepared_image, image_name)
        path = image_name

        # Run iterations:
        for frame_idx in range(number_of_iterations):
            save_as = output_folder + "/" + "image_" + str(image_idx).zfill(4) + "_frame_"+str(frame_idx + 1).zfill(4)+"."+save_format
            override_input = frame_idx == 0
            if method == 'sftgan':
                sftgan(load_name=path, save_name=save_as, override_input=override_input)

            elif method == 'isrgan':
                isrgan(load_name=path, save_name = save_as, rdn = rdn) # ~ default mode, may be memory limited, downscale first
                #isrgan_take2(load_name=path, save_name = save_as, rdn = rdn) # ~ mode using small patches
                # patch size ~ search in https://github.com/idealo/image-super-resolution/blob/3f6498cf1ac4dba162a52c5861aa9d90b7c2fe35/ISR/utils/image_processing.py#L42
                #              which value do we have there if we don't set it manually ... can it be the whole img?

            elif method == 'demo':
                demo_downscale(load_name=path, save_name = save_as)
    
            print('saved', save_as)
            path = save_as

        image_idx += 1


def gradation_with_kept_length_for_image(args):
    # Get all input images:

    input_folder = args.inp
    output_folder = args.out
    save_format = args.save_format
    width = args.width
    height = args.height
    skip_to = args.skip_to
    method = args.method

    if method == 'sftgan':
        from sftgan_handler import sftgan
    elif method == 'isrgan':
        from isr_gan_handler import prepare_isrgan, isrgan, isrgan_take2
        # Currently 4 models are available: - RDN: psnr-large, psnr-small, noise-cancel - RRDN: gans
        #rdn = prepare_isrgan('psnr-small')
        rdn = prepare_isrgan('gans')
    elif method == 'demo':
        from demodownscale_handler import demo_downscale


    number_of_iterations = args.number_of_iterations
    formats = [".jpg", ".png"]
    image_paths = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and f[-4:] in formats]
    image_paths.sort()


    # Repetitions prepare
    # I want to keep it at desired_length, while also doing rounds of Superresolutions...
    # ... there is graduation to higher number of rounds at the same fixed desired_lenght as a number of frames 
    # ... (plus minus, given the ceil)

    # fps 30 looks good
    rounds_0 = [1] + [80]
    rounds_A = list(range(10,40,2))
    rounds_B = [40] # list(range(40,80,4)) # jen jestli 80 neni moc iteraci...
    
    rounds_D = list(range(40,80,4)) + [80] * (5) # only last Z at full edit
    
    rounds_C = [40] * (len(image_paths) - len(rounds_B) - len(rounds_A) - len(rounds_D))
    rounds_all = rounds_0 + rounds_A + rounds_B + rounds_C + rounds_D
    desired_length_0 = [80] + [160] # long start with this one, same image twice... ~ works nicely, maybe too long corrosion, maybe could be with lower rounds?
    desired_length_A = [40] * len(rounds_A)
    desired_length_B = [80] * len(rounds_B)
    desired_length_C = [80] * len(rounds_C)
    desired_length_D = [80] * len(rounds_D)
    desired_lengths = desired_length_0 + desired_length_A + desired_length_B + desired_length_C + desired_length_D

    print("======================================")

    print("We have plan for", len(desired_lengths), "samples.") # 63 samples.
    print("0", len(desired_length_0), "samples. ~ fixed intro")
    print("A", len(desired_length_A), "samples. ~ speeding up from 10 to 40")
    print("B", len(desired_length_B), "samples. ~ kept at 40")
    print("C", len(desired_length_C), "samples. ~ still at 40")
    print("D", len(desired_length_D), "samples. ~ hyperdrive to 80!")

    print("Sum #", np.sum(desired_lengths), "frames, which at 30fps is ", (np.sum(desired_lengths)/30)/60, "min. At 15fps it is", (np.sum(desired_lengths)/15)/60, "min.")

    print("======================================")

    print("Found:", len(image_paths), image_paths)
    image_idx = 0
    for image_path in tqdm(image_paths):
        ### NEEDS EVOLUTION!
        ### strih ~ tematicky za sebou
        # zacit smysluplnym lidem 25 stoleti...
        # nesmi mit stejny cas vsechno = progress vic a vic snad
        # clovek logicky nejdriv cte titulku, posleze zacne vnimat ruznosti a pak si uzivat ten rozklad
        ## chce to foor koncept
        ## koncit mozno s to be continued 
        # Konec koncuuu ?

        if image_idx < skip_to:
            image_idx += 1
            continue
        full_path = input_folder + "/" + image_path
        print("####### IMAGE", image_path, "#######")
        
        # What's the number of repetitions and calls for this image?
        rounds = rounds_all[image_idx]
        desired_length = desired_lengths[image_idx]
        repeats = math.ceil( desired_length / rounds )

        # Prepare first image
        frame_idx = 0
        image_name = output_folder + "/" + "image_" + str(image_idx).zfill(4) + "_frame_"+str(frame_idx).zfill(4)+"."+save_format

        image = load_image(full_path)
        prepared_image = prepare_image(image, width, height)
        save_image(prepared_image, image_name)
        path = image_name

        # Run iterations:
        frame_idx = 0
        for round_idx in range(rounds):
            save_as = output_folder + "/" + "image_" + str(image_idx).zfill(4) + "_frame_"+str(frame_idx + 1).zfill(4)+"."+save_format
            override_input = frame_idx == 0
            if method == 'sftgan':
                sftgan(load_name=path, save_name=save_as, override_input=override_input)

            elif method == 'isrgan':
                isrgan(load_name=path, save_name = save_as, rdn = rdn) # ~ default mode, may be memory limited, downscale first
                #isrgan_take2(load_name=path, save_name = save_as, rdn = rdn) # ~ mode using small patches
                # patch size ~ search in https://github.com/idealo/image-super-resolution/blob/3f6498cf1ac4dba162a52c5861aa9d90b7c2fe35/ISR/utils/image_processing.py#L42
                #              which value do we have there if we don't set it manually ... can it be the whole img?

            elif method == 'demo':
                demo_downscale(load_name=path, save_name = save_as)

            print('saved', save_as)

            frame_idx += 1
            names = []
            for r in range(repeats):
                name = output_folder + "/" + "image_" + str(image_idx).zfill(4) + "_frame_"+str(frame_idx + 1).zfill(4)+"."+save_format
                frame_idx += 1
                names.append(name)

            repeat_file(save_as, names)
            print('also saved as', names)

            path = save_as

        image_idx += 1

def repeat_file(src, names):
    for name in names:
        copyfile(src, name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Project: SuperSuperSuper Resolution')
    parser.add_argument('-inp', help='Source images folder.', default='input_folder')
    parser.add_argument('-out', help='Save images into folder.', default='output_folder')

    parser.add_argument('-width', help='Image width (will be scaled in original ratio, then padded).', default=1920)
    parser.add_argument('-height', help='Image height (will be scaled in original ratio, then padded).', default=1920)

    parser.add_argument('-number_of_iterations', help='How many times run super res.', default=40)
    parser.add_argument('-method', help='Which super res method. (options: sftgan, isrgan, demo)', default='sftgan')
    
    parser.add_argument('-save_format', help='Save as.', default='png')

    parser.add_argument('-skip_to', help='Skip to file of this index.', default=0)


    args = parser.parse_args()
    args.save_format = "jpg"

    """
    args.inp = "/media/vitek/4E3EC8833EC86595/Vitek/Datasets/Art - film end titles/especially good ones (179)/"
    args.out = "/media/vitek/4E3EC8833EC86595/Vitek/OutSuperRes_Goodones_SFT_dynamic_T4/"
    
    #args.out = "/media/vitek/4E3EC8833EC86595/Vitek/OutSuperRes_Goodones_ISR_psnr-small/"
    #args.out = "/media/vitek/4E3EC8833EC86595/Vitek/OutSuperRes_Goodones_demo_dynamic/"
    args.skip_to = 68 #starts with skip_to + 1 , because of starting at 0, set 6 to start with "image_0006" 
                     # ~ if it < skip_to: continue


    args.inp = "/media/vitek/4E3EC8833EC86595/Vitek/Datasets/Art - film end titles/careful select flat (61)/"
    args.out = "/media/vitek/4E3EC8833EC86595/Vitek/OutSuperRes_Selected_SFT_dynamic1/"
    args.skip_to = 0
    """

    print("Project: SuperSuperSuper Resolution // with args=", args)


    args.width = int(args.width)
    args.height = int(args.height)
    args.number_of_iterations = int(args.number_of_iterations)
    args.skip_to = int(args.skip_to)
    
    constant_speed_loop(args)
    #gradation_with_kept_length_for_image(args)