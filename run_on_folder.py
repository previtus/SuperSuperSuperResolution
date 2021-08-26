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

def main(args):
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
        from isr_gan_handler import prepare_isrgan, isrgan
        rdn = prepare_isrgan()
    

    number_of_iterations = args.number_of_iterations
    formats = [".jpg", ".png"]
    image_paths = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) and f[-4:] in formats]
    image_paths.sort()

    print("Found:", len(image_paths), image_paths)
    image_idx = 0
    for image_path in tqdm(image_paths):
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
                isrgan(load_name=path, save_name = save_as, rdn = rdn)

            print('saved', save_as)
            path = save_as

        image_idx += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Project: SuperSuperSuper Resolution')
    parser.add_argument('-inp', help='Source images folder.', default='input_folder')
    parser.add_argument('-out', help='Save images into folder.', default='output_folder')

    parser.add_argument('-width', help='Image width (will be scaled in original ratio, then padded).', default=1920)
    parser.add_argument('-height', help='Image height (will be scaled in original ratio, then padded).', default=1920)

    parser.add_argument('-number_of_iterations', help='How many times run super res.', default=40)
    parser.add_argument('-method', help='Which super res method.', default='sftgan')
    
    parser.add_argument('-save_format', help='Save as.', default='png')

    parser.add_argument('-skip_to', help='Skip to file of this index.', default=0)


    args = parser.parse_args()

    #args.inp = "/home/vitek/Vitek/datasets/Art - film end titles/scraped - group_400716_N22/"
    #args.out = "/media/vitek/4E3EC8833EC86595/Vitek/OutSuperRes_Large/"
    args.save_format = "jpg"
    #args.width = 720
    #args.height = 720
    #args.number_of_iterations = 10
    #args.skip_to = 72 #starts with skip_to + 1 , because of starting at 0, set 6 to start with "image_0006" 
                     # ~ if it < skip_to: continue
    print("Project: SuperSuperSuper Resolution // with args=", args)


    args.width = int(args.width)
    args.height = int(args.height)
    args.number_of_iterations = int(args.number_of_iterations)
    args.skip_to = int(args.skip_to)
    main(args)