from PIL import Image
import os
import sys
from pathlib import Path
import natsort


def get_ordered_images(frame_folder):
    image_list1 = []
    image_list2 = []
    image_list4 = []

    for filename in Path(frame_folder).glob('*.png'):
        image_list1.append(filename)

    for i in image_list1:
        image_list2.append(i.stem)

    image_list3 = natsort.natsorted(image_list2, reverse=False)

    for i in image_list3:
        i = str(i) + ".png"
        image_list4.append(Path(frame_folder, i))

    frames = [Image.open(i) for i in image_list4]

    return frames


def make_gif(frame_folder, name_save_folder):
    try:
        frames = get_ordered_images(frame_folder)
        frame_one = frames[0]
        frame_one.save(f"{name_save_folder}", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)
    except IndexError:
        pass


if __name__ == "__main__":
    # Set working path
    path = os.path.abspath(sys.path[0])

    # Path to the deep field Catalog
    path_f_folder = path + r"/../../Output/2gif"
    name_s_file = path_f_folder + r"/Dist_run_1_gif_bin__i.gif"

    make_gif(path_f_folder, name_s_file)
