import os
import numpy as np
from PIL import Image
import shutil

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette)
    im.save(filename, format='PNG')


def save_seg(seg, ori_h, ori_w, frame_nm, video_folder, color_palette):
    frame_tar_seg = np.array(seg.squeeze().cpu(), dtype=np.uint8)
    frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
    imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


def make_zip(dataset, annotations_path):
    zip_path = annotations_path.split('/')[:-1]
    zip_path = '/'.join(zip_path) 
    if dataset == 'd17-test':
        print('Making zip for DAVIS test...')
        shutil.make_archive(f'{zip_path}.zip', 'zip', annotations_path)
    elif dataset == 'mose-val':
        print('Making zip for MOSE validation...')
        shutil.make_archive(f'{zip_path}.zip', 'zip', annotations_path)
    
    else:
        print(f'Not making zip for {dataset}.')