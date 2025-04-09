import os
from PIL import Image
import numpy as np

from utils.read_util import read_frame, read_seg, read_seg_hw, read_frame_mose, read_seg_mose

class VideoReaderPerObject:
    """
    Iterates over each object in the video
    """
    def __init__(self, image_directory: str, mask_directory: str, vid_name: str, is_mose: bool, device):
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.vid_name = vid_name
        # read all frames in the image directory
        frames = sorted(os.listdir(self.image_directory))

        self.first_mask_frame = sorted(os.listdir(self.mask_directory))[0]
        first_mask = Image.open(os.path.join(self.mask_directory, self.first_mask_frame)).convert('P')
        self.color_palette = first_mask.getpalette()

        self.n_objects = np.max(np.array(first_mask))
        self.idx = 1
        self.object_idx = 1
        
        self.frames = frames
        self.frames_object_idxs = []

        for object_id in range(1, self.n_objects+1):
            for frame in self.frames:
                self.frames_object_idxs.append([object_id, frame])

        self.max_idx = len(self.frames_object_idxs)

        self.device = device

        self.read_frame_func = read_frame_mose if is_mose else read_frame
        self.is_mose = is_mose

    
    def set_curr_hw(self, curr_h, curr_w):
        self.curr_h = curr_h
        self.curr_w = curr_w

        scale_factor_h = self.ori_h // curr_h
        scale_factor_w = self.ori_w // curr_w

        if self.is_mose:
            first_seg, seg_ori = read_seg_mose(os.path.join(self.mask_directory, self.first_mask_frame), curr_h, curr_w)
        else :
            if scale_factor_h != scale_factor_w:
                self.scale_factor_h = scale_factor_h
                self.scale_factor_w = scale_factor_w
                self.scale_factor = (scale_factor_h, scale_factor_w)
                first_seg, seg_ori = read_seg_hw(os.path.join(self.mask_directory, self.first_mask_frame), self.scale_factor_h, self.scale_factor_w)
            else:
                self.scale_factor = scale_factor_h
                first_seg, seg_ori = read_seg(os.path.join(self.mask_directory, self.first_mask_frame), self.scale_factor)

        
        self.first_seg = first_seg.to(self.device)
        self.first_seg_ori = seg_ori

      
    def get_f0_frame(self):
        f0_frame_name = self.frames[0]
        f0_frame, ori_h, ori_w = self.read_frame_func(os.path.join(self.image_directory, f0_frame_name))
        f0_frame = f0_frame.to(self.device)
        self.ori_h = ori_h
        self.ori_w = ori_w
        self.idx = 1

        return f0_frame
    
    def __iter__(self):
        self.idx = 1
        self.object_idx = 1
        return self
    
    def __next__(self):
        if self.idx >= self.max_idx:
            raise StopIteration
        
        object_id, curr_frame_name = self.frames_object_idxs[self.idx]
        curr_frame, _, _ = self.read_frame_func(os.path.join(self.image_directory, curr_frame_name))
        curr_frame = curr_frame.to(self.device)
      
        self.idx += 1

        return curr_frame, object_id, curr_frame_name
    
    def __len__(self):
        return self.max_idx
    