import os
import glob
import numpy as np
import cv2

import math
import torch
from PIL import Image

def color_normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 32) * 32)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 32) * 32)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img).cuda(non_blocking=True)
    return img, ori_h, ori_w



def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)



def read_seg(seg_dir, scale_factor, scale_size=[480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 32) * 32)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 32) * 32)
        small_seg = np.array(seg.resize((_tw // scale_factor, _th // scale_factor), 0))
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
        small_seg = np.array(seg.resize((_tw, _th), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)

    return to_one_hot(small_seg), np.asarray(seg)


def read_seg_hw(seg_dir, scale_factor_h, scale_factor_w, scale_size=[480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 32) * 32)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 32) * 32)
        small_seg = np.array(seg.resize((_tw // scale_factor_w, _th // scale_factor_h), 0))
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
        small_seg = np.array(seg.resize((_tw, _th), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)

    return to_one_hot(small_seg), np.asarray(seg)


def read_frame_mose(frame_dir, height=480):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    
    aspect_ratio = ori_w / ori_h
    new_width = math.ceil(height * aspect_ratio)
    new_width = int((new_width // 64) * 64)

    # Resize the image
    img = cv2.resize(img, (new_width, height), interpolation=cv2.INTER_AREA)
    #breakpoint()

    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img).cuda(non_blocking=True)
    return img, ori_h, ori_w\



def read_seg_mose(seg_dir, small_h, small_w):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    
    small_seg = np.array(seg.resize((small_w, small_h), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)

    return to_one_hot(small_seg), np.asarray(seg)