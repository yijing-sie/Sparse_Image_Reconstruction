import argparse
import os
import sys
from PIL import Image
import numpy as np
import pywt
import scipy.io


def imagepath2numpy(im_pth):
    img = Image.open(im_pth).convert('L')
    arr = np.array(img)
    return arr


def main(args):
    img_arr = imagepath2numpy(args.image_pth)
    mask = scipy.io.loadmat(args.mask_pth)['mask']


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_pth', type=str, default='data/image.png', help="path to the input image")
    parser.add_argument('--mask_pth', type=str, default='mask.mat', help="path to mask")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    main(args)