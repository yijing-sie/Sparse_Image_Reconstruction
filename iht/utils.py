from PIL import Image
import numpy as np


def imagepath2numpy(im_pth):
    img = Image.open(im_pth).convert('L')  # grayscale
    arr = np.array(img)
    return arr

def save_nparray2image(arr, fpath):
    im = Image.fromarray(arr).convert("L")  # grayscale
    im.save(fpath)