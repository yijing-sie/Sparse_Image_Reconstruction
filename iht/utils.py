from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error


def imagepath2numpy(im_pth):
    img = Image.open(im_pth).convert('L')  # grayscale
    arr = np.array(img)
    return arr

def save_nparray2image(arr, fpath):
    im = Image.fromarray(arr).convert("L")  # grayscale
    im.save(fpath)


def calc_mse(arr1, arr2):
    assert(arr1.shape == arr2.shape)
    return mean_squared_error(arr1, arr2)

def normalize01(arr):
    return (arr - np.min(arr)) / np.ptp(arr)
