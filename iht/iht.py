import argparse
import os
import sys
from PIL import Image
import numpy as np
import pywt
import scipy.io
import cv2
import matplotlib.pyplot as plt
from iht.utils import *


def A(x, mask):
    return x * mask

def AAdj(x, mask):
    return A(x, mask)  # adjoint is the same

def iht(y, x_init, sparsity, mask, level, wavelet, iters=1000, alpha=0.1):
    K = round(len(y.flatten()) * sparsity)
    Kprime = len(y.flatten()) - K
    xhat = x_init

    for i in range(iters):
        gradient_x = -AAdj(y - A(xhat, mask), mask)
        xhat = xhat - alpha * gradient_x

        # project into wavelet
        coeffs = pywt.wavedec2(xhat, wavelet=wavelet, level=level, mode=pywt.Modes.periodization)

        # calculate support and zero out values not in support
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        coeff_arr = coeff_arr.flatten()
        idxs = np.argsort(np.abs(coeff_arr))  # make sure to sort on absolute value of wavelet coefficients
        coeff_arr[idxs[:Kprime]] = 0
        coeff_arr = coeff_arr.reshape(xhat.shape)

        # inverse wavelet
        icoeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
        xhat = pywt.waverec2(icoeffs, wavelet=wavelet, mode=pywt.Modes.periodization)

        y_recon = A(xhat, mask=mask)

        diff = y - y_recon
        norm_residual = np.linalg.norm(diff) / np.linalg.norm(y)
        print(f"iteration {i+1}/{iters}. Residual Norm: {norm_residual}")

    return xhat


def main(args):
    img_arr = imagepath2numpy(args.image_pth)
    mask = scipy.io.loadmat(args.mask_pth)['mask']

    img_name = os.path.basename(args.image_pth).split('.')[0]

    # resize to args.resolution
    img_resize = cv2.resize(img_arr, (args.resolution, args.resolution), interpolation=cv2.INTER_CUBIC)
    mask_resize = mask[:args.resolution, :args.resolution]
    assert(mask_resize.shape == img_resize.shape)

    corrupted_img = mask_resize * img_resize

    # initialize and run IHT
    x_init = np.random.rand(*corrupted_img.shape) * 255
    xhat = iht(y=corrupted_img, x_init=x_init, sparsity=args.sparsity, mask=mask_resize, level=4, wavelet='db4', iters=args.iters)

    # save reconstructed image, original processed image, corrupted image
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created New Directory {args.out_dir}")

    recon_img_path = os.path.join(args.out_dir, img_name + "_recon.jpg")
    orig_process_img_path = os.path.join(args.out_dir, img_name + ".jpg")
    corrupted_img_path = os.path.join(args.out_dir, img_name + "_corrupted.jpg")

    mse = calc_mse(normalize01(img_resize), normalize01(xhat))
    print(f"MEAN SQUARED ERROR: {mse}")

    save_nparray2image(xhat, recon_img_path)
    save_nparray2image(corrupted_img, corrupted_img_path)
    save_nparray2image(img_resize, orig_process_img_path)


def sparsity_vs_mse(args):
    img_arr = imagepath2numpy(args.image_pth)
    mask = scipy.io.loadmat(args.mask_pth)['mask']

    # resize to args.resolution
    img_resize = cv2.resize(img_arr, (args.resolution, args.resolution), interpolation=cv2.INTER_CUBIC)
    mask_resize = mask[:args.resolution, :args.resolution]
    assert(mask_resize.shape == img_resize.shape)

    corrupted_img = mask_resize * img_resize

    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    mses = []

    # initialize and run IHT
    x_init = np.random.rand(*corrupted_img.shape) * 255

    for sparsity in sparsities:
        xhat = iht(y=corrupted_img, x_init=x_init, sparsity=sparsity, mask=mask_resize, level=4, wavelet='db4', iters=args.iters)
        mse = calc_mse(normalize01(img_resize), normalize01(xhat))
        mses.append(mse)

    print(mses)

    plt.plot(sparsities, mses)
    plt.title("Mean Squared Error Reconstruction vs Sparsity Level")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("MSE")
    plt.xticks(sparsities)
    plt.savefig(os.path.join(args.out_dir, "mse_vs_sparsity.png"))
    plt.close()


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_pth', type=str, default='data/image.png', help="path to the input image")
    parser.add_argument('--mask_pth', type=str, default='mask.mat', help="path to mask")
    parser.add_argument('--resolution', type=int, default=512, help="resolution of images to be resized to")
    parser.add_argument('--sparsity', type=float, default=0.1, help='percentage of how sparse the signal in wavelet domain will be for IHT')
    parser.add_argument('--out_dir', type=str, default='output', help="output directory path to save reconstructed image. Directory will be created if it doesn't exist")
    parser.add_argument('--iters', type=int, default=1000, help="number of iterations to run algorithm")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    main(args)
    # sparsity_vs_mse(args)
