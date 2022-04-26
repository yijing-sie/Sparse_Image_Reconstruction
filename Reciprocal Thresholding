# -*- coding: utf-8 -*-
"""
Resources: Between hard and soft thresholding optimal iterative thresholding algorithms
"""
from PIL import Image
import numpy as np
import scipy.io
import os
import pywt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
#%% LOAD IMAGEs and GET CORRUPTED IMAGEs
#Preprocessing
resize_shape = 512 #or 600
original_mask = scipy.io.loadmat("mask")["mask"] #600x600 mask
mask_reshape = original_mask[:resize_shape, :resize_shape] #reshape mask
#gray images are inside the folder gray
image_folder = "gray"
img_arr = [] #original image array
corr_img = [] #corrupted image array
for filename in os.listdir(image_folder):
    try: 
        img = Image.open(os.path.join(image_folder,filename)).resize((resize_shape, resize_shape), Image.BICUBIC)
        img_arr.append(np.array(img))
        assert(img_arr[-1].shape == mask_reshape.shape)
        corr_img.append(img_arr[-1] * mask_reshape)
    except:
        print("Something went wrong when opening this image : {}".format(filename))
#%% SHOW CORRUPTED IMAGE
# index = 0
# img = Image.fromarray(corr_img[index], 'L')
# img.show()
#%%
def A(x, mask):
    return x * mask

def AAdj(x, mask):
    return A(x, mask)  


def calc_mse(arr1, arr2):
    assert(arr1.shape == arr2.shape)
    sub = arr1 - arr2
    squared = np.square(sub)
    mse = np.sum(squared) / squared.size
    return mse
          
def project(z, tau):
    return np.sign(z)*(0.5*np.abs(z) + 0.5*np.sqrt(np.abs(z)**2  - tau**2))
def normalize(img):
    return (img - np.min(img)) / np.ptp(img)
#%%
"""
Reciprocal Thresholding
"""
def reciprocal_t(corr_img, x_init, sparsity, mask, level, wavelet, iters=1000, alpha=0.1):
    keep = round(corr_img.size * sparsity) # number of coefficeints to keep (entries inside the set)
    kprime = corr_img.size - keep # number of coefficeints to set to 0  (entries outside the set)
    recov_img = x_init

    for i in tqdm(range(iters)):
        gradient = -AAdj(corr_img - A(recov_img, mask), mask)
        recov_img = recov_img - alpha * gradient #gradient step

        # Perform Multilevel 2D Inverse Discrete Wavelet Transform. 
        coeffs = pywt.wavedec2(recov_img, wavelet=wavelet, level=level, mode=pywt.Modes.periodization)

        # Concatenating all coefficients into a single n-d array
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs) #coeff_arr = Wavelet transform coefficient array
        coeff_arr = coeff_arr.flatten()
        sort_idx = np.argsort(np.abs(coeff_arr), axis=None)  #sort the absolute coefficients from least to greatest
        tau = coeff_arr[sort_idx[kprime - 1]] #tau is the larget entry outside the set S
        coeff_arr[sort_idx[:kprime]] = 0
        mask_0 = coeff_arr != 0
        coeff_arr[mask_0] = project(coeff_arr[mask_0], tau=tau)#project step
        coeff_arr = coeff_arr.reshape(recov_img.shape)
        

        icoeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2') #the inverse of array_to_coeffs
        recov_img = pywt.waverec2(icoeffs, wavelet=wavelet, mode=pywt.Modes.periodization) # inverse of Multilevel 2D Inverse Discrete Wavelet Transform

        yhat = A(recov_img, mask=mask)

        diff = corr_img - yhat
        norm_residual = np.linalg.norm(diff) / np.linalg.norm(corr_img)
    
    print(f"Residual Norm: {norm_residual} ")
    return recov_img
#%%
img_shape = mask_reshape.shape
for idx,img in enumerate(corr_img):
    #initial guess of recovered image as random samples from a uniform distribution over [0, 255).
    x_init = np.random.rand(*img_shape)*255 
    recov_img = reciprocal_t(corr_img=img, x_init=x_init, sparsity=0.1, mask=mask_reshape, level=4, wavelet='db4')
    img = Image.fromarray(recov_img).convert("L")  # grayscale
    
    img.save(f"recon_{idx+1}.jpg")
    # img.show()
    mse = calc_mse(normalize(img), normalize(recov_img))
    print(f" MSE for {idx+1}.jpg = {mse}\n")
