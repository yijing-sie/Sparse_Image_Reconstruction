# Sparse Image Reconstruction

This is a group project of three students for the course  **Optimization**

* In this project, we explore sparse image reconstruction where we assume the observed gray image is sparse under some
domain, and the goal is to recover the original image as closely as possible via three iterative thresholding approaches: **Soft Thresholding**, **Hard Thresholding**, and **Reciprocal Thresholding**
 
* Each of us is responsoble for one of the methods, and I am responsible for the implementation of **[Reciprocal Thresholding](https://arxiv.org/pdf/1804.08841)**
 
* The sparse optimization problem is defined as follows:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\min\limits_{x\in\mathbb{R}^d} f(x)$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\text{s.t} ||x||_0 \leq s$

> The sparsity
constraint $||x||_0 \leq s$ requires that at most $s$ many number entries of the solution vector $x$ are non zero

> $f(x) : \mathbb{R}^d\to\mathbb{R}$ is a differentiable loss function




* Specifically, we combined Wavelet Transform with iterative thresholding algorithms for better performance

* My implementation of Reciprocal Thresholding is [irt.py](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/irt/reciprocal_thresholding.py)

Here is the report for our project : [Sparse Image Reconstruction](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/report.pdf) 


## Sample Outputs
---
Each image is corrupted by [mask.mat](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/mask.mat). More results can be found in our [report](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/report.pdf) 

**Original (left) v.s. Corrupted (right)**

<p float="left">
  <img src="/gray/gray_1.jpg" width="400" />
  <img src="/corrupted/corrupted_1.jpg" width="400" /> 
</p>

**Iterative Soft Thresholding (IST left) v.s. Iterative Hard Thresholding (IHT middle) v.s. Reciprocal Thresholding (right)**

<p float="left">
  <img src="/ist/output/rimg1.jpg" width="250" />
  <img src="/iht/output/test1_recon.jpg" width="250" /> 
  <img src="/irt/recon_1.jpg" width="250" /> 
</p>

## MSE reconstruction error
---
We utilize the Mean Squared Error (MSE) metric to compare the error between original and reconstructed images.For consistency, we normalize all values to be between 0 and 1 before calculating MSE:


Methods\Images | [Image 1](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/gray/gray_1.jpg) | [Image 2](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/gray/gray_2.jpg) | [Image 3](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/gray/gray_3.jpg) | [Image 4](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/gray/gray_4.jpg) | [Image 5](https://github.com/yijing-sie/Sparse_Image_Reconstruction/blob/master/gray/gray_5.jpg) 
--- | --- | --- | --- |--- |--- 
Iterative Soft Thresholding |0.0303|0.0219|0.0242|0.0187|0.0083 
Iterative Hard Thresholding |0.0483|0.0221|0.0379|0.0238|0.0152
Reciprocal Thresholding|0.025|0.0071|0.0135|0.00856|0.0074

**We conclude that Reciprocal Thresholding demonstrates better result for the purpose of our project**
