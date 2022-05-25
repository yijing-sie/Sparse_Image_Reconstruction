# Sparse_Image_Reconstruction

In imaging problems, an original image of interest can be corrupted by noisy measurements, leading to a deteriorated observed image. The goal of image reconstruction is to then recover the original signal as closely as possible. These imaging problems can often be formulated as an inverse problem involving the recovery of an image $ x^* \in R ^n $ from $ n $ noisy measurements. This can be expressed as an inverse problem $ y = Ax \text{*} + e $ where $ y \in R ^m$ is the observed corrupted image, $ A \in R ^ {mxn} $ is the measurement operator and $ e \in R ^ m $ is possible added noise.

We explore sparse image reconstruction where we assume the observed image is sparse under the wavelet domain and we seek to recover this sparse representation. Specifically, given an image corrupted by an unknown mask creating random regions of blank pixels, we aim to recover the original image utilizing optimization techniques. This task can be compared to compressed sensing techniques which aim to recover a sparse vector $ x \text{*} $ from $ m < n $ measurements. We assume some prior properties of the measurement mask and compare a few algorithms to observe the image reconstruction quality.

## Methods

We consider three methods for optimizing sparse image reconstruction: Soft Thresholding, Hard Thresholding, and Reciprocal Thresholding. Refer to the official [Report](/report.pdf) for detailed formulation as well as qualitative/quantitative results.

## Sample Outputs

Shown below are a few reconstructed outputs. Further examples can be found in the same report.

<p float="left">
  <img src="/gray/gray_1.jpg" width="100" />
  <img src="/corrupted/corrupted_1.jpg" width="100" /> 
</p>
