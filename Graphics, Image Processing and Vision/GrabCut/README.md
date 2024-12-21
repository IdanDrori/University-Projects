# Foreground Segmentation Using GrabCut

This repository contains my implementation of the GrabCut algorithm for foreground segmentation as part of the Computer Graphics, Image Processing and Vision course,
as well as a batch testing script, and a detailed report on results (To be added later). 
The algorithm iteratively refines a binary mask to segment an object from its background, using Gaussian Mixture Models (GMMs) and graph-based optimization techniques.

## Overview
The GrabCut algorithm, introduced by Rother, Kolmogorov and Blake, is a tool for interactive image segmentation.

### Rough outline of the algorithm
1. Initialization of GMMs
   * Initialize Gaussian Mixture Models (GMMs) for foreground and background using K-Means clustering.
2. Update GMMs
   * Calculate and update mean, covariance, and weights of GMM components based on the current mask.
3. Min-Cut Calculation
   * Build a graph using the mask and energy terms, then compute the min-cut to segment the image.
4. Mask Update
   * Iteratively update the mask based on the min-cut results, until convergence.
