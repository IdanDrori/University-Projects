# Introduction
This repository contains implementations of the GrabCut algorithm for foreground segmentation and of Poisson Blending, an algorithm that blends source and target images, as part of the Computer Graphics, Image Processing and Vision course. In addition, we included a batch testing script and a report of the results of these two algorithms on a collection of images.

## Overview
### Foreground Segmentation Using GrabCut
GrabCut is an iterative algorithm for foreground segmentation. It uses Gaussian Mixture Models (GMMs) for classifying pixels into foreground and background and graph-cut optimization to minimize segmentation energy.

### Poisson Blending
Poisson blending is a technique for combining a foreground object with a new background while preserving texture and color consistency at the boundaries. It solves the Poisson equation to achieve seamless blending.

## Features

1. **Foreground Segmentation:**
   - Implements the GrabCut algorithm with GMMs for background and foreground classification.
   - Utilizes graph optimization for segmentation refinement.

2. **Poisson Blending:**
   - Computes the Laplacian operator.
   - Solves the Poisson equation using sparse matrix representations.
   - Supports blending foreground objects with larger target images.


## Collaborators
* Alon Zajicek
