# Ray Tracer

## Overview

This project implements a basic **ray tracer** as part of the *Computer Graphics – Spring 2024* course. 
The ray tracer simulates light rays traveling through a scene, detecting intersections with surfaces, and computing colors based on material properties and lighting conditions.

## Features
- **Surface Types**: 
  - Spheres
  - Infinite planes
  - Axis-aligned cubes
- **Materials**:
  - Diffuse and specular reflection
  - Phong shading model
  - Reflective and transparent surfaces
- **Lighting**:
  - Point light sources
  - Specular intensity control
  - Soft shadows via multiple shadow rays
- **Camera & Scene Settings**:
  - Position, look-at point, and up vector
  - Screen distance and width
  - Background color, recursion limits, and shadow settings

## Running the Ray Tracer

To run the ray tracer, use the following command:

```sh
python raytracer.py scenes/example.txt output/image.png --width 500 --height 500
```
-First argument is the scene file
-Second argument is the output image file
-the `--width` and `--height` parameters are optional (defaults to 500x500)

## Scene file format
The scene is defined using a text based format. Each line specifies an object, material, or setting. Example:
```sh
# Camera settings - Positioned diagonally looking at the intersection of planes
cam   4 4 4   1 1 0   0 1 0   2.0   4.0

# Scene settings
set   0.1 0.1 0.1   3   2

# Material:	dr    	dg    	db	sr   	sg   	sb 	rr   	rg  	rb	phong 	trans
# Green material for planes
mtl		0.3	0.8	0	0	0	0	0	0	0	1	0
# White material for spheres
mtl   		1.0 	1.0 	1.0   	0.5 	0.5 	0.5   	0.2 	0.2 	0.2   	50   	0
# Red material for boxes
mtl   		0.8 	0.1 	0.1   	0.5 	0.5 	0.5   	0.0 	0.0 	0.0   	30   	0
# Black material for spheres
mtl		0	0	0	0.5	0.5	0.5	0	0	0	80	0
# Purple-ish clear material for big sphere
mtl		0.5	0.5	0.8	1	1	1	0.1	0.1	0.1	20	0.65

# Plane:	nx	ny	nz	offset	mat_idx
# Green planes 
#pln   		1 	0	0   	-1   	1
pln   		0 	1 	0   	0   	1
#pln   		0 	0 	1   	-1   	1

# White spheres in a column, sitting on a box
# Spheres:	cx   	cy   	cz  	radius 	mat_idx
sph   		2 	1.3 	2   	1.0   	2
sph   		2 	2.3 	2   	0.6   	2
sph   		2 	2.95 	2   	0.45   	2
sph		2.37	3	1.8	0.1	4
sph		2.37	3	2.2	0.1	4
sph		2	2	2	1.6	5

# Boxes:	cx	cy	cz	scale	mat_idx
box   		2 	0 	2   	2.0   	3

# Light sources above and diagonal to the spheres
# Lights:	px	py	pz	r	g	b	spec	shadow	width
lgt   		2 	5 	4   	1.0 	1.0 	1.0   	0.7   	0.5   	0.5
lgt   		4 	5 	2   	1.0 	1.0 	1.0   	0.7   	0.5   	0.5
```

## Requirements
The project uses Python and requires the following libraries:
-`numpy`
-`PIL`

## Results
Rendered images are found in the `output/` directory.

## Contributors
Alon Zajicek
