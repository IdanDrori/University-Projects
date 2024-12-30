import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse

def get_neighbors(idx, width, height):
    neighbors = []
    if idx % width > 0:  
        neighbors.append(idx - 1)
    if idx % width < width - 1: 
        neighbors.append(idx + 1)
    if idx // width > 0:  
        neighbors.append(idx - width)
    if idx // width < height - 1: 
        neighbors.append(idx + width)
    return neighbors

def poisson_blend(im_src, im_tgt, im_mask, center):
    h, w = im_mask.shape
    flat_mask = im_mask.astype(np.float64).flatten()
    flat_src = im_src.astype(np.float64).reshape(-1, 3)
    
    y_start = int(center[1] - h // 2)
    y_end = int(center[1] + h // 2 + (h % 2))
    x_start = int(center[0] - w // 2)
    x_end = int(center[0] + w // 2 + (w % 2))
    
    if y_start<0 or y_end >=im_tgt.shape[0] or x_start<0 or x_end>im_tgt.shape[1]:
        print('mask is bigger than target image')
        exit(1)
    cut_tgt = im_tgt[y_start:y_end, x_start:x_end]
    flat_cut = cut_tgt.astype(np.float64).reshape(-1, 3)
    
    A_data = []
    A_row_indices = []
    A_col_indices = []
    b = np.zeros((h * w, 3), dtype=np.float64)
    is_inside_mask = lambda idx : flat_mask[idx] > 0

    for idx in range(h * w):
        if is_inside_mask(idx):  
            neighbors = get_neighbors(idx, w, h)
            
            A_data.append(4.0)
            A_row_indices.append(idx)
            A_col_indices.append(idx)
            
            for neighbor in neighbors:
                if is_inside_mask(neighbor):
                    A_data.append(-1.0)
                    A_row_indices.append(idx)
                    A_col_indices.append(neighbor)
                else:
                    b[idx] += flat_cut[neighbor]
            
            b[idx] += 4.0 * flat_src[idx]
            for neighbor in neighbors:
                b[idx] -= flat_src[neighbor]
        else:
            A_data.append(1.0)
            A_row_indices.append(idx)
            A_col_indices.append(idx)
            b[idx] = flat_cut[idx]
    
    A = csr_matrix((A_data, (A_row_indices, A_col_indices)), shape=(h * w, h * w))
    blended = np.zeros_like(flat_cut)
    for channel in range(3):
        blended[:, channel] = spsolve(A, b[:, channel])
    
    blended = np.clip(blended, 0, 255).astype(np.uint8).reshape(h, w, 3)
    im_tgt[y_start:y_end, x_start:x_end] = blended
    
    return im_tgt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

import os

if __name__ == "__main__":
    # Load the source and target images
    args = parse()
    print(os.getcwd())
    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))
    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
