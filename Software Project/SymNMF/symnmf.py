import math
import sys
import numpy as np
import symnmf_module

def print_matrix(matrix):
    for row in matrix:
        print(','.join(f'{x:.4f}' for x in row))

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)
        i = 0
        line = lines[0]
        col_count = len([float(x) for x in line.split(",")])
        vectors = np.empty(shape=[col_count, line_count])

    for i in range(len(lines)):
        line = lines[i]
        vector = [float(x) for x in line.split(",")]
        vectors[:, i] = vector
        
    return vectors

def generate_mat_h(mat_w):
    m = np.mean(mat_W)
    mat_H = []
    for i in range(k):
        for j in range(len(vectors)):
             mat_H[i][j] = np.random.uniform(0, 2*math.sqrt(m / k))
    return mat_H

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
    np.random.seed(1234)    
    k = int(sys.argv[0])
    goal = sys.argv[1]
    filename = sys.argv[2]
    
    vectors = read_file(filename)
    mat_W = symnmf_module.norm_C(vectors, len(vectors))
    mat_w = np.array(mat_W)
    mat_H = generate_mat_h(mat_w)

    if goal == "symnmf":
        result = symnmf_module.symnmf_C(mat_H, vectors, len(vectors))
        print_matrix(result)

    elif goal == "sym":
        sym_mat = symnmf_module.sym_C(vectors, len(vectors))
        print_matrix(sym_mat)

    elif goal == "ddg":
        ddg_mat = symnmf_module.ddg_C(vectors, len(vectors))
        print_matrix(ddg_mat)

    elif goal == "norm":
        print_matrix(mat_w)


