#!/usr/bin/env python3
import math
import sys
import numpy as np
import pandas as pd
import kmeans


def find_closest_centroid(point, centroids):
    """
    Given list of centroids and a point, finds the closest centroid.
    point is a pandas 1-dimensional dataFrame, centroids is a list of pandas 1-dimensional dataFrames.
    """
    min_dist = float("inf")
    for centroid in centroids:
        dist = distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def distance(point1, point2): 
    """ 
    Calculates euclidean distance between two points represented as 1-dimensional dataFrames
    """
    cur_sum = 0
    dim = len(point1.columns)
    for i in range(1, dim):
        cur_sum += (point1.iloc[0, i] - point2.iloc[0, i]) ** 2
    return math.sqrt(cur_sum)

def read_file(file_path):
    """ 
    Reads file and returns an np array of the vectors in the file.
    Assumes each line in the file is formatted as: float1,float2,float3,... 
    """
    with open(file_path, 'r') as file:
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

def input_handling():
    """
    Gets user inputted parameters, assumes it gets: number of clusters, (number of iterations), epsilon, data file 1, data file 2.
    Checks parameters for validity and merges both data files into one DataFrame according to the first column.
    Returns all parameters and merged DataFrame
    """
    k = sys.argv[1]
    if (len(sys.argv) == 6):
        iter = sys.argv[2]
        epsilon = sys.argv[3]
        data_file1 = sys.argv[4]
        data_file2 = sys.argv[5]
    else:
        iter = 300
        epsilon = sys.argv[2]
        data_file1 = sys.argv[3]
        data_file2 = sys.argv[4]

    data_array1 = read_file(data_file1)
    data_array2 = read_file(data_file2)

    data1_dict = {
        "Index": data_array1[0, :],  # First column as "Index"
    **{f"Column{i}": data_array1[i, :] for i in range(1, data_array1.shape[0])}  # Other columns as "Column{i}"
    }
    data2_dict = {
        "Index": data_array2[0, :],  # First column as "Index"
    **{f"Column{i}": data_array2[i, :] for i in range(1, data_array2.shape[0])}  # Other columns as "Column{i}"
    }

    data1 = pd.DataFrame(data1_dict)
    data2 = pd.DataFrame(data2_dict)
    data = pd.merge(data1, data2, on='Index')
    data = data.sort_values(by=['Index'])

    try:
        k = int(k)
        if (k > data_array1.shape[1]):
            print("Invalid number of clusters!")
            sys.exit(1)
        elif (k <= 0):
            print("Invalid number of clusters!")
            sys.exit(1)
    except ValueError:
        print("Invalid number of clusters!")
        sys.exit(1)

    try:
        iter = int(iter)
        if (iter > 1000):
            print("Invalid maximum iteration!")
            sys.exit(1)
        elif (iter <= 0):
            print("Invalid maximum iteration!")
            sys.exit(1)
    except ValueError:
        print("Invalid maximum iteration!")
        sys.exit(1) 

    try:
        epsilon = float(epsilon)
        if (epsilon < 0):
            print("Invalid epsilon!")
            sys.exit(1)
    except ValueError:
        print("Invalid epsilon!")
        sys.exit(1)

    return k, iter, epsilon, data

if __name__ == '__main__':
    centroids_lst = []
    k, iter, epsilon, merged_data = input_handling()
    n = len(merged_data)
    np.random.seed(1234)
    frst_centroid = merged_data[merged_data['Index'] == float(np.random.choice(range(0,n)))]
    centroids_lst.append(frst_centroid)
    D_arr = np.empty(shape=[1,n])  # Creating empty array of distances of points to centroids
    P_arr = np.empty(shape=[1,n])  # Creating empty array of probabilities
    for i in range(k - 1):
        for j in range(n):
            D_arr[0, j] = find_closest_centroid(merged_data[merged_data['Index'] == float(j)], centroids_lst)
        D_arr_sum = D_arr.sum()
        for j in range(n):
            P_arr[0, j] = (D_arr[0, j] / D_arr_sum)
        rnd_DataPoint = merged_data[merged_data['Index'] == float(np.random.choice(range(0,n), p=P_arr[0, :]))] # Choosing random data point, 
        # we wont get repeats because the probability of choosing a point that's already a Centroid is 0
        centroids_lst.append(rnd_DataPoint)
    new_centroid_lst = []
    indexes_lst = []

    for centroid in centroids_lst:
        indexes_lst.append(int(centroid.iloc[0, 0]))
        centroid = centroid.iloc[0, 1:] # Slicing the first column away from the centroid
        centroid = centroid.tolist()
        new_centroid_lst.append(centroid)

    # Convering merged data into list of list of doubles
    datapoints = []
    for i in range(n):
        datapoint = merged_data[merged_data["Index"] == float(i)]
        datapoint = datapoint.iloc[0, 1:] # Slicing the first column away
        datapoint = datapoint.tolist()
        datapoints.append(datapoint)
    
    dim = len(merged_data.columns) - 1
    final_centroids = kmeans.kmeans_C(new_centroid_lst, datapoints, n, k, dim, k, iter, epsilon)    
    print(",".join(map(str,indexes_lst)))
    for centroid in final_centroids:
        centroid = ",".join(f"{coord:.4f}" for coord in centroid)
        print(f"{centroid}")
