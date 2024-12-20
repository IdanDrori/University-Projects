#!/usr/bin/env python3
import math
import sys
import numpy as np
import pandas as pd

# class Cluster:
#     def __init__(self, centroid: 'DataPoint' = None) -> None:
#         if centroid is not None:
#             self.points = [centroid]
#         else:
#             self.points = []
#         self.centroid: DataPoint = centroid

#     def add_point(self, point: 'DataPoint') -> None:
#         self.points.append(point)

#     def remove_point(self, point: 'DataPoint') -> None:
#         self.points.remove(point)

#     def update_centroid(self) -> float:
#         #Updates the centroid of a cluster to be the pointwise mean of its vectors     
#         d = self.centroid.dim
#         k = len(self.points)

#         new_coords = [0 for i in range(d)]
#         for point in self.points:
#             for i in range(d):
#                 new_coords[i] += point.coords[i]
        
#         new_coords = [x/k for x in new_coords]
#         new_centroid = DataPoint(new_coords, -1)
#         delta = self.centroid.distance(new_centroid)
#         self.centroid = new_centroid
#         return delta

# class DataPoint:
#     def __init__(self, coords: 'list[float]', index: int) -> None:
#         self.coords = coords
#         self.dim = len(coords)
#         self.index = index
#         self.cluster = None

#     def __repr__(self) -> str:
#         formatted_coords = ",".join(f"{coord:.4f}" for coord in self.coords)
#         return f"{formatted_coords}"
    
#     def distance(self, other: 'DataPoint') -> float:
#         # Calculate Euclidean distance between data points 
#         cur_sum = 0
#         for i in range(self.dim):
#             cur_sum += (self.coords[i] - other.coords[i]) ** 2
#         return math.sqrt(cur_sum)
    
#     def assign_to_closest(self, clusters: 'list[Cluster]') -> None:
#         # Assign DataPoint to its closest cluster
#         min_dist = float("inf")
#         target_cluster = None

#         # Search for closest cluster
#         for cluster in clusters:
#             distance = self.distance(cluster.centroid)
#             if distance < min_dist:
#                 min_dist = distance
#                 target_cluster = cluster

#         # Adds the current point to the cluster and removes it from the previous one
#         if target_cluster != self.cluster:
#             target_cluster.add_point(self)
#             if self.cluster is not None:
#                 self.cluster.remove_point(self)
#             self.cluster = target_cluster


def find_closest_centroid(point, centroids):
    min_dist = float("infty")
    for centroid in centroids:
        dist = point.distance(centroid)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)
        i = 0
        line = file.readline()
        col_count = len([float(x) for x in line.split(",")])
        vectors = np.empty(shape=[col_count, line_count])

        while line:
            vector = [float(x) for x in line.split(",")]
            vectors[i, :] = vector
            i += 1
            line = file.readline()
    return vectors

def input_handling():
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

    column_names = ['Index'] + [f'Column{i}' for i in range(1, data_array1.shape[1])] 
    data1 = pd.DataFrame(data_array1, column_names)
    data2 = pd.DataFrame(data_array2, column_names) # This syntax was not tested yet

    data = pd.merge(data1, data2, on='Index')
    
    try:
        k = int(k)
        k < n
    except ValueError:
        print("Invalid number of clusters!")
        sys.exit(1)

    try:
        iter = int(iter)
        iter < 1000
    except ValueError:
        print("Invalid maximum iteration!")
        sys.exit(1) 

    try:
        epsilon = float(epsilon)
        epsilon < 0
    except ValueError:
        print("Invalid epsilon!")
        sys.exit(1)

    return k, iter, epsilon, data

if __name__ == '__main__':
    centroids_lst = []
    k, iter, point_lst = input_handling()
    n = len(point_lst)
    np.random.seed(1234) 
    frst_centroid = np.random.choice(point_lst)
    D_arr = np.empty(shape=[0,n])  # Creating empty array of distances of points to centroids
    P_arr = np.empty(shape=[0,n])  # Creating empty array of probabilities

    for i in range(k):
        for i in range(n):
            D_arr[i] = find_closest_centroid(point_lst[i])
        D_arr_sum = D_arr.sum()
        for i in range(n):
            P_arr[i] = (D_arr[i] / D_arr_sum)

        rnd_DataPoint = np.random.choice(point_lst, p=P_arr) # Choosing random data point, 
        # we wont get repeats because the probability of choosing a point that's already a Centroid is 0
        centroids_lst.append(rnd_DataPoint)

