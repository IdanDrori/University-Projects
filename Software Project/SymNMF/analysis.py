#!/usr/bin/env python3
import math
import sys
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import symnmf
import symnmf_module
import numpy as np

''' Kmeans.py from assignment 1 '''
class Cluster:
    def __init__(self, centroid: 'DataPoint' = None) -> None:
        if centroid is not None:
            self.points = [centroid]
        else:
            self.points = []
        self.centroid: DataPoint = centroid

    def add_point(self, point: 'DataPoint') -> None:
        self.points.append(point)

    def remove_point(self, point: 'DataPoint') -> None:
        self.points.remove(point)

    def update_centroid(self) -> float:
        #Updates the centroid of a cluster to be the pointwise mean of its vectors     
        d = self.centroid.dim
        k = len(self.points)

        new_coords = [0 for i in range(d)]
        for point in self.points:
            for i in range(d):
                new_coords[i] += point.coords[i]
        
        new_coords = [x/k for x in new_coords]
        new_centroid = DataPoint(new_coords, -1)
        delta = self.centroid.distance(new_centroid)
        self.centroid = new_centroid
        return delta

class DataPoint:
    def __init__(self, coords: 'list[float]', index: int) -> None:
        self.coords = coords
        self.dim = len(coords)
        self.index = index
        self.cluster = None

    def __repr__(self) -> str:
        formatted_coords = ",".join(f"{coord:.4f}" for coord in self.coords)
        return f"{formatted_coords}"
    
    def distance(self, other: 'DataPoint') -> float:
        # Calculate Euclidean distance between data points 
        cur_sum = 0
        for i in range(self.dim):
            cur_sum += (self.coords[i] - other.coords[i]) ** 2
        return math.sqrt(cur_sum)
    
    def assign_to_closest(self, clusters: 'list[Cluster]') -> None:
        # Assign DataPoint to its closest cluster
        min_dist = float("inf")
        target_cluster = None

        # Search for closest cluster
        for cluster in clusters:
            distance = self.distance(cluster.centroid)
            if distance < min_dist:
                min_dist = distance
                target_cluster = cluster

        # Adds the current point to the cluster and removes it from the previous one
        if target_cluster != self.cluster:
            target_cluster.add_point(self)
            if self.cluster is not None:
                self.cluster.remove_point(self)
            self.cluster = target_cluster

def input_handling() -> 'tuple[int, int, list[list[DataPoint]]]':
    k = sys.argv[1]
    iter = 300
    data_file = sys.argv[2]
    vector_list = read_file(data_file)
    n = len(vector_list)
    try:
        k = int(k)
        if k >= n:
            print("Invalid number of clusters!")
            sys.exit(1)
    except ValueError:
        print("Invalid number of clusters!")
        sys.exit(1)
    return (k, iter, vector_list)

def read_file(file_path: str) -> 'list[list[DataPoint]]':
    # Reads a text file where each line is formatted as "float1,float2,...", and returns a list of DataPoints
    vectors = []
    with open(file_path, 'r') as file:
        i = 0
        line = file.readline()
        while line:
            vector = [float(x) for x in line.split(",")]
            vector = DataPoint(vector, i)
            vectors.append(vector)
            i += 1
            line = file.readline()
    return vectors


def kmeans():
    epsilon = 0.0001
    k, iter, vectors = input_handling()
    # Initializing k clusters
    clusters = []
    for i in range(k):
        clust = Cluster(vectors[i])
        vectors[i].cluster = clust
        clusters.append(clust)
    converged = False
    i = 0
    while (not converged) and (i < iter):
        # Assign vectors to their closest cluster
        for vector in vectors:
            vector.assign_to_closest(clusters)
        # Update the centroids
        converged = True
        for cluster in clusters:
            delta = cluster.update_centroid()
            if delta >= epsilon:
                converged = False
        i += 1    
    return clusters

def get_symnmf_clusters(final_h, vectors):
    final_H = np.array(final_h)
    cluster_list = [Cluster() for i in range(len(final_h[0]))]
    for i in range(len(final_H)):
        max_idx = np.argmax(final_H[i])
        cluster_list[max_idx].add_point(vectors[i])
    return cluster_list
        
def get_label_arr(clusters, num_datapoints):
    # Creates a label array whose length is the number of datapoints, where label_arr[i]=j,
    # where j is the index of cluster in the cluster array that the i-th point belongs to
    label_arr = np.zeros(num_datapoints)
    for j in range(len(clusters)):
        for point in clusters[j].points:
            label_arr[point.index] = j

def get_silhouette_score(clusters, vectors):
    X = pairwise_distances(vectors, metric='euclidean')
    label_arr = get_label_arr(clusters, len(vectors))
    return silhouette_score(X, label_arr, metric='precomputed')

def analysis():
    vectors = symnmf.read_file(sys.argv[2])
    mat_W = symnmf.norm_C(vectors, len(vectors))
    mat_w = np.array(mat_W)
    mat_H = symnmf.generate_mat_h(mat_w)
    final_h = symnmf_module.symnmf_C(mat_H, vectors, len(vectors))
    kmeans_clusters = kmeans()
    symnmf_clusters = get_symnmf_clusters(final_h, vectors)

    score_symnmf = get_silhouette_score(symnmf_clusters, vectors)
    print(f"nmf: {score_symnmf:.4f}")
    score_kmeans = get_silhouette_score(kmeans_clusters, vectors)
    print(f"kmeans: {score_kmeans:.4f}")

# Ask in the forum about whether we can get empty clusters from the symnmf result matrix