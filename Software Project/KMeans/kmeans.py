#!/usr/bin/env python3
import math
import sys

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
    if (len(sys.argv) == 4):
        iter = sys.argv[2]
        data_file = sys.argv[3]
    else:
        iter = 200
        data_file = sys.argv[2]

    vector_list = read_file(data_file)
    n = len(vector_list)

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


if __name__ == '__main__':
    epsilon = 0.001
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

        print(f"iteration num = {i}")
        print(f"number of points in cluster[0] ={len(clusters[0].points)}")
        print(f"number of points in cluster[1] ={len(clusters[1].points)}")
        print(f"number of points in cluster[2] ={len(clusters[2].points)}")
        # Update the centroids
        converged = True
        for cluster in clusters:
            delta = cluster.update_centroid()
            if delta >= epsilon:
                converged = False
        i += 1
    
    # Printing results
    for cluster in clusters:
        print(cluster.centroid)
        
