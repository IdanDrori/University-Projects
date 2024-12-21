# Introduction

The final project for the Software Project course - Implementation of the SymNMF algorithm.

Symmetric nonnegative matrix factorization (SymNMF) is an unsupervised algorithm for graph clustering. In this project we also compare its performance to the K-Means algorithm.

## SymNMF

Given a set of $N$ points as a matrix $X=x_1,x_2,...,x_n\in \mathbb{R}^{d}$:
1. Form the similarity matrix $A$ from $X$
2. Compute the diagonal degree Matrix $D$
3. Compute the normalized similarity matrix $W$
4. Find $H_{N\times k}$ that solves $\min_{H\geq 0} \|W-HH^T\|_F^2$

Where $k$ is a parameter that denotes the required number of clusters and $\|\|_F^2$ is the squared Frobenius norm.

## K-Means

Given a set of $N$ datapoints $X=x_1,x_2,...x_n\in\mathbb{R}^d$, the gaol is to group the data into $K\in\mathbb{N}$ clusters, each datapoint is assigned to exactly one cluster and the number of clusters $K$ is such that $1<K<N$.
Each cluster $k$ is represented by its centroid, which is the mean $\mu_k\in\mathbb{R}^d$ of the clusster's members. The algorithm is as follows:
1. Initialize centroid as first $k$ datapoints: $\mu_k=x_k, \forall k\in K$
2. Repeat:
   * Assign every $x_i$ to the closest cluster $k$: $argmin_k d(x_i,\mu_k),\forall k. 1\leq k\leq K$
   * Update the centroids: $\mu_k=\frac{1}{|k|}\sum_{x_i\in k} x_i$
3. Until convergence: $(\delta\mu_k<\varepsilon)\; OR\; (iteration\_number=iter)$

Where $d(p,q)$ is the Euclidean distance, and $\delta\mu_k$ is the Euclidean distance between the updated centroid to the previous one (this is checked for every centroid).

## Usage

### Running through Python
```bash
make build-python
python3 symnmf.py 2 symnmf input.txt
(symnmf, "k", "goal", "input file")
```

### Running C directly
```bash
make build-c
./symnmf sym input.txt
(symnmf, "goal", "input file")
```

### Running the  analysis
```bash
python3 analysis.py input.txt
```

# Contributions
* Ravid Prozan
