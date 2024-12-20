#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>


typedef struct DataPoint {
    /* Represents a data point */
    double *coords;
    /* Double array of vector coordinates */
    int dim;
    /* Dimension of the vector */
    int index;
    /* Property that helps with debugging */
    struct Cluster *cluster;
    /* Pointer to the cluster the point belongs to */
} DataPoint;

typedef struct Cluster {
    /* Represents a cluster */
    DataPoint *centroid;
    /* Pointer to a DataPoint that's the mean of all points in the cluster */
    DataPoint **points; 
    /* Array of DataPoints that belong to the cluster */
    int num_points;
    /* Number of points in the cluster */
    int capacity;
    /* Total capacity of points that the cluster can hold. Currently of dubious usefulness */
    int index;
    /* This is used for debugging purposes */
} Cluster;

void print_arr(double*, int); /* Function for the purposes of debugging */
void add_point(Cluster*, DataPoint*);
void remove_point(Cluster*, DataPoint*);
double update_centroid(Cluster*);
double distance(DataPoint*, DataPoint*);
void assign_to_closest(DataPoint*, Cluster**, int);
int get_dimension(const char*);


void add_point(Cluster *cluster, DataPoint *point) {
    /* Adds point to a cluster */
    void *tmp;
    /* Increasing cluster capacity by 1 each time a point is added. 
    This could get into a situation where we constantly increase and decrease the size of the point array in a cluster by 1, meaning a lot of memory access */
    cluster->capacity = cluster->capacity + 1;
    tmp = realloc(cluster->points, (size_t)cluster->capacity * sizeof(DataPoint *));
    if (tmp == NULL) {
        fprintf(stderr, "Memory allocation failed in add_point\n");
        exit(EXIT_FAILURE);
    }
    cluster->points = tmp;
    cluster->points[cluster->num_points++] = point; 
    point->cluster = cluster;
}

void print_arr(double* arr, int len) {
    /* Function that simply prints an array of doubles. Used for debugging purposes */
    int i;
    for (i = 0; i < len; i++) {
        printf("%f, ",arr[i]);
    }
    printf("\n");
}

void remove_point(Cluster *cluster, DataPoint *point) {
    /* Removes point from cluster */
    int index = -1;
    /* Finding the index of the point to be removed */
    int i;
    void *tmp;
    for (i = 0; i < cluster->num_points; i++) {
        if (cluster->points[i] == point) {
            index = i;
            break;
        }
    }
    /* I'm assuming this function will only be called for points that are definitely in the point array,
    So I don't bother dealing with potentially not finding the element in the array */

    /* Move the last point to the current index, the order of the point array doesn't matter */
    cluster->points[index] = cluster->points[--cluster->num_points];

    cluster->capacity = cluster->capacity - 1;
    tmp = realloc(cluster->points, (size_t)cluster->capacity * sizeof(DataPoint *));
    if (tmp == NULL) {
        fprintf(stderr, "Memory allocation failed in remove_point\n");
        printf("Cluster capacity = %i\n", cluster->capacity);
        exit(EXIT_FAILURE);
    }
    cluster->points = tmp;
    point->cluster = NULL;
}

double update_centroid(Cluster *cluster) {
    /* Updates centroid of a cluster. Pointwise mean of all vectors in the cluster */
    int i;
    int j;
    double *new_coords;
    double delta;
    DataPoint *new_centroid;

    new_coords = (double*)calloc((size_t)cluster->centroid->dim, sizeof(double));
    if (new_coords == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    new_centroid = (DataPoint*)malloc(sizeof(DataPoint));
    if (new_centroid == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < cluster->num_points; i++) {
        for (j = 0; j < cluster->centroid->dim; j++) {
            new_coords[j] += cluster->points[i]->coords[j];
        }
    }

    for (i = 0; i < cluster->centroid->dim; i++) {
        new_coords[i] = new_coords[i] / cluster->num_points;
    }
    
    new_centroid->coords = new_coords;
    new_centroid->dim = cluster->centroid->dim;
    new_centroid->index = -1;
    new_centroid->cluster = NULL;
    delta = distance(cluster->centroid, new_centroid);

    if (cluster->centroid->index == -1) {
        free(cluster->centroid->coords);
        free(cluster->centroid);
    }

    cluster->centroid = new_centroid;

    return delta;
}

double distance(DataPoint *point1, DataPoint *point2) {
    /* Calculate Euclidean distance between two points */
    double cur_sum = 0.0;
    int i;
    for (i = 0; i < point1->dim; ++i) {
        cur_sum += pow((point1->coords[i] - point2->coords[i]), 2);
    }
    return sqrt(cur_sum);
}

void assign_to_closest(DataPoint *point, Cluster **clusters, int num_clusters) {
    /* Assigns a point to its closest cluster */
    double min_dist = INFINITY;
    Cluster *target_cluster = NULL;
    Cluster *cur_cluster = point->cluster;

    /* Search for closest cluster */
    int i;
    for (i = 0; i < num_clusters; ++i) {
        double dist = distance(point, clusters[i]->centroid);
        if (dist < min_dist) {
            min_dist = dist;
            target_cluster = clusters[i];
        }
    }

    /* Add the current point to the cluster and remove it from its previous one */
    if (target_cluster != cur_cluster) {
        add_point(target_cluster, point);
        if (cur_cluster != NULL) {
            remove_point(cur_cluster, point);
        }
        point->cluster = target_cluster;
    }

}

int get_dimension(const char *line) {
    /* Gets dimension of a vector given as a string. Assumes the string is formatted "float1,float2,...".
    Used for debugging purposes. */
    int dim = 1;
    const char *p;
    for (p = line; *p != '\0'; p++) {
        if (*p == ',') {
            dim++;
        }
    }
    return dim;
}


DataPoint** DataPoint_structarray_from_array(double** arr, int arr_size, int dim) {
    /* Takes array of vectors and returns an array of DataPoints */
    int i;
    DataPoint** datapoints;

    datapoints = (DataPoint**)malloc(sizeof(DataPoint*) * arr_size);
    if (datapoints == NULL) {
        printf("Error: Could not allocate memory for data points\n");
        exit(EXIT_FAILURE);
    }

    /* Looping over the vector array, in each iteration we create a DataPoint and put it in the DataPoint array */
    for (i = 0; i < arr_size; ++i) {
        datapoints[i] = (DataPoint*)malloc(sizeof(DataPoint));
        if (datapoints[i] == NULL) {
            printf("Error: Could not allocate memory for data point\n");
            exit(EXIT_FAILURE);
        }

        datapoints[i]->coords = arr[i];
        datapoints[i]->dim = dim;
        datapoints[i]->cluster = NULL;
        datapoints[i]->index = i;
    }
    return datapoints;
}


int* get_centroid_indexes(double** centroids, DataPoint** datapoints, int datapoints_size, int centroids_size, int dim) {
    /* Function to find the indexes of the centroids in the datapoints array */
    int i;
    int j;
    int k;
    int match;
    int* idx_arr = (int*)calloc(centroids_size, sizeof(int));
    if (idx_arr == NULL) {
        printf("Error: Could not allocate memory for indexes\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < centroids_size; ++i) {
        for (j = 0; j < datapoints_size; ++j){
            match = 1;
            for (k = 0; k < dim; ++k) {
                if (datapoints[j]->coords[k] != centroids[i][k]) {
                    match = 0;
                    break;
                }
            }
            if (match == 1) {
                idx_arr[i] = j;
                break;
            }
        }
    }
    return idx_arr;
}

Cluster** Clusters_from_centroid_array (double** centroids, int size_of_centroids, int dim, DataPoint** datapoints, int size_of_datapoints) {
    /* Given an array of vectors representing centroids, creates an array of Clusters */
    Cluster** clusters;
    int i;
    int* idx_array = get_centroid_indexes(centroids, datapoints, size_of_datapoints, size_of_centroids, dim);
    clusters = (Cluster**)malloc(sizeof(Cluster*) * size_of_centroids);
    if (clusters == NULL) {
        printf("Error: Could not allocate memory for clusters\n");
        exit(EXIT_FAILURE);
    }
    
    /* Looping over the centroids array, creating a Cluster at each iteration, and assigning it a matching centroid from the centroids array */
    for (i = 0; i < size_of_centroids; ++i) {
        clusters[i] = (Cluster*)malloc(sizeof(Cluster));
        if (clusters[i] == NULL) {
            printf("Error: Could not allocate memory for cluster\n");
            exit(EXIT_FAILURE);
        } 
        clusters[i]->centroid = datapoints[idx_array[i]];
        clusters[i]->capacity = 1;
        clusters[i]->num_points = 1;
        clusters[i]->index = (double)i;
        clusters[i]->points = (DataPoint**)malloc(sizeof(DataPoint*));
        if (clusters[i]->points == NULL) {
            printf("Error: Could not allocate memory for data points\n");
            exit(EXIT_FAILURE);
        }
        clusters[i]->points[0] = datapoints[idx_array[i]];
        datapoints[idx_array[i]]->cluster = clusters[i];
    }
    free(idx_array);

    return clusters;
}


double** fit(double** centroids, double** datapoints, int datapoints_size, int centroids_size, int dim, int k, int iter, double epsilon) {
    /* fit function as described in the assignment */
    DataPoint** DataPoint_arr;
    Cluster** clusters;
    double** final_centroids = (double**)calloc(centroids_size, sizeof(double*));
    int j;
    int i;
    int converged = 0;
    double delta;

    DataPoint_arr = DataPoint_structarray_from_array(datapoints, datapoints_size, dim);
    clusters = Clusters_from_centroid_array(centroids, centroids_size, dim, DataPoint_arr, datapoints_size);

    /* Running k-means clustering */
    i = 0;
    while ((!converged) && (i < iter)) {
        /* Assign points to closest cluster */
        for (j = 0; j < datapoints_size; j++) {
            assign_to_closest(DataPoint_arr[j], clusters, k);
        }

        /* Update centroids */
        converged = 1;
        for (j = 0; j < k; j++) {
            delta = update_centroid(clusters[j]);
            if (delta >= epsilon) {
                converged = 0;
            }
        }
        ++i;
    }
    
    /* Assign final centroids to array of arrays of doubles */
    for (i = 0; i < centroids_size; ++i) {
        double* final_coords = (double*)calloc(dim, sizeof(double));
        for (j = 0; j < dim; ++j) {
            final_coords[j] = clusters[i]->centroid->coords[j];
        }
        final_centroids[i] = final_coords;
    }

    /* Cleanup */
    /* Free DataPoint array (note that the coordinates of each datapoint is not freed, those are freed elsewhere) */
    for (i = 0; i < datapoints_size; i++) {
        free(DataPoint_arr[i]);
    }
    free(DataPoint_arr);

    /* Free Cluster array */
    for (i = 0; i < centroids_size; i++) {
        if (clusters[i]->centroid->index == -1) {
            free(clusters[i]->centroid->coords);
            free(clusters[i]->centroid);
        }
        free(clusters[i]->points);
        free(clusters[i]);
    }
    free(clusters);

    return final_centroids;
}

static PyObject *fit_wrapper(PyObject *self, PyObject *args) {
    /* Wrapper function that takes arguments from Python, unpacks them, uses fit(), and finally returns the final centroids to the Python program. */
    PyObject *centroid_list; /* List of vectors of doubles gotten from Python */
    PyObject *datapoint_list; /* List of vectors of doubles gotten from Python */
    PyObject *centroid_item; /* Single vector from the centroid_list, also from Python */
    PyObject *datapoint_item; /* Single vector from the datapoint_list */
    PyObject *coord; /* Coordinate in a vector */
    double** centroids; /* Array of arrays of doubles, the actual centroid array as a result from fit() */
    double* centroid_vector; /* Vector in the centroid array */
    double** datapoints; /* Array of array of doubles, all the datapoints, converted from datapoint_list */
    double* datapoints_vector; /* Vector in the datapoint array */
    int datapoints_size; /* Size of the datapoint list, gotten dynamically from the datapoint_list */
    int centroids_size; /* Size of the centroid list, gotten dynamically from the centroid_list */
    int dim; /* Dimension of the vectors in datapoints */
    int k; 
    int iter;
    double epsilon;
    double** final_centroids; /* Result of the fit() function */
    int i;
    int j;

    if (!PyArg_ParseTuple(args, "OOiiiiid", &centroid_list, &datapoint_list, &datapoints_size, &centroids_size, &dim, &k, &iter, &epsilon)) {
        return NULL;
    }

    /* Unpack centroid Python list into array of arrays of doubles. */
    centroids_size = PyObject_Length(centroid_list);
    if (centroids_size < 0) {
        return NULL;
    }
    centroids = (double**)malloc(centroids_size * sizeof(double*));
    for (i = 0; i < centroids_size; ++i) {
        centroid_item = PyList_GetItem(centroid_list, i);
        centroid_vector = (double*)malloc(dim * sizeof(double));
        for (j = 0; j < dim; ++j) {
            coord = PyList_GetItem(centroid_item, j);
            centroid_vector[j] = PyFloat_AsDouble(coord);
        }
        centroids[i] = centroid_vector;
    }

    /* Unpack datapoint Python list into array of arrays of doubles. */
    datapoints_size = PyObject_Length(datapoint_list);
    if (datapoints_size < 0) {
        return NULL;
    }
    datapoints = (double**)malloc(datapoints_size * sizeof(double*));
    for (i = 0; i < datapoints_size; ++i) {
        datapoint_item = PyList_GetItem(datapoint_list, i);
        datapoints_vector = (double*)malloc(dim * sizeof(double));
        for (j = 0; j < dim; ++j) {
            coord = PyList_GetItem(datapoint_item, j);
            datapoints_vector[j] = PyFloat_AsDouble(coord);
        }
        datapoints[i] = datapoints_vector;
    }

    /* Run k-means algorithm */
    final_centroids = fit(centroids, datapoints, datapoints_size, centroids_size, dim, k, iter, epsilon);

    /* Cleanup */
    /* Freeing datapoint array (Note that this also frees the coordinates assigned to each DataPoint struct) */
    for (i = 0; i < datapoints_size; i++) {
        free(datapoints[i]);
    }
    free(datapoints);

    /* Freeing centroid array */
    for (i = 0; i < centroids_size; i++) {
        free(centroids[i]);
    }
    free(centroids);

    // Create a new Python list to hold the lists of doubles
    PyObject* final_centroids_list = PyList_New(centroids_size);
    if (!final_centroids_list) {
        return NULL; // Memory allocation failed
    }

    // Populate the Python list with lists of doubles
    for (int i = 0; i < centroids_size; ++i) {
        PyObject* cent_vector = PyList_New(dim);

        for (int j = 0; j < dim; ++j) {
            PyObject* num = Py_BuildValue("f", final_centroids[i][j]);
            PyList_SET_ITEM(cent_vector, j, num); 
        }

        PyList_SET_ITEM(final_centroids_list, i, cent_vector); 
    }
    /* Freeing final_centroids */
    for (i = 0; i < centroids_size; ++i) {
        free(final_centroids[i]);
    }
    free(final_centroids);

    return final_centroids_list;
}

static PyMethodDef kmeansMethods[] = {
    {
        "kmeans_C",
        (PyCFunction) fit_wrapper,
        METH_VARARGS,
        PyDoc_STR("Returns the final centroids using the kmeans algorithm.\n Args:\ncentroids - array of array of doubles, the centroid array.\ndatapoints - array of array of doubles, the datapoints array.\ndatapoints_size - integer, size of the datapoints array.\ncentroids_size - integer, size of the centroid array.\ndim - integer, dimension of the datapoint/centroid vectors.\nk - integer, number of final centroids.\niter - integer, maximum number of iterations.\nepsilon - double, epsilon.")},
         {
        NULL, NULL, 0, NULL
        }
};

static struct PyModuleDef Kmeans_Module = {
    PyModuleDef_HEAD_INIT,
    "kmeans",
    "Kmeans Python wrapper",
    -1,
    kmeansMethods
};

PyMODINIT_FUNC PyInit_kmeans(void) {
    return PyModule_Create(&Kmeans_Module);
}

