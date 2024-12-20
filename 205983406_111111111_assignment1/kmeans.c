#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct DataPoint {
    double *coords;
    int dim;
    int index;
    struct Cluster *cluster;
} DataPoint;

typedef struct Cluster {
    DataPoint *centroid;
    DataPoint **points;
    int num_points;
    int capacity;
    int index;
} Cluster;

void add_point(Cluster*, DataPoint*);
void remove_point(Cluster*, DataPoint*);
double update_centroid(Cluster*);
double distance(DataPoint*, DataPoint*);
void assign_to_closest(DataPoint*, Cluster**, int);
int get_dimension(const char*);
DataPoint** read_data(int);
DataPoint* get_DataPoint(const char*, int);
int is_natural(char*);

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
    /* This function simply gets the dimension of a vector */
    int dim = 1;
    const char *p;
    for (p = line; *p != '\0'; p++) {
        if (*p == ',') {
            dim++;
        }
    }
    return dim;
}

DataPoint** read_data(int num_points) {
    /* Reads a txt file from standard input, and returns an array of DataPoints */
    int i = 0;
    char *buffer = NULL;
    size_t bufsize = 0;
    int nread;
    DataPoint **points;

    /*  Allocate memory for data points */
    points = (DataPoint**)malloc((size_t)num_points * sizeof(DataPoint*));
    if (points == NULL) {
        printf("Error: Could not allocate memory for data points\n");
        exit(EXIT_FAILURE);
    }

    /* Read the data points */
    while ((nread = (int)getline(&buffer, &bufsize, stdin)) != -1) {
        points[i] = get_DataPoint(buffer, i);
        i++;
    }
    free(buffer);
    return points;
}

DataPoint* get_DataPoint(const char* line, int index) {
    /* Reads a string formatted as "float1,float2,...", and returns a DataPoint */
    int idx = 0;
    int dim = get_dimension(line);
    double value = 0.0;
    DataPoint *point; 
    point = (DataPoint*)malloc(sizeof(DataPoint));
    if (point == NULL) {
        printf("Error: Could not allocate memory for data point\n");
        exit(EXIT_FAILURE);
    }

    point->coords = (double *)calloc((size_t)dim, sizeof(double));
    if (point->coords == NULL) {
        printf("Error: Could not allocate memory for coordinates\n");
        exit(EXIT_FAILURE);
    }

    /* Parse the string until the newline character or end of string */
    while (*line && *line != '\n') {
        /* Use sscanf to extract a double */
        if (sscanf(line, "%lf,", &value) == 1) {
            point->coords[idx++] = value;
            /* Move the line pointer to the next character after the comma */
            while (*line && *line != ',') {
                line++;
            }
            /* Skip the comma */
            if (*line == ',') {
                line++;
            }
        } 
        else {
            /* If sscanf fails, break the loop */
            break; 
        }

        if (idx > dim) {
            break; 
        }
    }

    point->cluster = NULL;
    point->dim = dim;
    point->index = index;
    return point;
}

int is_natural(char* str) {
    /* Checks whether a string properly represents a natural number, and returns it converted to int. If not a natural number returns -1*/
    double converted_str = atof(str);
    if (converted_str != 0.0) { /* String is valid float */
        if (converted_str == floor(converted_str)) { /* Converted string is an integer */
            return (int)converted_str;
        }
    }
    return -1;
}

void print_centroids(Cluster **clusters, int num_clusters) {
    /* Function goes over the cluster list and prints out their centroids */
    int i;
    int j;
    for (i = 0; i < num_clusters; i++) {
        for (j = 0; j < clusters[i]->centroid->dim; j++) {
            if (j > 0) printf(",");
            printf("%0.4f", clusters[i]->centroid->coords[j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    double epsilon = 0.001;
    int k;
    int iter;
    int i;
    int converged = 0;
    int j;
    int num_points = 0;
    char *buffer = NULL;
    size_t bufsize = 0;
    int nread;
    DataPoint **points;
    Cluster **clusters;
    double delta;

    /* Reading data file from stdin and constructing an array of DataPoints */
    /* Calculating the number of lines */
    while ((nread = (int)getline(&buffer, &bufsize, stdin)) != -1) {
        num_points++;
    }
    free(buffer);
    rewind(stdin);

    /* Both k and iter were provided by user */
    if (argc == 3) {
        k = is_natural(argv[1]);
        iter = is_natural(argv[2]);
        if (iter <= 0 || iter >= 1000) {
            printf("Invalid maximum iterations!\n");
            exit(EXIT_FAILURE);
        }
    }
    /* Only k was provided by user */
    else {
        k = is_natural(argv[1]);
        iter = 200;
    }

    if (k <= 0 || k >= num_points) {
            printf("Invalid number of clusters!\n");
            exit(EXIT_FAILURE);
        }

    /* Initializing k clusters */
    points = read_data(num_points);
    clusters = (Cluster**)malloc(sizeof(Cluster*) * (size_t)k);
    if (points == NULL) {
        printf("Error: Could not allocate memory for cluster array\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < k; ++i) {
        clusters[i] = (Cluster*)malloc(sizeof(Cluster));
        if (clusters[i] == NULL) {
            printf("Error: Could not allocate memory for cluster\n");
            exit(EXIT_FAILURE);
        }
        clusters[i]->centroid = points[i];
        clusters[i]->capacity = 1;
        clusters[i]->points = (DataPoint**)malloc(sizeof(DataPoint*));
        if (clusters[i]->points == NULL) {
            printf("Error: Could not allocate memory for data points\n");
            exit(EXIT_FAILURE);
        }
        clusters[i]->points[0] = points[i];
        clusters[i]->num_points = 1;
        clusters[i]->index = i;
        points[i]->cluster = clusters[i];
    }

    /* Running k-means clustering */
    i = 0;
    while ((!converged) && (i < iter)) {
        /* Assign points to closest cluster */
        for (j = 0; j < num_points; j++) {
            assign_to_closest(points[j], clusters, k);
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

    /* Print the final centroids */
    print_centroids(clusters, k);

    /* Cleanup */
    for (i = 0; i < num_points; i++) {
        free(points[i]->coords);
        free(points[i]);
    }

    for (i = 0; i < k; i++) {
        if (clusters[i]->centroid->index == -1) {
            free(clusters[i]->centroid->coords);
            free(clusters[i]->centroid);
        }
        free(clusters[i]->points);
        free(clusters[i]);
    }
    free(clusters);
    free(points);
    return 0;
}
