#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _POSIX_C_SOURCE 200809L


#define BETA (0.5)
#define EPSILON (0.0001)
#define MAX_ITER (300)

#define DEBUG /* Comment this line for final submission! */
#ifdef DEBUG
    #define ALLOC_ASSERT(x) { \
        if (x == NULL) { \
            printf("Error has occurred in line:%d",__LINE__); \
            exit(EXIT_FAILURE); \
            }}
#else
    #define ALLOC_ASSERT(x) { \
        if (x == NULL) { \
            printf("An Error Has Occurred"); \
            exit(EXIT_FAILURE); \
            }}

#endif

typedef struct vector {
    int len;
    double* coords; 
} VECTOR;

typedef struct matrix {
    /* Struct contains array of column vectors, also number of rows and number of columns */
    int num_rows;
    int num_cols;
    VECTOR** columns;
} MATRIX;

/*General purpose functions */
void swap(double, double); /* Swaps two doubles */
VECTOR** read_file(char*, int); /* Reads file from stdinput and return array of VECTORs */
VECTOR* get_VECTOR(const char*, int); /* Returns VECTOR from line formatted as "float1,float2,..." */
int get_len(const char*); /* Returns length of the vector given a line */
int count_lines(const char*); /* Returns the number of lines in the file - 1 */

/* General math functions */
double sq_distance(VECTOR*, VECTOR*); /* Calculate square of euclidean distance */
double get_exp(double); /* Calculate e^(-0.5*x) */
double sum_vector(VECTOR*); /* Calculates sum of vector (This is what they call the degree of a vertex x_i) */
double vect_mul(VECTOR*, VECTOR*); /* Vector multiplication */

/* Matrix functions */
MATRIX* transpose(MATRIX*); /* Transposes a matrix */
MATRIX* mat_mul(MATRIX*, MATRIX*); /* Multiply two matrices */
void pow_diag(MATRIX*, double); /* Gets diagonal matrix and raises it to the power of the given variable. in-place */
MATRIX* create_mat(VECTOR**, int, int); /* Create matrix from array of vectors, also gets dimensions */
MATRIX* create_zero_mat(int,int); /* Create empty matrix */
MATRIX* mat_diff(MATRIX*, MATRIX*); /* Calculates difference between matrices. Assumes matrices are of the same dimensions */
double frob_norm(MATRIX*); /* Calculates Frobenius norm of matrix*/
void free_mat(MATRIX*); /* Frees memory of a matrix*/
void print_matrix(MATRIX*);
void set_index(MATRIX*, int, int, double); /* Sets the matrix at index i,j to be the double */
double get_index(MATRIX*, int, int); /* Gets the element at index i,j */
VECTOR* create_vect(double*, int); /* Creates vector struct from array of doubles */
VECTOR** create_vect_arr(double**, int, int); /* Create VECTOR array from array of double arrays */

/* Symnmf functions */
MATRIX* sym(VECTOR**, int); /* Create similarity matrix from vector array */
MATRIX* ddg(VECTOR**, int); /* Create diagonal degree matrix from vector array */
MATRIX* norm(VECTOR**, int); /* Create graph Laplacian matrix from vector array */ 
MATRIX* update_mat(MATRIX*, MATRIX*); /* Update matrix in accordance to symnmf algorithm */
MATRIX* symnmf(MATRIX*, MATRIX*); /* Finds the optimal matrix that solves min||W-HH^T||^2 as described in the SymNMF algorithm */
