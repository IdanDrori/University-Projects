#include "symnmf.h"

void swap(double a, double b) {
    double tmp;
    tmp = a;
    a = b;
    b = tmp;
}

// VECTOR** read_file(int num_vectors, char* file_name) {
//     /* Reads a provided txt file, and returns an array of VECTORs */
//     int i = 0;
//     char *buffer = NULL;
//     size_t bufsize = 0;
//     int nread;
//     VECTOR** vectors;

//     /*  Allocate memory for vectors */
//     vectors = (VECTOR**)malloc((size_t)num_vectors * sizeof(VECTOR*));
//     if (points == NULL) {
//         printf("Error: Could not allocate memory for vectors\n");
//         exit(EXIT_FAILURE);
//     }

//     /* Read the vectors */
//     while ((nread = (int)getline(&buffer, &bufsize, stdin)) != -1) {
//         points[i] = get_VECTOR(buffer, i);
//         i++;
//     }
//     free(buffer);
//     return vectors;
// }

/* FROM CHATGPT */
int count_lines(const char* filename) {
    FILE *file = fopen(filename, "r");
    ALLOC_ASSERT(file);

    int count = 0;
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        // Skip empty lines or lines with only whitespace
        int is_empty = 1;
        for (int i = 0; line[i] != '\0'; i++) {
            if (!isspace(line[i])) {
                is_empty = 0;
                break;
            }
        }

        if (!is_empty) {
            count++;
        }
    }

    fclose(file);
    return count;
}

/* FROM CHATGPT*/
VECTOR* read_vectors(const char* filename, int num_vectors) {
    FILE *file = fopen(filename, "r");
    ALLOC_ASSERT(file);
    VECTOR** vectors = (VECTOR**)malloc(num_vectors * sizeof(VECTOR*));
    ALLOC_ASSERT(vectors);

    char line[1024];
    int vector_idx = 0;
    
    while (fgets(line, sizeof(line), file) && vector_idx < num_vectors) {
        if (strcmp(line, "\n") == 0) {  // Check for empty line
            break;
        }

        // Use get_VECTOR to extract the VECTOR from the line
        vectors[vector_idx] = get_VECTOR(line, vector_idx);
        vector_idx++;
    }

    fclose(file);
    return vectors;
}

VECTOR* get_VECTOR(const char* line, int index) {
    /* Reads a string formatted as "float1,float2,...", and returns a VECTOR */
    int idx = 0;
    int len = get_len(line);
    double value = 0.0;
    VECTOR* vector; 
    vector = (VECTOR*)malloc(sizeof(VECTOR));
    ALLOC_ASSERT(vector);
    vector->coords = (double*)calloc((size_t)len, sizeof(double));
    ALLOC_ASSERT(vector->coords);
    while (*line && *line != '\n') { /* Parse the string until the newline character or end of string */
        if (sscanf(line, "%lf,", &value) == 1) { /* Use sscanf to extract a double */
            vector->coords[idx++] = value;
            while (*line && *line != ',') { /* Move the line pointer to the next character after the comma */
                line++;
            }
            if (*line == ',') { /* Skip the comma */
                line++;
            }
        } 
        else {
            break; /* If sscanf fails, break the loop */
        }
        if (idx > len) {
            break; 
        }
    }
    vector->len = len;
    return vector;
}

int get_len(const char* line) {
    /* This function simply gets the length of a vector */
    int len = 1;
    const char *p;
    for (p = line; *p != '\0'; p++) {
        if (*p == ',') {
            len++;
        }
    }
    return len;
}

/* MINIMALLY TESTED */
double sq_distance(VECTOR* vect_a, VECTOR* vect_b) { 
    int i;
    int sum = 0;
    int len = vect_a->len;
    for (i = 0; i < len; i++) {
        sum += pow((vect_a->coords[i] - vect_b->coords[i]), 2);
    }
    return sum;
} 

/* MINIMALLY TESTED */
double get_exp(double x) {
    return exp(-0.5 * x);
}

/* MINIMALLY TESTED */
double sum_vector(VECTOR* vect) {
    double sum = 0.0;
    int i;
    for (i = 0; i < vect->len; i++) {
        sum += vect->coords[i];
    }
    return sum;
}

void set_index(MATRIX* mat, int j, int i, double val) {
    mat->columns[i]->coords[j] = val;
    return;
}

double get_index(MATRIX* mat, int j, int i) {
    return mat->columns[i]->coords[j];
}

VECTOR* create_vect(double* arr, int len) {
    VECTOR* vect = (VECTOR*)malloc(sizeof(VECTOR));
    ALLOC_ASSERT(vect)
    // vect->len = sizeof(arr) / sizeof(arr[0]);
    vect->len = len;
    vect->coords = arr;
}

VECTOR** create_vect_arr(double** mat, int num_cols, int num_rows) {
    int i;
    VECTOR** vectors = (VECTOR**)malloc(sizeof(VECTOR*) * num_cols);
    ALLOC_ASSERT(vectors);
    for (i = 0; i < num_cols; i++) {
        vectors[i] = create_vect(mat[i], num_rows);
    }
    return vectors;
}

/* MINIMALLY TESTED */
double vect_mul(VECTOR* vect_a, VECTOR* vect_b) {
    double result = 0.0;
    int i;
    for (i = 0; i < vect_a->len; i++) {
        result += (vect_a->coords[i]) * (vect_b->coords[i]);
    }
    return result;
}

/* MINIMALLY TESTED */
MATRIX* transpose(MATRIX* mat) {
    int i, j;
    MATRIX* mat_transpose = create_zero_mat(mat->num_cols, mat->num_rows);
    // MATRIX* mat_transpose = (MATRIX*)malloc(sizeof(MATRIX));
    // ALLOC_ASSERT(mat_transpose)
    // VECTOR** vectors = (VECTOR**)malloc(sizeof(VECTOR*) * mat->num_cols);
    // mat_transpose->columns = vectors;
    // mat_transpose->num_cols = mat->num_rows;
    // mat_transpose->num_rows = mat->num_cols;
    for (i = 0; i < mat_transpose->num_cols; i++) {
        // VECTOR* vector = (VECTOR*)malloc(sizeof(VECTOR));
        // ALLOC_ASSERT(vector)
        // vector->len = mat_transpose->num_cols;
        // vector->coords = (double*)malloc(sizeof(double) * vector->len);
        // ALLOC_ASSERT(vector->coords);
        // mat_transpose->columns[i] = vector;
        for (j = 0; j < mat_transpose->num_rows; j++) {
            // mat_transpose->columns[i]->coords[j] = mat->columns[j]->coords[i];
            set_index(mat_transpose, j, i, get_index(mat, i, j));
        }
    }

    return mat_transpose;
}

/* MINIMALLY TESTED */
MATRIX* mat_mul(MATRIX* mat_a, MATRIX* mat_b) {
    int i, j;
    // MATRIX* res = (MATRIX*)malloc(sizeof(MATRIX));
    // res->num_cols = mat_a->num_rows;
    // res->num_rows = mat_b->num_cols;
    // ALLOC_ASSERT(res);
    // VECTOR** vectors = (VECTOR**)malloc(sizeof(VECTOR*) * res->num_cols);
    // ALLOC_ASSERT(vectors);
    // res->columns = vectors;
    MATRIX* res = create_zero_mat(mat_b->num_cols, mat_a->num_rows);
    MATRIX* mat_a_T = transpose(mat_a);
    for (i = 0; i < res->num_cols; i++) {
        // VECTOR* vector = (VECTOR*)malloc(sizeof(VECTOR));
        // ALLOC_ASSERT(vector);
        // res->columns[i] = vector;
        // vector->len = res->num_cols;
        // vector->coords = (double*)malloc(sizeof(double) * vector->len);
        // ALLOC_ASSERT(vector->coords);
        for (j = 0; j < res->num_rows; j++) {
            //res->columns[i]->coords[j] = vect_mul(mat_a_T->columns[j], mat_b->columns[i]);
            set_index(res, j, i, vect_mul(mat_a_T->columns[j], mat_b->columns[i]));
        }
        
    }
    free_mat(mat_a_T);
    return res;
}

/* SORT OF TESTED */
void free_mat(MATRIX* mat) {
    int i, j;
    for (i = 0; i < mat->num_cols; i++) {
        free(mat->columns[i]->coords);
        free(mat->columns[i]);
    }
    free(mat->columns);
    free(mat);
}

/* Courtesy of ChatGPT */
void print_matrix(MATRIX* matrix) {
    if (!matrix) return;  // Handle null matrix pointer

    for (int row = 0; row < matrix->num_rows; ++row) {
        for (int col = 0; col < matrix->num_cols; ++col) {
            // Access the (row, col) element using the column vector's coordinates
            printf("%10.4f ", matrix->columns[col]->coords[row]);
        }
        printf("\n");  // Move to the next row
    }
}  

/* MINIMALLY TESTED */
/* Function raises diagonal matrix to the power of exp */
void pow_diag(MATRIX* mat, double exp) {
    int i;
    for (i = 0; i < mat->num_cols; i++) {
        //mat->columns[i]->coords[i] = pow(mat->columns[i]->coords[i], exp); 
        set_index(mat, i, i, pow(get_index(mat, i, i), exp));
    }
}

MATRIX* create_zero_mat(int rows, int cols) {
    int i;
    MATRIX* mat = (MATRIX*)malloc(sizeof(MATRIX));
    ALLOC_ASSERT(mat);
    mat->num_cols = cols;
    mat->num_rows = rows;
    VECTOR** vectors = (VECTOR**)malloc(sizeof(VECTOR*) * cols);
    ALLOC_ASSERT(vectors);
    mat->columns = vectors;
    for (i = 0; i < mat->num_cols; i++) {
        VECTOR* vector = (VECTOR*)malloc(sizeof(VECTOR));
        ALLOC_ASSERT(vector);
        vector->len = rows;
        vector->coords = (double*)calloc(sizeof(double), rows);
        ALLOC_ASSERT(vector->coords);
        mat->columns[i] = vector;
    }
    return mat;
}

MATRIX* create_mat(VECTOR** vectors, int rows, int cols) {
    int i;
    MATRIX* mat = (MATRIX*)malloc(sizeof(MATRIX));
    ALLOC_ASSERT(mat);
    mat->num_cols = cols;
    mat->num_rows = rows;
    mat->columns = vectors;
    return mat;
}

/* MINIMALLY TESTED */
MATRIX* mat_diff(MATRIX* mat_a, MATRIX* mat_b) {
    /* Function assumes mat_a and mat_a are of the same dimensions */
    int i, j;
    double diff;
    MATRIX* res = create_zero_mat(mat_a->num_rows, mat_a->num_cols);
    for (i = 0; i < res->num_cols; i++) {
        for (j = 0; j < res->num_rows; j++) {
            //res->columns[i]->coords[j] = (mat_a->columns[i]->coords[j]) - (mat_b->columns[i]->coords[j]);
            diff = get_index(mat_a, j, i) - get_index(mat_b, j, i);
            set_index(res, j, i, diff);
        }
    }
    return res;
}

/* MINIMALLY TESTED */
double frob_norm(MATRIX* mat) {
    /* https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm */
    int i, j;
    double res;
    for (i = 0; i < mat->num_cols; i++) {
        for (j = 0; j < mat->num_rows; j++) {
            res += pow(fabs(get_index(mat, j, i)), 2);
        }
    }
    return sqrt(res);
}

/* TESTED BEFORE SIGNATURE CHANGE */
MATRIX* sym(VECTOR** vectors, int num_vectors) {
    int i, j;
    ALLOC_ASSERT(vectors); /* Checking whether the vectors array is not empty */
    double res;
    MATRIX* sim_mat = create_zero_mat(num_vectors, num_vectors);
    for (i = 0; i < sim_mat->num_cols; i++) {
        for (j = 0; j < sim_mat->num_cols; j++) {
            if (i != j) {
                res = get_exp(sq_distance(vectors[i], vectors[j]));
                set_index(sim_mat, j, i, res);
            } 
        }
    }
    return sim_mat;
}


MATRIX* ddg(VECTOR** vectors, int num_vectors) {
    /* TESTED BEFORE CHANGE */
    int i;
    MATRIX* sim_mat = sym(vectors, num_vectors); /* NEW LINE */
    MATRIX* sim_mat_transpose = transpose(sim_mat);
    MATRIX* res = create_zero_mat(sim_mat->num_rows, sim_mat->num_cols);
    for (i = 0; i < res->num_cols; i++) {
        set_index(res, i, i, sum_vector(sim_mat_transpose->columns[i]));
    }
    return res;
}

MATRIX* norm(VECTOR** vectors, int num_vectors) {
    /* The graph Laplacian W is defined as W = D^(-1/2)AD^(-1/2) */
    MATRIX* mat_A = sym(vectors, num_vectors);
    MATRIX* diag_mat = ddg(vectors, num_vectors);
    MATRIX* res;
    pow_diag(diag_mat, -0.5);
    res = mat_mul(mat_mul(diag_mat, mat_A), diag_mat);
    return res;
}

MATRIX* update_mat(MATRIX* mat_h, MATRIX* mat_w) {
    int i,j;
    MATRIX* result;
    MATRIX* denominator;
    MATRIX* nominator;
    result = create_zero_mat(mat_h->num_rows, mat_h->num_cols);
    nominator = mat_mul(mat_w, mat_h);
    denominator = mat_mul(mat_mul(mat_h, transpose(mat_h)), mat_h);
    for (i = 0; i < result->num_cols; i++) {
        for (j = 0; j < result->num_rows; j++) {
            get_index(mat_h, j, i) * (1 - BETA + BETA*(get_index(nominator, j, i)/get_index(denominator, j ,i)));
        }
    }
    free_mat(nominator);
    free_mat(denominator);
    return result;
}

MATRIX* symnmf(MATRIX* mat_h, MATRIX* mat_W) {
    int t;
    // MATRIX* mat_W;
    // MATRIX* mat_A;
    // MATRIX* diag;
    MATRIX* prev_iteration = mat_h;
    MATRIX* cur_iteration;
    // mat_A = create_sim(X, num_vectors);
    // diag = diagonal_degree_mat(mat_A);
    // mat_W = norm_sim_matrix(diag, mat_A);
    for (t = 0; t < MAX_ITER; t++) {
        cur_iteration = update_mat(prev_iteration, mat_W);
        free_mat(prev_iteration);
        prev_iteration = cur_iteration;
        if (pow(frob_norm(mat_diff(cur_iteration, prev_iteration)), 2) < EPSILON) {
            break;
        }
    }
    return cur_iteration;
}

/* NOT TESTED YET */
int main(int argc, char const *argv[])
{
    char* goal;
    char* filename;
    MATRIX* result;
    if (argc != 3){ /* Incorrect number of command line arguments */
        printf("An Error Has Occurred");
        exit(EXIT_FAILURE);
    }
    goal = argv[1];
    filename = argv[2];
    int num_vectors = count_lines(filename);
    VECTOR** vectors = read_file(filename, num_vectors);
    if (goal == "sym") {
        result = sym(vectors, num_vectors);
    }
    else if (goal == "ddg") {
        result = ddg(vectors, num_vectors);
    }
    else if (goal == "norm") {
        result = norm(vectors, num_vectors);
    }
    print_matrix(result);
    return 1;
}
