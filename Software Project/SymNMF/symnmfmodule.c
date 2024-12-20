#include <Python.h>
#define PY_SSIZE_T_CLEAN
#include "symnmf.h"

/* Function Table */
static PyMethodDef symnmfMethods[] = { 
    {
    "symnmf_C",
    (PyCFunction) symnmf_wrapper,
    METH_VARARGS,
    PyDoc_STR(
        "Returns the matrix H (as list of lists of floats) that solves min||W-HH^T||^2, " 
        "calculated as described in the SymNMF algorithm\n"
        "Args: \n"
        "Matrix H - represented as an array of double arrays\n"
        "Matrix W, the graph laplacian - represented as an array of double arrays\n"
        ) 
    }, 

    {
        "sym_C",
        (PyCFunction) sym_wrapper,
        METH_VARARGS,
        PyDoc_STR(
            "Returns the similarity matrix given array of vectors X\n"
            "Args:\n"
            "Vector array X - represented as an array of double arrays\n"
            "Number of vectors - integer\n"
            )
    },

    {
        "ddg_C",
        (PyCFunction) ddg_wrapper,
        METH_VARARGS,
        PyDoc_STR(
            "Returns the diagonal degree matrix given array of vectors X\n"
            "Args:\n"
            "Vector array X - represented as an array of double arrays\n"
            "Number of vectors - integer\n"
        )
    },

    {
        "norm_C",
        (PyCFunction) norm_wrapper,
        METH_VARARGS,
        PyDoc_STR(
            "Returns the graph laplacian given array of vectors X\n"
            "Args:\n"
            "Vector array X - represented as an array of double arrays\n"
            "Number of vectors - integer\n"
        )
    },

    {
    NULL, NULL, 0, NULL
    }
};

/* Module Definition */
static struct PyModuleDef Symnmf_Module = { /* TODO FINISH THIS */
    PyModuleDef_HEAD_INIT,
    "symnmf_module",
    "SymNMF Python wrapper",
    -1,
    symnfMethods
};

VECTOR** double_to_vectors(double**, int); /* Create array of VECTORs from array of arrays of doubles */
double** vector_to_doubles(VECTOR**, int); /* Create array of arrays of doubles from array of VECTORs */
int* get_mat_dimensions(PyObject); /* Get Matrix dimensions from PyObject, return an array of size 2, rows first then cols */
PyObject* mat_to_PyObj(MATRIX*);
void free_vector_arr(VECTOR**, int); /* Frees VECTOR array */
pyMODINIT_FUNC PyInit_symnf(void); 

int* get_mat_dimensions(PyObject mat) {
    int dims[2];
    int row_num, col_num;
    PyObject mat_col;
    col_num = PyObject_Length(mat);
    mat_col = PyList_GetItem(mat, 0);
    row_num = PyObject_Length(mat_col);
    dims[0] = row_num;
    dims[1] = col_num;
    return dims;
}

double** double_arr_from_PyObject(PyObject* mat) {
    int i, j;
    PyObject* mat_col;
    PyObject* mat_coordinate;
    double* vector;
    int dim, num_cols;
    double** result = (double**)malloc(sizeof(double*) * num_cols);
    ALLOC_ASSERT(result);
    int num_cols = PyObject_Length(mat);
    for (i = 0; i < num_cols; ++i) {
        mat_col = PyList_GetItem(mat, i);
        vector = (double*)malloc(num_rows * sizeof(double));
        ALLOC_ASSERT(vector);
        dim = PyObject_Length(mat_col);
        for (j = 0; j < dim; ++j) {
            mat_coordinate = PyList_GetItem(mat_col, j);
            vector[j] = PyFloat_AsDouble(mat_coordinate);
        }
        centroids[i] = vector;
    }
    return result;
}

void free_vector_arr(VECTOR** vectors, int vect_num) {
    int i;
    for (i = 0; i < vect_num; i++) {
        free(vectors[i]->coords);
        free(vectors[i]);
    }
    free(vectors);
}

static PyObject* symnmf_wrapper(PyObject *self, PyObject *args) {
    int i;
    PyObject *Py_mat_h; /* Randomized matrix H, gotten from Python as array of double arrays */
    PyObject *Py_mat_w; /* Matrix W, gotten from Python as array of double arrays; */
    double** mat_h;
    int mat_h_dims[2]; /* rows, cols */
    double** mat_w;
    int mat_w_dims[2]; /* rows, cols*/
    MATRIX* MATRIX_H;
    VECTOR** VECTORS_H;
    VECTOR** VECTORS_W;
    MATRIX* MATRIX_W;
    MATRIX* result_mat;
    if (!PyArg_ParseTuple(args, "OO", &Py_mat_h, &Py_mat_w)) {
        return NULL;
    }
    mat_h = double_arr_from_PyObject(Py_mat_h);
    mat_h_dims = get_mat_dimensions(Py_mat_h);
    mat_w = double_arr_from_PyObject(Py_X);
    mat_w_dims = get_mat_dimensions(Py_X);
    VECTORS_H = create_vect_arr(mat_h, mat_h_dims[0], mat_h_dims[1]);  /* dims[0] = num of rows, dims[1] = num of cols */
    MATRIX_H = create_mat(VECTORS_H, mat_h_dims[0], mat_h_dims[1]);
    VECTORS_W = create_vect_arr(X, X_dims[0], X_dims[1]);
    MATRIX_W = create_mat(VECTORS_W, mat_w_dims[0], mat_w_dims[1]);
    result_mat = symnmf(MATRIX_H, MATRIX_W);
    result_list = mat_to_PyObj(result_mat);
    free_mat(MATRIX_H);
    free(mat_h);
    free(VECTORS_H);
    free_mat(MATRIX_W);
    free(mat_w);
    free(VECTORS_W);
    free_mat(result_mat);
    return result_list;
}

static PyObject *sym_wrapper(PyObject *self, PyObject *args) {
    int i;
    PyObject *Py_X; /* Vector array X, gotten from Python as array of double arrays; */
    double** X;
    int X_dims[2]; /* rows, cols*/
    VECTOR** VECTORS_X;
    MATRIX* result_mat;
    if (!PyArg_ParseTuple(args, "O", &Py_X)) {
        return NULL;
    }
    X = double_arr_from_PyObject(Py_X);
    X_dims = get_mat_dimensions(Py_X);
    VECTORS_X = create_vect_arr(X, X_dims[0], X_dims[1]);
    result_mat = sym(VECTORS_X, X_dims[1]);
    result_list = mat_to_PyObj(result_mat);
    free_vector_arr(VECTORS_X, X_dims[1]);
    free_mat(result_mat);
    return result_list;
}

static PyObject *ddg_wrapper(PyObject *self, PyObject *args) {
    int i;
    PyObject *Py_X; /* Vector array X, gotten from Python as array of double arrays; */
    double** X;
    int X_dims[2]; /* rows, cols*/
    VECTOR** VECTORS_X;
    MATRIX* result_mat;
    if (!PyArg_ParseTuple(args, "O", &Py_X)) {
        return NULL;
    }
    X = double_arr_from_PyObject(Py_X);
    X_dims = get_mat_dimensions(Py_X);
    VECTORS_X = create_vect_arr(X, X_dims[0], X_dims[1]);
    result_mat = ddg(VECTORS_X, X_dims[1]);
    result_list = mat_to_PyObj(result_mat);
    free_vector_arr(VECTORS_X, X_dims[1]);
    free_mat(result_mat);
    return result_list;
}

static PyObject *norm_wrapper(PyObject *self, PyObject *args) {
    int i;
    PyObject *Py_X; /* Vector array X, gotten from Python as array of double arrays; */
    double** X;
    int X_dims[2]; /* rows, cols*/
    VECTOR** VECTORS_X;
    MATRIX* result_mat;
    if (!PyArg_ParseTuple(args, "O", &Py_X)) {
        return NULL;
    }
    X = double_arr_from_PyObject(Py_X);
    X_dims = get_mat_dimensions(Py_X);
    VECTORS_X = create_vect_arr(X, X_dims[0], X_dims[1]);
    result_mat = norm(VECTORS_X, X_dims[1]);
    result_list = mat_to_PyObj(result_mat);
    free_vector_arr(VECTORS_X, X_dims[1]);
    free_mat(result_mat);
    return result_list;
}

PyMODINIT_FUNC PyInit_kmeans(void) {
    return PyModule_Create(&Symnmf_Module);
}

/* PAY ATTENTION!: The double array is not freed and the vector coordinates are the elements in the double array. */
VECTOR** double_to_vectors(double** double_array, int len) {
    int i;
    VECTOR** vector_arr = (VECTOR**)malloc(sizeof(VECTOR*) * len);
    ALLOC_ASSERT(vector_arr);
    for (i = 0; i < len; i++) {
        VECTOR* vector = (VECTOR*)malloc(sizeof(VECTOR));
        ALLOC_ASSERT(vector);
        vector->coords = double_array[i];
    }
    return vector_arr;
}

PyObject mat_to_PyObj(MATRIX* mat) {
    int i, j;
    PyObject* result_list, vector, num;
    PyObject* result_list = PyList_New(mat->num_cols);
    if (!result_list) {
        return NULL; // Memory allocation failed
    }
    // Populate the Python list with lists of doubles
    for (i = 0; i < mat->num_cols; ++i) {
        PyObject* vector = PyList_New(mat->num_rows);

        for (j = 0; j < mat->num_rows; ++j) {
            PyObject* num = Py_BuildValue("f", mat->columns[i]->coords[j]);
            PyList_SET_ITEM(vector, j, num); 
        }

        PyList_SET_ITEM(result_list, i, vector); 
    }
    return result_list;
}
