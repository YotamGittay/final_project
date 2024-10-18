#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Todo: make sure the error messages are in the correct format */

double* transform_python_nested_list_to_c_flat_array(PyObject *data_list, int rows, int cols) {
    double *data = (double *)calloc((rows) * (cols), sizeof(double));
    if (!data) {
        printf("Unable to allocate memory for data.");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        PyObject *row = PyList_GetItem(data_list, i);
        for (int j = 0; j < cols; j++) {
            PyObject *item = PyList_GetItem(row, j);
            data[i * cols + j] = PyFloat_AsDouble(item);
        }
    }

    return data;
}

/* Function that gets a python nested list and derives rows and cols dimensions */
void calc_dimensions_by_data_list_and_assign_to_vars_by_address(PyObject *nested_list, int *rows, int *cols) {
    *rows = PyList_Size(nested_list);
    if (*rows == 0) {
        printf("Data list is empty.");
        return;
    }
    *cols = PyList_Size(PyList_GetItem(nested_list, 0));
}

PyObject* transform_c_array_to_python_nested_list(double *array, int rows, int cols) {
    PyObject *result = PyList_New(rows);
    for (int i = 0; i < rows; i++) {
        PyObject *row = PyList_New(cols);
        for (int j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(array[i * cols + j]));
        }
        PyList_SetItem(result, i, row);
    }
    return result;
}

/* Wrapper function for sym */ 
static PyObject* py_sym(PyObject* self, PyObject* args) {
    int n = 0 , d = 0;
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }
    calc_dimensions_by_data_list_and_assign_to_vars_by_address(data_list, &n, &d);
    double *points = transform_python_nested_list_to_c_flat_array(data_list, n, d);

    double *similarity_matrix = (double *)calloc(n * n, sizeof(double));
    if (!similarity_matrix) {
        free(points);
        printf("Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    sym(points, n, d, similarity_matrix);

    PyObject *result = transform_c_array_to_python_nested_list(similarity_matrix, n, n);

    free(points);
    free(similarity_matrix);
    return result;
}

/* Wrapper function for ddg */
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    int n, d;
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }

    calc_dimensions_by_data_list_and_assign_to_vars_by_address(data_list, &n, &d);
    double *points = transform_python_nested_list_to_c_flat_array(data_list, n, d);

    double *degree_matrix = (double *)calloc(n * n, sizeof(double));
    if (!degree_matrix) {
        free(points);
        printf("Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    ddg(points, n, d, degree_matrix);

    PyObject *result = transform_c_array_to_python_nested_list(degree_matrix, n, n);

    free(points);
    free(degree_matrix);
    return result;
}

/* Wrapper function for norm */
static PyObject* py_norm(PyObject* self, PyObject* args) {
    int n = 0, d = 0;
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }
    calc_dimensions_by_data_list_and_assign_to_vars_by_address(data_list, &n, &d);
    double *points = transform_python_nested_list_to_c_flat_array(data_list, n, d);
    
    double *norm_matrix = (double *)calloc(n * n, sizeof(double));
    if (!norm_matrix) {
        free(points);
        printf("Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    norm(points, n, d, norm_matrix);

    PyObject *result = transform_c_array_to_python_nested_list(norm_matrix, n, n);

    free(points);
    free(norm_matrix);
    return result;
}

/* Wrapper function for symnmf */
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *W_list, *H_list;
    int n, k;
    if (!PyArg_ParseTuple(args, "OO", &W_list, &H_list)) {
        return NULL;
    }
    calc_dimensions_by_data_list_and_assign_to_vars_by_address(W_list, &n, &n);
    calc_dimensions_by_data_list_and_assign_to_vars_by_address(H_list, &n, &k);
    double *W = transform_python_nested_list_to_c_flat_array(W_list, n, n);
    double *H = transform_python_nested_list_to_c_flat_array(H_list, n, k);

    symnmf(W, H, n, k);

    PyObject *result = transform_c_array_to_python_nested_list(H, n, k);

    free(W);
    free(H);
    return result;
}

/* Method definitions */
static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute the similarity matrix."},
    {"ddg", py_ddg, METH_VARARGS, "Compute the diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "Compute the normalized similarity matrix."},
    {"symnmf", py_symnmf, METH_VARARGS, "Compute the Symmetric Non-negative Matrix Factorization."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymnmfMethods
};

/* TODO: it looks a bit different in the tutorial notebook, might want to change it accordingly */
/* Module initialization function */ 
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
