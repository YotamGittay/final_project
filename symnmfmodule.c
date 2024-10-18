#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

//Todo: make sure the error messages are in the correct format
//Todo: decide whether we want to give better names for n and d

// function to parse the data matrix from the python nested list,
// also inits n and d vars
double* parse_python_nested_list_to_c_flat_array_and_calc_dims(PyObject *data_list, int *n, int *d) {
    *n = PyList_Size(data_list);
    if (*n == 0) {
        printf("Data list is empty.");
        return NULL;
    }
    *d = PyList_Size(PyList_GetItem(data_list, 0));

    double *data = (double *)malloc((*n) * (*d) * sizeof(double));
    if (!data) {
        printf("Unable to allocate memory for data.");
        return NULL;
    }

    for (int i = 0; i < *n; i++) {
        PyObject *row = PyList_GetItem(data_list, i);
        for (int j = 0; j < *d; j++) {
            PyObject *item = PyList_GetItem(row, j);
            data[i * *d + j] = PyFloat_AsDouble(item);
        }
    }

    return data;
}

// function to turn the c 1d array back to a python nested list
PyObject* parse_c_array_to_python_nested_list(double *array, int n, int d) {
    PyObject *result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(array[i * d + j]));
        }
        PyList_SetItem(result, i, row);
    }
    return result;
}

// Wrapper function for sym
static PyObject* py_sym(PyObject* self, PyObject* args) {
    int n, d;
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }
    double *data = parse_python_nested_list_to_c_flat_array_and_calc_dims(data_list, &n, &d);

    double *similarity_matrix = (double *)malloc(n * n * sizeof(double));
    if (!similarity_matrix) {
        free(data);
        printf("Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    sym(data, n, d, similarity_matrix);

    PyObject *result = parse_c_array_to_python_nested_list(similarity_matrix, n, n);

    free(data);
    free(similarity_matrix);
    return result;
}

// Wrapper function for ddg
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    int n, d;
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }
    double *data = parse_python_nested_list_to_c_flat_array_and_calc_dims(data_list, &n, &d);

    double *degree_matrix = (double *)malloc(n * n * sizeof(double));
    if (!degree_matrix) {
        free(data);
        printf("Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    ddg(data, n, d, degree_matrix);

    PyObject *result = parse_c_array_to_python_nested_list(degree_matrix, n, n);

    free(degree_matrix);
    return result;
}

// Wrapper function for norm
static PyObject* py_norm(PyObject* self, PyObject* args) {
    int n, d;
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }
    double *data = parse_python_nested_list_to_c_flat_array_and_calc_dims(data_list, &n, &d);

    double *norm_matrix = (double *)malloc(n * n * sizeof(double));
    if (!norm_matrix) {
        free(data);
        printf("Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    norm(data, n, d, norm_matrix);

    PyObject *result = parse_c_array_to_python_nested_list(norm_matrix, n, n);

    free(data);
    free(norm_matrix);
    return result;
}

// // Wrapper function for symnmf
// static PyObject* py_symnmf(PyObject* self, PyObject* args) {
//     PyObject *W_list, *H_list;
//     int max_iter;
//     double tol;
//     if (!PyArg_ParseTuple(args, "OOid", &W_list, &H_list, &max_iter, &tol)) {
//         return NULL;
//     }

//     int n = PyList_Size(W_list);
//     int k = PyList_Size(PyList_GetItem(H_list, 0));

//     double *W = (double *)malloc(n * n * sizeof(double));
//     double *H = (double *)malloc(n * k * sizeof(double));
//     if (!W || !H) {
//         PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for matrices.");
//         return NULL;
//     }

//     for (int i = 0; i < n; i++) {
//         PyObject *row = PyList_GetItem(W_list, i);
//         for (int j = 0; j < n; j++) {
//             W[i * n + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
//         }
//     }

//     for (int i = 0; i < n; i++) {
//         PyObject *row = PyList_GetItem(H_list, i);
//         for (int j = 0; j < k; j++) {
//             H[i * k + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
//         }
//     }

//     symnmf(W, H, n, k, max_iter, tol);

//     PyObject *result = PyList_New(n);
//     for (int i = 0; i < n; i++) {
//         PyObject *row = PyList_New(k);
//         for (int j = 0; j < k; j++) {
//             PyList_SetItem(row, j, PyFloat_FromDouble(H[i * k + j]));
//         }
//         PyList_SetItem(result, i, row);
//     }

//     free(W);
//     free(H);
//     return result;
// }

// Method definitions
static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute the similarity matrix."},
    {"ddg", py_ddg, METH_VARARGS, "Compute the diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "Compute the normalized similarity matrix."},
    // {"symnmf", py_symnmf, METH_VARARGS, "Compute the Symmetric Non-negative Matrix Factorization."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymnmfMethods
};

//TODO: it looks a bit different in the tutorial notebook, might want to change it accordingly
// Module initialization function
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
