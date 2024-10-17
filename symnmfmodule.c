#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"


// Wrapper function for sym
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject *data_list;
    if (!PyArg_ParseTuple(args, "O", &data_list)) {
        return NULL;
    }

    int n = PyList_Size(data_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Data list is empty.");
        return NULL;
    }
    int d = PyList_Size(PyList_GetItem(data_list, 0));

    double *data = (double *)malloc(n * d * sizeof(double));
    if (!data) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for data.");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(data_list, i);
        for (int j = 0; j < d; j++) {
            PyObject *item = PyList_GetItem(row, j);
            data[i * d + j] = PyFloat_AsDouble(item);
        }
    }

    double *similarity_matrix = (double *)malloc(n * n * sizeof(double));
    if (!similarity_matrix) {
        free(data);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for similarity matrix.");
        return NULL;
    }

    sym(data, n, d, similarity_matrix);

    PyObject *result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(similarity_matrix[i * n + j]));
        }
        PyList_SetItem(result, i, row);
    }

    free(data);
    free(similarity_matrix);
    return result;
}


// Wrapper function for symnmf
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *W_list, *H_list;
    int max_iter;
    double tol;
    if (!PyArg_ParseTuple(args, "OOid", &W_list, &H_list, &max_iter, &tol)) {
        return NULL;
    }

    int n = PyList_Size(W_list);
    int k = PyList_Size(PyList_GetItem(H_list, 0));

    double *W = (double *)malloc(n * n * sizeof(double));
    double *H = (double *)malloc(n * k * sizeof(double));
    if (!W || !H) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for matrices.");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(W_list, i);
        for (int j = 0; j < n; j++) {
            W[i * n + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(H_list, i);
        for (int j = 0; j < k; j++) {
            H[i * k + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    symnmf(W, H, n, k, max_iter, tol);

    PyObject *result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_New(k);
        for (int j = 0; j < k; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(H[i * k + j]));
        }
        PyList_SetItem(result, i, row);
    }

    free(W);
    free(H);
    return result;
}

// Wrapper function for ddg
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject *similarity_list;
    if (!PyArg_ParseTuple(args, "O", &similarity_list)) {
        return NULL;
    }

    int n = PyList_Size(similarity_list);
    double *similarity_matrix = (double *)malloc(n * n * sizeof(double));
    double *degree_matrix = (double *)calloc(n * n, sizeof(double));
    if (!similarity_matrix || !degree_matrix) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for matrices.");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(similarity_list, i);
        for (int j = 0; j < n; j++) {
            similarity_matrix[i * n + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    ddg(similarity_matrix, n, degree_matrix);

    PyObject *result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(degree_matrix[i * n + j]));
        }
        PyList_SetItem(result, i, row);
    }

    free(similarity_matrix);
    free(degree_matrix);
    return result;
}

// Wrapper function for norm
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject *similarity_list, *degree_list;
    if (!PyArg_ParseTuple(args, "OO", &similarity_list, &degree_list)) {
        return NULL;
    }

    int n = PyList_Size(similarity_list);
    double *similarity_matrix = (double *)malloc(n * n * sizeof(double));
    double *degree_matrix = (double *)malloc(n * n * sizeof(double));
    double *norm_matrix = (double *)malloc(n * n * sizeof(double));
    if (!similarity_matrix || !degree_matrix || !norm_matrix) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for matrices.");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *similarity_row = PyList_GetItem(similarity_list, i);
        PyObject *degree_row = PyList_GetItem(degree_list, i);
        for (int j = 0; j < n; j++) {
            similarity_matrix[i * n + j] = PyFloat_AsDouble(PyList_GetItem(similarity_row, j));
            degree_matrix[i * n + j] = PyFloat_AsDouble(PyList_GetItem(degree_row, j));
        }
    }

    norm(similarity_matrix, degree_matrix, n, norm_matrix);

    PyObject *result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject *row = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(norm_matrix[i * n + j]));
        }
        PyList_SetItem(result, i, row);
    }

    free(similarity_matrix);
    free(degree_matrix);
    free(norm_matrix);
    return result;
}

// Method definitions
static PyMethodDef SymnmfMethods[] = {
    {"symnmf", py_symnmf, METH_VARARGS, "Compute the Symmetric Non-negative Matrix Factorization."},
    {"ddg", py_ddg, METH_VARARGS, "Compute the diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "Compute the normalized similarity matrix."},
    {"sym", py_sym, METH_VARARGS, "Compute the similarity matrix."},
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

// Module initialization function
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
