#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf_helpers.h"

double const EPSILON = 1e-4;

void print_matrix(double *mat, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            printf("%.4f", mat[row * d + col]);
            if (col < d - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

void transpose_matrix(double *mat, double *target, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            target[col * n + row] = mat[row * d + col];
        }
    }
}

/* calc matrix^-0.5 */
void calc_degree_inv_sqrt_matrix(double *degree_matrix, double *degree_inv_sqrt, int n) {
    int i;
    for (i = 0; i < n * n; i++) {
        double degree_value = degree_matrix[i];
        if (degree_value != 0.0) {
            degree_inv_sqrt[i] = 1.0 / sqrt(degree_value);
        } else {
            degree_inv_sqrt[i] = 0.0;
        }
    }
}

void copy_matrix(double *mat, double *target, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            target[row * d + col] = mat[row * d + col];
        }
    }
}

void multiply_matrices(double *A, double *B, double *result, int n, int m, int p) {
    int i, j, k;
    
    /* Initialize the result matrix to zero */ 
    for (i = 0; i < n * p; i++) {
        result[i] = 0.0;
    }
    
    /* Perform matrix multiplication */
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            for (k = 0; k < m; k++) {
                result[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}

int check_matrix_convergence(double *H, double *new_H, int n, int k) {
    int i, j;
    double diff = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff += (new_H[i * k + j] - H[i * k + j]) * (new_H[i * k + j] - H[i * k + j]);
        }
    }

    return diff < EPSILON ? 1 : 0;

}

/* Function to get the dimensions of the data (n and d) from the file */
void get_dimensions_from_file(char *file_name, int *n, int *d) {
    FILE *file;
    char line[1024], *str;
    *n = 0; *d = 0;
    
    file = fopen(file_name, "r");
    if (!file) {
        printf("An Error Has Occurred");
        exit(1);
    }
    
    /* Count the number of rows and columns */
    while (fgets(line, sizeof(line), file)) {
        if (*n == 0) { /* If it's the first line, count the number of columns */
            str = line;
            while (*str) {
                if (*str == ',') {
                    (*d)++;
                }
                str++;
            }
            (*d)++;  /* Number of columns is one more than the number of commas */
        }
        (*n)++;
    }
    
    fclose(file);
}

/* Function to read data points from file */ 
double* read_points_from_file(char *file_name, int n, int d) {
    FILE *file;
    char line[1024], *token;
    int row = 0, col = 0;
    double *data;
    
    /* Allocate memory for the data array based on n and d */
    data = (double*)calloc(n * d, sizeof(double));
    if (data == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    
    file = fopen(file_name, "r");
    if (!file) {
        printf("An Error Has Occurred");
        exit(1);
    }
    
    while (fgets(line, sizeof(line), file)) {
        /* Split each line by commas and convert to double */
        token = strtok(line, ",");
        for (col = 0; col < d; col++) {
            if (token) {
                data[row * d + col] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        row++;
    }
    
    fclose(file);

    return data;
}

/* Function to calculate the squared Euclidean distance between two vectors */
double calc_squared_euclidean_distance_for_two_vectors(double *A, double *B, int d){
    double dist_squared = 0.0;
    int k;
    for (k = 0; k < d; k++) {
        double diff = A[k] - B[k];
        dist_squared += diff * diff;
    }
    return dist_squared;
}
