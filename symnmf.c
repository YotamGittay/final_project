#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include "symnmf.h"

/* CONSTS*/
double const BETA = 0.5;
double const EPSILON = 1e-4;
int const MAX_ITER = 300;

/* HELPER FUNCTIONS */


void print_one_dim_mat(double *mat, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            printf("%.4f ", mat[row * (d) + col]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
}

void transpose_one_dim_mat(double *mat, double *target, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            target[col * n + row] = mat[row * d + col];
        }
    }
}

void copy_one_dim_mat(double *mat, double *target, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            target[row * d + col] = mat[row * d + col];
        }
    }
}

void multiply_matrices(double *A, double *B, double *result, int n, int m, int p) {
    int i, j, k;
    
    // Initialize the result matrix to zero
    for (i = 0; i < n * p; i++) {
        result[i] = 0.0;
    }
    
    // Perform matrix multiplication
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

/* Function to read data points from file */ 
double* read_data(char *file_name, int *n, int *d) {
    FILE *file;
    char line[1024];
    char *str, *token;
    int count, row, col;
    double *data;
    
    *n = 0;
    *d = 0;
    
    /* Open the file for reading */
    file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "An Error Has Occurred\n");
        exit(1);
    }
    
    /* First pass: Count the number of rows and columns */
    while (fgets(line, sizeof(line), file)) {
        /* If it's the first line, count the number of columns */
        if (*n == 0) {
            count = 0;
            str = line;
            while (*str) {
                if (*str == ',') {
                    count++;
                }
                str++;
            }
            *d = count + 1;  /* Number of columns is one more than the number of commas */
        }
        (*n)++;
    }
    
    /* Close the file after counting */
    fclose(file);
    
    /* Allocate memory for the data array based on n and d */
    data = (double*)calloc((*n) * (*d), sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    /* Second pass: Reopen the file and read the data into the array */
    file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "An Error Has Occurred\n");
        exit(1);
    }
    
    row = 0;
    
    while (fgets(line, sizeof(line), file)) {
        /* Split each line by commas and convert to double */
        token = strtok(line, ",");
        for (col = 0; col < *d; col++) {
            if (token) {
                data[row * (*d) + col] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        row++;
    }
    
    /* Close the file after reading */
    fclose(file);

    return data;
}

/* Function to calculate the squared Euclidean distance between two vectors */
double calc_squared_euclidan_distance_for_two_vectors(double *A, double *B, int d){
    double dist_squared = 0.0;
    for (int k = 0; k < d; k++) {
        double diff = A[k] - B[k];
        dist_squared += diff * diff;
    }
    return dist_squared;
}

/* MAIN LOGIC FUNCTIONS*/

/* Function to optimize H for SymNMF */
void symnmf(double *W, double *H, int n, int k) {
    int iter, i, j, converged;
    // W = n x n. H = n x k. WH = n x k. HHT = n x n.  HHTH = n x k
    double *H_transpose = (double *)calloc(k * n, sizeof(double));
    double *WH = (double *)calloc(n * k, sizeof(double));
    double *temp_HHT = (double *)calloc(n * n, sizeof(double));
    double *HHTH = (double *)calloc(n * k, sizeof(double));
    double *new_H = (double *)calloc(n * k, sizeof(double));
    if (!H_transpose ||!WH || !temp_HHT || !HHTH) {
        fprintf(stderr, "An Error Has Occurred100");
        exit(1);
    }

    for (iter = 0; iter < MAX_ITER; iter++) {
        /* Compute WH = W * H */ 
        multiply_matrices(W, H, WH, n, n, k);

        /* transpose H and compute HHTH = (H*H^T) * H */ 
        transpose_one_dim_mat(H, H_transpose, n, k );
        multiply_matrices(H, H_transpose, temp_HHT, n, k, n);
        multiply_matrices(temp_HHT, H, HHTH, n, n, k);

        /* calc new_H */ 
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                new_H[i * k + j] = H[i * k + j] * (1 - BETA + (BETA * WH[i * k + j] / HHTH[i * k + j]));
            }
        }

        converged = check_matrix_convergence(H, new_H, n, k);
        copy_one_dim_mat(new_H, H, n, k);
        if (converged == 1) {
            break;
        }
        
    }

    free(WH);
    free(HHTH);
    free(H_transpose);
    free(temp_HHT);
    free(new_H);
} 


/* Function to calculate the similarity matrix */
void sym(double *data, int n, int d, double *similarity_matrix) {
    // printf("entered sym\n");

    int i, j;
    double dist_squared;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                similarity_matrix[i * n + j] = 0.0;
            } else {
                dist_squared = calc_squared_euclidan_distance_for_two_vectors(&data[i * d], &data[j * d], d);
                similarity_matrix[i * n + j] = exp(-dist_squared / 2.0);
            }
        }
    }
    // printf("similarity matrix:\n");
    // print_one_dim_mat(similarity_matrix, n, n);
    // printf("exiting sym\n");
}


/* Function to compute diagonal degree matrix*/ 
void ddg(double *data, int n, int d, double *degree_matrix) {
    // printf("entered ddg\n");
    
    int i, j;
    double degree_sum;
    double *similarity_matrix = (double*)calloc(n * n, sizeof(double));
    if (similarity_matrix == NULL) {
        fprintf(stderr, "An Error Has Occurred 101\n");
        exit(1);
    }

    sym(data, n, d, similarity_matrix);
    
    /* Calc degree sum for every row */
    for (i = 0; i < n; i++) {
        degree_sum = 0.0;
        
        for (j = 0; j < n; j++) {
            degree_sum += similarity_matrix[i * n + j];
        }
        
        degree_matrix[i * n + i] = degree_sum;
    }

    // printf("degree matrix:\n");
    // print_one_dim_mat(degree_matrix, n, n);
    // printf("exiting ddg\n");

    free(similarity_matrix);
}


/* Function to compute normalized similarity matrix*/ 
void norm(double *data, int n, int d, double *norm_matrix) {
    // printf("entered norm\n");
    int i;
    double *similarity_matrix = (double*)calloc(n * n, sizeof(double));
    double *degree_matrix = (double*)calloc(n * n, sizeof(double));
    double *degree_inv_sqrt = (double*)calloc(n * n, sizeof(double));
    
    if (similarity_matrix == NULL || degree_matrix == NULL || degree_inv_sqrt == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    sym(data, n, d, similarity_matrix);
    ddg(data, n, d, degree_matrix);
    for (i = 0; i < n * n; i++) {
        double degree_value = degree_matrix[i];
        if (degree_value != 0.0) {
            degree_inv_sqrt[i] = 1.0 / sqrt(degree_value);
        } else {
            degree_inv_sqrt[i] = 0.0;
        }
    }
    
    double *result_1 = (double*)calloc(n * n, sizeof(double));
    multiply_matrices(degree_inv_sqrt, similarity_matrix, result_1, n, n, n);
    multiply_matrices(result_1, degree_inv_sqrt, norm_matrix, n, n, n); 

    // printf("normalized matrix:\n");
    // print_one_dim_mat(norm_matrix, n, n);
    // printf("exiting norm\n");

    free(similarity_matrix);
    free(degree_matrix);
}

/* Main function*/ 
int main(int argc, char *argv[]) {
    char *goal;
    char *file_name;
    int n, d;
    double *data, *similarity_matrix, *degree_matrix, *norm_matrix;

    if (argc != 3) {
        fprintf(stderr, "An Error Has Occurred7\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];
    data = read_data(file_name, &n, &d);


    if (strcmp(goal, "sym") == 0) {
        similarity_matrix = (double *)calloc(n * n, sizeof(double));
        if (!similarity_matrix) {
            fprintf(stderr, "An Error Has Occurred8\n");
            return 1;
        }
        sym(data, n, d, similarity_matrix);
        print_one_dim_mat(similarity_matrix, n, n); 
        free(similarity_matrix);
    } else if (strcmp(goal, "ddg") == 0) {
        degree_matrix = (double *)calloc(n * n, sizeof(double));
        if (!degree_matrix) {
            fprintf(stderr, "An Error Has Occurred9\n");
            return 1;
        }
        ddg(data, n, d, degree_matrix);
        print_one_dim_mat(degree_matrix, n, n); 
        free(degree_matrix);
    } else if (strcmp(goal, "norm") == 0) {
        norm_matrix = (double *)calloc(n * n, sizeof(double));
        if (!norm_matrix) {
            fprintf(stderr, "An Error Has Occurred10\n");
            return 1;
        }
        norm(data, n, d, norm_matrix);
        print_one_dim_mat(norm_matrix, n, n); 
        free(norm_matrix);
    } else {
        fprintf(stderr, "An Error Has Occurred11\n");
        return 1;
    }

    free(data);
    return 0;
}
