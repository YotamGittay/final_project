#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

void print_one_dim_mat(double *mat, int n, int d) {
    int row, col;
    for (row = 0; row < n; row++) {
        for (col = 0; col < d; col++) {
            printf("%.4f ", mat[row * (d) + col]);
        }
        printf("\n");
    }
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
    data = (double*)malloc((*n) * (*d) * sizeof(double));
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


/* Function to optimize H for SymNMF */
void symnmf(double *W, double *H, int n, int k, int max_iter, double tol) {
    int iter, i, j, l;
    double *WH = (double *)malloc(n * k * sizeof(double));
    double *HHTH = (double *)malloc(n * k * sizeof(double));
    if (!WH || !HHTH) {
        fprintf(stderr, "An Error Has Occurred100");
        exit(1);
    }

    for (iter = 0; iter < max_iter; iter++) {
        /* Compute WH = W * H */ 
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                WH[i * k + j] = 0;
                for (l = 0; l < n; l++) {
                    WH[i * k + j] += W[i * n + l] * H[l * k + j];
                }
            }
        }

        /* Compute HHTH = (H^T * H) * H */ 
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                HHTH[i * k + j] = 0;
                for (l = 0; l < k; l++) {
                    HHTH[i * k + j] += H[i * k + l] * H[l * k + j];
                }
            }
        }

        /* Update H */ 
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                /*looks like the update is incorrect*/
                H[i * k + j] *= (0.5 * (WH[i * k + j] / HHTH[i * k + j]));
            }
        }

        /* Check for convergence (optional) */ 
        double diff = 0;
        for (i = 0; i < n * k; i++) {
            diff += fabs(WH[i] - HHTH[i]);
        }
        if (diff < tol) {
            break;
        }
    }

    free(WH);
    free(HHTH);
} 

/* Function to compute similarity matrix*/ 
#include <math.h>

/* Function to calculate the similarity matrix */
void sym(double *data, int n, int d, double *similarity_matrix) {
   
    int i, j, k;
    double dist_squared;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                similarity_matrix[i * n + j] = 0.0;
            } else {
                dist_squared = 0.0;
                for (k = 0; k < d; k++) {
                    double diff = data[i * d + k] - data[j * d + k];
                    dist_squared += diff * diff;
                }
                similarity_matrix[i * n + j] = exp(-dist_squared / 2.0);
            }
        }
    }
}


/* Function to compute diagonal degree matrix*/ 
#include <stddef.h>

void ddg(double *data, int n, int d, double *degree_matrix) {
    int i, j;
    double degree_sum;
    
    for (i = 0; i < n * n; i++) {
        degree_matrix[i] = 0.0;
    }
    
    for (i = 0; i < n; i++) {
        degree_sum = 0.0;
        
        for (j = 0; j < n; j++) {
            degree_sum += data[i * n + j];
        }
        
        degree_matrix[i * n + i] = degree_sum;
    }
}


/* Function to compute normalized similarity matrix*/ 
void norm(double *data, int n, int d, double *norm_matrix) {
    int i, j, k;
    double *similarity_matrix = (double*)malloc(n * n * sizeof(double));
    double *degree_matrix = (double*)malloc(n * n * sizeof(double));
    double *degree_inv_sqrt = (double*)malloc(n * n * sizeof(double));
    
    if (similarity_matrix == NULL || degree_matrix == NULL || degree_inv_sqrt == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    sym(data, n, d, similarity_matrix);
    //print_one_dim_mat(similarity_matrix, n, n);
    ddg(similarity_matrix, n, d, degree_matrix);
    //print_one_dim_mat(degree_matrix, n, n);
    for (i = 0; i < n * n; i++) {
        double degree_value = degree_matrix[i];
        if (degree_value != 0.0) {
            degree_inv_sqrt[i] = 1.0 / sqrt(degree_value);
        } else {
            degree_inv_sqrt[i] = 0.0;
        }
    }
    
    double *result_1 = (double*)malloc(n * n * sizeof(double));
    mul_mat(degree_inv_sqrt, similarity_matrix, result_1, n);
    mul_mat(result_1, degree_inv_sqrt, norm_matrix, n); 
}

void mul_mat(double *A, double *B, double *result, int dim) {
    int i, j, k;
    
    // Initialize the result matrix to zero
    for (i = 0; i < dim * dim; i++) {
        result[i] = 0.0;
    }
    
    // Perform matrix multiplication
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            for (k = 0; k < dim; k++) {
                result[i * dim + j] += A[i * dim + k] * B[k * dim + j];
            }
        }
    }
}

/* Main function*/ 
int main(int argc, char *argv[]) {
    char *goal;
    char *file_name;
    int n, d, i, j;
    double *data, *similarity_matrix, *degree_matrix, *norm_matrix;

    if (argc != 3) {
        fprintf(stderr, "An Error Has Occurred7\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];
    data = read_data(file_name, &n, &d);


    if (strcmp(goal, "sym") == 0) {
        similarity_matrix = (double *)malloc(n * n * sizeof(double));
        if (!similarity_matrix) {
            fprintf(stderr, "An Error Has Occurred8\n");
            return 1;
        }
        sym(data, n, d, similarity_matrix);
        /* print_one_dim_mat(similarity_matrix, n, n); */
    } else if (strcmp(goal, "ddg") == 0) {
        similarity_matrix = (double *)malloc(n * n * sizeof(double));
        degree_matrix = (double *)calloc(n * n, sizeof(double));
        if (!similarity_matrix || !degree_matrix) {
            fprintf(stderr, "An Error Has Occurred9\n");
            return 1;
        }
        sym(data, n, d, similarity_matrix);
        ddg(similarity_matrix, n, d, degree_matrix);
        print_one_dim_mat(degree_matrix, n, n);
    } else if (strcmp(goal, "norm") == 0) {
        norm_matrix = (double *)malloc(n * n * sizeof(double));
        if (!similarity_matrix || !degree_matrix || !norm_matrix) {
            fprintf(stderr, "An Error Has Occurred10\n");
            return 1;
        }
        norm(data, n, d, norm_matrix);
        print_one_dim_mat(norm_matrix, n, n);
    } else {
        fprintf(stderr, "An Error Has Occurred11\n");
        return 1;
    }

    free(data);
    return 0;
}
