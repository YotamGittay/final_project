#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"
#include "symnmf_helpers.h"

/* CONSTS*/
double const BETA = 0.5;
int const MAX_ITER = 300;


/* MAIN LOGIC FUNCTIONS*/

/* Function to optimize H for SymNMF */
void symnmf(double *W, double *H, int n, int k) {
    int iter, i, j, converged;
    double denominator, numerator;
    double *H_transpose = (double *)calloc(k * n, sizeof(double));
    double *WH = (double *)calloc(n * k, sizeof(double));
    double *temp_HHT = (double *)calloc(n * n, sizeof(double));
    double *HHTH = (double *)calloc(n * k, sizeof(double));
    double *new_H = (double *)calloc(n * k, sizeof(double));
    check_memory_allocation_symnmf(H_transpose, WH, temp_HHT, HHTH, new_H);

    for (iter = 0; iter < MAX_ITER; iter++) {
        /* Compute WH = W * H */ 
        multiply_matrices(W, H, WH, n, n, k);
        /* transpose H and compute HHTH = (H*H^T) * H */ 
        transpose_matrix(H, H_transpose, n, k );
        multiply_matrices(H, H_transpose, temp_HHT, n, k, n);
        multiply_matrices(temp_HHT, H, HHTH, n, n, k);

        /* calc new_H */ 
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                numerator = WH[i * k + j];
                denominator = HHTH[i * k + j];
                if (denominator == 0.0) {
                    new_H[i * k + j] = 0.0;
                } else {
                    new_H[i * k + j] = H[i * k + j] * (1 - BETA + (BETA * numerator / denominator));
                }
            }
        }
        converged = check_matrix_convergence(H, new_H, n, k);
        copy_matrix(new_H, H, n, k);
        if (converged == 1) {
            break;
        }  
    }
    free(WH); free(HHTH); free(H_transpose); free(temp_HHT); free(new_H);
} 


/* Function to calculate the similarity matrix */
void sym(double *data, int n, int d, double *similarity_matrix) { 
    int i, j;
    double dist_squared;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                similarity_matrix[i * n + j] = 0.0;
            } else {
                dist_squared = calc_squared_euclidean_distance_for_two_vectors(&data[i * d], &data[j * d], d);
                similarity_matrix[i * n + j] = exp(-dist_squared / 2.0);
            }
        }
    }
}


/* Function to compute diagonal degree matrix*/ 
void ddg(double *data, int n, int d, double *degree_matrix) {
    
    int i, j;
    double degree_sum;
    double *similarity_matrix = (double*)calloc(n * n, sizeof(double));
    if (similarity_matrix == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }

    sym(data, n, d, similarity_matrix);
    
    /* Calc degree sum for every row of the symetric matrix */
    for (i = 0; i < n; i++) {
        degree_sum = 0.0;
        
        for (j = 0; j < n; j++) {
            degree_sum += similarity_matrix[i * n + j];
        }
        
        degree_matrix[i * n + i] = degree_sum;
    }

    free(similarity_matrix);
}


/* Function to compute normalized similarity matrix*/ 
void norm(double *data, int n, int d, double *norm_matrix) {
    double *similarity_matrix = (double*)calloc(n * n, sizeof(double));
    double *degree_matrix = (double*)calloc(n * n, sizeof(double));
    double *degree_inv_sqrt = (double*)calloc(n * n, sizeof(double));
    double *result_1= (double*)calloc(n * n, sizeof(double));
    check_memory_allocation_norm(similarity_matrix, degree_matrix, degree_inv_sqrt, result_1);

    /* calc sym and ddg matrices*/
    sym(data, n, d, similarity_matrix);
    ddg(data, n, d, degree_matrix);
    /* calc degree_inv_sqrt matrix (D^-0.5)*/
    calc_degree_inv_sqrt_matrix(degree_matrix, degree_inv_sqrt, n);
    /* Calc (D^-0.5 * A * D^-0.5) in two steps */
    multiply_matrices(degree_inv_sqrt, similarity_matrix, result_1, n, n, n);
    multiply_matrices(result_1, degree_inv_sqrt, norm_matrix, n, n, n); 

    free(similarity_matrix);
    free(degree_matrix);
    free(degree_inv_sqrt);
    free(result_1);
}

/* Main function*/ 
int main(int argc, char *argv[]) {
    char *goal, *file_name;
    int n = 0, d = 0;
    double *data, *goal_matrix;

    if (argc != 3) {
        printf("An Error Has Occurred");
        return 1;
    }
    goal = argv[1];
    file_name = argv[2];

    /* Read the data points from the file, first calculate num of points(n) and dimension(d) */
    get_dimensions_from_file(file_name, &n, &d); /* updates n and d*/
    data = read_points_from_file(file_name, n, d);
    /* initialize our goal_matrix, will be used for similarity/degree/norm, depending on goal */
    goal_matrix = (double *)calloc(n * n, sizeof(double));
    if (!goal_matrix) {
        printf("An Error Has Occurred");
        return 1;
    }

    if (strcmp(goal, "sym") == 0) {
        sym(data, n, d, goal_matrix);
    } else if (strcmp(goal, "ddg") == 0) {
        ddg(data, n, d, goal_matrix);
    } else if (strcmp(goal, "norm") == 0) {
        norm(data, n, d, goal_matrix);
    } else {
        printf("An Error Has Occurred");
        free(goal_matrix);
        free(data);
        return 1;
    }

    print_matrix(goal_matrix, n, n); 
    free(goal_matrix);
    free(data);
    return 0;
}
