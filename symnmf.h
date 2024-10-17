#ifndef SYMNMF_H
#define SYMNMF_H

/* Function to read data points from file */
double* read_data(char *file_name, int *n, int *d);

/* Function to compute similarity matrix */
void sym(double *data, int n, int d, double *similarity_matrix);

/* Function to compute diagonal degree matrix */
void ddg(double *data, int n, int d, double *degree_matrix);

/* Function to compute normalized similarity matrix */
void norm(double *data, int n, int d, double *norm_matrix);

/* Function to compute symnmf*/
void symnmf(double *W, double *H, int n, int k, int max_iter, double tol);

/* Function to multiply square matrices*/
void mul_mat(double *A, double *B, double *result, int dim);

#endif
