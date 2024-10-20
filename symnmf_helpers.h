
void print_matrix(double *mat, int n, int d);

void transpose_matrix(double *mat, double *target, int n, int d);

void calc_degree_inv_sqrt_matrix(double *degree_matrix, double *degree_inv_sqrt, int n);

void copy_matrix(double *mat, double *target, int n, int d);

void multiply_matrices(double *A, double *B, double *result, int n, int m, int p);

int check_matrix_convergence(double *H, double *new_H, int n, int k);

void get_dimensions_from_file(char *file_name, int *n, int *d);

double* read_points_from_file(char *file_name, int n, int d);

double calc_squared_euclidean_distance_for_two_vectors(double *A, double *B, int d);
