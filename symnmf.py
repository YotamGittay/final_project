import sys
import numpy as np
import symnmf_module


# Function to parse command line arguments
def parse_args():
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
        return k, goal, file_name
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)


# Function to read data points from file
def read_data(file_name):
    try:
        data = np.loadtxt(file_name, delimiter=",").tolist()
        return data
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


# Initialize H according to Section 1.4.1
def initialize_h(W, k):
    m = m = sum(map(sum, W)) / (len(W) * len(W[0]))
    np.random.seed(1234)
    return [[np.random.uniform(0, 2 * np.sqrt(m / k)) for _ in range(k)] for _ in range(len(W))]


# A function for analysis, gets the point and the number of clusters and returns the H matrix
def get_symnmf_H_matrix(X, k):
    W = symnmf_module.norm(X)
    H_init = initialize_h(X, k)
    return symnmf_module.symnmf(W, H_init)


# Main function to execute the desired operation
def main():
    k, goal, file_name = parse_args()
    X = read_data(file_name)

    # check if k is an integer smaller than the number of points (rows of x)
    if not isinstance(k, int) or k < 1 or k > len(X):
        print("An Error Has Occurred")
        sys.exit(1)

    if goal == "sym":
        similarity_matrix = symnmf_module.sym(X)
        np.savetxt(sys.stdout, similarity_matrix, delimiter=",", fmt="%.4f")

    elif goal == "ddg":
        degree_matrix = symnmf_module.ddg(X)
        np.savetxt(sys.stdout, degree_matrix, delimiter=",", fmt="%.4f")

    elif goal == "norm":
        norm_matrix = symnmf_module.norm(X)
        np.savetxt(sys.stdout, norm_matrix, delimiter=",", fmt="%.4f")

    elif goal == "symnmf":
        W = symnmf_module.norm(X)
        H_init = initialize_h(W, k)
        H_final = symnmf_module.symnmf(W, H_init)
        np.savetxt(sys.stdout, H_final, delimiter=",", fmt="%.4f")

    else:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
