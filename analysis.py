import sys
import numpy as np
from sklearn.metrics import silhouette_score
from kmeans import kmeans_hw1
from symnmf import get_symnmf_H_matrix


# Function to parse command line arguments
def parse_args():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
        return k, file_name
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


# Function that gets the points and the H matrix and returns the cluster label for each point
def H_matrix_to_cluster_labels(H):
    return np.argmax(H, axis=1)


# TODO: test that it is working correctly
# Function that gets the points and the centroids and returns the clusters
def get_kmeans_labels_by_centroids(X, centroids):
    return np.argmax(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)


# Main function to compare SymNMF and KMeans
def main():
    k, file_name = parse_args()
    X = read_data(file_name)

    # SymNMF clustering
    # here we need somehow to get the H matrix, this function should do the job
    # but I wasn't able to import it
    H = get_symnmf_H_matrix(X, k)
    labels_symnmf = H_matrix_to_cluster_labels(H)

    # KMeans clustering
    kmeans_centroids = kmeans_hw1(X, k)
    labels_kmeans = get_kmeans_labels_by_centroids(X, kmeans_centroids)

    # Calculate silhouette scores
    score_symnmf = silhouette_score(X, labels_symnmf)
    score_kmeans = silhouette_score(X, labels_kmeans)

    # Output the scores
    print(f"nmf: {score_symnmf:.4f}")
    print(f"kmeans: {score_kmeans:.4f}")


if __name__ == "__main__":
    main()
