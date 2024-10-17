import sys
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf
from sklearn.cluster import KMeans

# Function to parse command line arguments
def parse_args():
    if len(sys.argv) != 3:
        print("An Error Has Occurred1")
        sys.exit(1)
    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
        return k, file_name
    except ValueError:
        print("An Error Has Occurred2")
        sys.exit(1)

# Function to read data points from file
def read_data(file_name):
    try:
        data = np.loadtxt(file_name)
        return data
    except Exception:
        print("An Error Has Occurred3")
        sys.exit(1)

# Main function to compare SymNMF and KMeans
def main():
    k, file_name = parse_args()
    X = read_data(file_name)

    # SymNMF clustering
    W = symnmf.compute_similarity_matrix(X.tolist())
    H_init = np.random.uniform(0, 2 * np.sqrt(np.mean(W) / k), size=(len(W), k))
    H_final = symnmf.symnmf(W, H_init.tolist())
    labels_symnmf = np.argmax(H_final, axis=1)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=1234).fit(X)
    labels_kmeans = kmeans.labels_

    # Calculate silhouette scores
    score_symnmf = silhouette_score(X, labels_symnmf)
    score_kmeans = silhouette_score(X, labels_kmeans)

    # Output the scores
    print(f"nmf: {score_symnmf:.4f}")
    print(f"kmeans: {score_kmeans:.4f}")

if __name__ == "__main__":
    main()
