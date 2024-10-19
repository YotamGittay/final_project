import sys
import os

MAX_ITER = 300


# TODO: Check that my tests are in the correct format
def parse_and_validate_args() -> tuple:
    if len(sys.argv) != 3:
        print("Please insert 2 arguments only: k, input_file")
        sys.exit(1)

    try:
        k = float(sys.argv[1])
        if k.is_integer():
            k = int(k)
        else:
            print("Invalid number of clusters!")
            sys.exit(1)
    except ValueError:
        print("Invalid number of clusters!")
        sys.exit(1)

    if k <= 1:
        print("Invalid number of clusters!")
        sys.exit(1)

    input_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"NA")
        sys.exit(1)
    return k, input_file


def load_data_as_tuples(file_path: str) -> list:
    points = []
    with open(file_path, "r") as file:
        for line in file:
            # Strip any leading/trailing whitespace and split by comma
            values = line.strip().split(",")
            # Convert values to float and create a tuple
            points.append(tuple(float(x) for x in values))
    return points


def euclidean_distance(point1: tuple, point2: tuple):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5


def assign_points_to_clusters(points: list, centroids: list):
    clusters = [[] for _ in range(len(centroids))]
    for point in points:
        closest_centroid = None
        closest_distance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(point, centroid)
            if distance < closest_distance:
                closest_distance = distance
                closest_centroid = i
        clusters[closest_centroid].append(point)
    return clusters


def update_centroids(clusters: list):
    new_centroids = []
    for cluster in clusters:
        cluster_size = len(cluster)
        cluster_sum = [sum(x) for x in zip(*cluster)]
        new_centroids.append(tuple(x / cluster_size for x in cluster_sum))
    return new_centroids


def check_convergence(centroids: list, new_centroids: list):
    epsilon = 0.0001
    for centroid, new_centroid in zip(centroids, new_centroids):
        if euclidean_distance(centroid, new_centroid) > epsilon:
            return False
    return True


def display_centroids(centroids: list):
    for centroid in centroids:
        print(",".join(str(f"{x:.4f}") for x in centroid))


def turn_nested_list_to_list_of_tuples(nested_list: list) -> list:
    return [tuple(x) for x in nested_list]


def kmeans_hw1(k: int, X: list) -> list:
    points = turn_nested_list_to_list_of_tuples(X)
    centroids = points[:k]
    for _ in range(MAX_ITER):
        clusters = assign_points_to_clusters(points, centroids)
        new_centroids = update_centroids(clusters)
        if check_convergence(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


def main():
    k, input_file = parse_and_validate_args()
    points = load_data_as_tuples(input_file)
    centroids = points[:k]
    for _ in range(MAX_ITER):
        clusters = assign_points_to_clusters(points, centroids)
        new_centroids = update_centroids(clusters)
        if check_convergence(centroids, new_centroids):
            break
        centroids = new_centroids
    display_centroids(centroids)


if __name__ == "__main__":
    main()
