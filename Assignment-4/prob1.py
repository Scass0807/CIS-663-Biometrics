import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_similarity,
)

iris_df = pd.read_csv(
    "iris_deduplicated.data",
    names=[
        "sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm",
        "class",
    ],
)
iris_data = iris_df[
    [
        "sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm",
    ]
].to_numpy()


def find_k_closest_pairs(data: np.ndarray, algorithim: str, k: int = 3):

    distance_funcs = {
        "euclidean": euclidean_distances,
        "manhattan": manhattan_distances,
        "cosine": cosine_similarity,
    }
    results = {}
    fn = distance_funcs[algorithim]
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance = fn(np.expand_dims(data[i], 0), np.expand_dims(data[j], 0)).item()
            key = (i, j)
            results[key] = distance

    sorted_results = sorted(
        results.items(),
        key=lambda cols: cols[1],
        reverse=True if algorithim == "cosine" else False,
    )
    top_k = sorted_results[:k]
    return top_k


algorithims = ["euclidean", "manhattan", "cosine"]
k_pairs = 3
for alg in algorithims:
    print(f"{alg} results")
    results = find_k_closest_pairs(iris_data, alg, k_pairs)
    for pair in range(k_pairs):
        print(f"Pair {pair}:")
        indices, distance = results[pair]
        print(iris_df.iloc[list(indices)])
        print(
            f"Distance: {distance}\n"
            if alg != "cosine"
            else f"Similarity: {distance}\n"
        )


#     np.fill_diagonal(distance_matrix, distance_matrix.max())
#     print(distance_matrix)


# find_k_closest_pairs(iris_data,algorithim="cityblock")
