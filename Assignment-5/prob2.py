import numpy as np

x = np.array(
    [
        2.10,
        4.20,
        6.00,
        4.50,
        4.20,
        4.30,
        2.50,
        6.90,
        8.10,
        7.00,
        9.10,
        5.30,
        6.20,
        4.80,
        1.80,
        5.80,
        3.30,
        3.30,
        4.00,
        6.4,
    ]
)

mu = 5.1
sigma = 1.9
is_outlier = (x < (mu - 2 * sigma)) | (x > (mu + 2 * sigma))
outliers = x[is_outlier]
print(f"Outliiers:\n{outliers}")

x_outliers_removed = x[~is_outlier]
print(f"X Outliers Removed:")
print(x_outliers_removed)
