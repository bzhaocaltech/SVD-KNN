# SVD and KNN CPU and GPU Implementations

## CPU Implementations
To run KNN, first make sure you have a `data` directory in the root folder.
Then, run `create_test_data.py`. If you wish to generate new data, make sure to
manually delete old data before running the creation script again.
Then, run `make run_knn` and then `./run_knn`.
NOTE: The data/ directory should already be set up in the zip file, so unless
you wish to make changes to the data, you shouldn't need to rerun the python
file.

The generated KNN data has the following form. Each row represents a data point:
[`class`, `x_0`, `x_1`]

The SVD algorithm takes as a training set a sparse matrix. The sparse matrix
is represented by a float**. The leading dimension is the number of data points
while the each individual float* array takes the form (x_index, y_index, value).
The SVD algorithm uses gradient descent to generate two matrices of latent
factors that are used to predict the value of other points given a x_index and
y_index.

The executable `./run_svd` runs the SVD algorithm on a dataset found in the data
directory. The dataset comes from https://www.kaggle.com/netflix-inc/netflix-prize-data/downloads/netflix-prize-data.zip/1. To make the executable, run `make run_svd`


## GPU Implementations
Work in progress.
