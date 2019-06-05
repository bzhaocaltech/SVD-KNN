# SVD and KNN CPU and GPU Implementations

# SVD Implementation

## Overview

Our SVD algorithm attempts to find two matrices U and V such that U * V^T = Y
where Y is a sparse matrix given as a training set. The SVD algorithm also takes
into account two bias vectors, a and b, for rows U and V respectively as well as
a global bias mu. Parameters that can be given to the SVD include latent_factors
(the length of each row of U and V), the learning rate eta, a regularization
constant reg, and the number of epochs to train the SVD on. The SVD is trained
on the Netflix dataset (found at https://www.kaggle.com/netflix-inc/netflix-prize-data/downloads/netflix-prize-data.zip/1),
which contains about 100 million points. Every hundredth point from the dataset
is placed onto a validation dataset instead of the training set.

You can build the `./run_svd` executable by running `make run_svd`. `run_svd`
takes in 7 command line arguments: `./run_svd [blocks] [threads/block] [latent_factors] [eta] [reg] [num_epochs] [use_cpu (0 for no cpu 1 for cpu)]`
The last argument is there since the cpu code can be realllllyyy slow. Also
note that since the dataset is so large, occasionally the kernel will crash
with a cudaErrorLaunchTimeout error. This seems to happen whenever the kernel
takes longer than 15 seconds to run. If this occurs, just increase the number
of blocks and/or threads/block. The run_svd executable prints out the training
and validation errors for both the cpu and gpu code. Additionally, (if the last
command line argument is set to 1), then it also prints the mean squared
difference between the predictions of the cpu and gpu on the validation set.

run_svd.cpp is the test script for svds. That source code for svds is in
svd/ directory.

## CPU Implementation

The code for this is primarily within svd.cpp and svd.hpp. The SVD class can
constructed and uses the methods svd->train and svd->predict to train on a
dataset and predict a dataset, respectively. An optional validation set can
be passed to svd->train. For the CPU algorithm, each point from the dataset
is processed sequentially. To do dot products and vector math in general, for
loops are used.

## GPU Implementation

The code for this is primarily within gpu_svd.cu and gpu_svd.cuh. Unlike the
CPU code, which makes use of a SVD class, the GPU code has a much more
light-weight setup. All relevant information for a SVD is stored within the
GPU_SVD struct. A GPU_SVD struct can be initialized and free by using the
createGPUSVD and freeGPUSVD methods, respectively.
Train and predict, rather than being class methods, take in
a GPU_SVD struct as an argument. We did this since device functions and class
methods don't go too well together.

The train and predict kernels of the GPU code work have two levels of
parallelization. Firstly, in each loop, each block will be responsible for
processing a single point. The index of the training data that the block
processes is increased by gridDim.x per loop. Individual threads of each block
are responsible for doing the reductions necessary to do the vector math.

For the predict kernel, the shared memory only has enough space for the
reduction to occur. For the train kernel, since the thread needs to use indices
from U and V multiple times (instead of just once like for the prediction),
the shared memory also contains space to copy down rows of U and V.

Also note that the training method of the GPU code uses Hogwild. Hogwild algorithms
ignores race conditions, locks, and the like. Despite this, it has seem great
success in a number of machine learning algorithms. However, because we are
ignoring race conditions in the GPU code, the GPU code produces less accurate
results than the CPU code. That being said, the fact that the gpu code is soooo
much faster per epoch more than makes up (in my opinion)
for any slight increase in the error per epoch.

## KNN Implementation

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
