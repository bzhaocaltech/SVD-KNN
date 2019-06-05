#include "helper_cuda.h"
#include "gpu_knn.cuh"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>


#define IDX2C(i, j, ld) ((i * ld) + j)


/**
 * Attempt to predict on a single data point given a training set of similarly
 * sized points using KNN. This method is GPU accelerated using a reduction not
 * unlike the one done in Lab 3, but with a euclidian distance and min.
 */
__global__ void gpu_knn_predict_one_kernel(
  float *d_point,
  float *d_training_set,
  int num_points,
  int point_size,
  int num_neighbors,
  int num_classes,
  float *d_neighbors
) {
  // Has size sizeof(float) * 2 * threads_per_block * num_neighbors
  extern __shared__ float min_points[];
  
  // Initialize min_points for this thread.
  for (int i = threadIdx.x; i < threadIdx.x + num_neighbors; i++) {
    min_points[IDX2C(i, 0, 2)] = -1;
    min_points[IDX2C(i, 1, 2)] = 0;
  }
  
  // Loop through the points in the training set.
  int stride = blockDim.x * gridDim.x;
  for (
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    thread_index < num_points;
    thread_index += stride
  ) {
    // Get the point and calculate its distance.
    float dist_sum = 0;
    for (int i = 1; i < point_size; i++) {
      dist_sum += pow(d_point[i] - d_training_set[IDX2C(thread_index, i, point_size)], 2);
    }
    float dist = sqrt(dist_sum);
    
    // Loop through this thread's shared memory slot and insert into shared
    // memory the point and it's distance if it is a min dist neighbor.
    for (int i = threadIdx.x; i < threadIdx.x + num_neighbors; i++) {
      if (min_points[IDX2C(i, 0, 2)] == -1 || dist < min_points[IDX2C(i, 1, 2)]) {
        min_points[IDX2C(i, 0, 2)] = thread_index;
        min_points[IDX2C(i, 1, 2)] = dist;
        break;
      }
    }
  }

  __syncthreads();

  // The shared memory should be filled with the min distances.
  // We can do a reduction to boil down the minimums.
  for (stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      // Go over the minimum neighbors that the thread found and reduce only
      // if they are smaller than the current minimums.
      for (int i = threadIdx.x; i < threadIdx.x + num_neighbors; i++) {
        if (
          min_points[IDX2C(i + stride, 0, 2)] != -1 &&
          min_points[IDX2C(i + stride, 1, 2)] < min_points[IDX2C(i, 1, 2)]
        ) {
          min_points[IDX2C(i, 0, 2)] = min_points[IDX2C(i + stride, 0, 2)];
          min_points[IDX2C(i, 1, 2)] = min_points[IDX2C(i + stride, 1, 2)];
        }
      }
    }

    __syncthreads();
  }

  // The first thread now just tries to copy over its minimums.
  if (threadIdx.x == 0) {
    for (int i = 0; i < num_neighbors; i++) {
      if (
        d_neighbors[IDX2C(i, 0, 2)] == -1 || (
          min_points[IDX2C(i, 0, 2)] != -1 &&
          min_points[IDX2C(i, 1, 2)] < d_neighbors[IDX2C(i, 1, 2)]
        )
      ) {
        d_neighbors[IDX2C(i, 0, 2)] = min_points[IDX2C(i, 0, 2)];
        d_neighbors[IDX2C(i, 1, 2)] = min_points[IDX2C(i, 1, 2)];
      }
    }
  }
}

/**
 * Perform a single GPU KNN prediction.
 */
float gpu_knn_predict_one(
  const unsigned num_blocks,
  const unsigned num_threads_per_block,
  float *point,
  float *training_set,
  int num_points,
  int point_size,
  int num_neighbors,
  int num_classes
) {
  // Allocate space on the device for the training data, prediction point,
  // and for the result neighbors.
  float *d_point;
  CUDA_CALL( cudaMalloc(&d_point, point_size * sizeof(float)));
  CUDA_CALL( cudaMemcpy(d_point, point, point_size * sizeof(float), cudaMemcpyHostToDevice));

  float *d_training_set;
  CUDA_CALL( cudaMalloc(&d_training_set, num_points * point_size * sizeof(float)));
  CUDA_CALL( cudaMemcpy(d_training_set, training_set, num_points * point_size * sizeof(float), cudaMemcpyHostToDevice));

  float *d_neighbors;
  CUDA_CALL( cudaMalloc(&d_neighbors, num_neighbors * 2 * sizeof(float)));
  
  // Use thrust to fill the neighbors with a default value.
  thrust::device_ptr<float> d_neighbors_ptr(d_neighbors);
  thrust::fill(d_neighbors_ptr, d_neighbors_ptr + (num_neighbors * 2), -1);
  
  gpu_knn_predict_one_kernel<<<
    num_blocks,
    num_threads_per_block,
    num_threads_per_block * num_neighbors * 2 * sizeof(float)
  >>>(
    d_point,
    d_training_set,
    num_points,
    point_size,
    num_neighbors,
    num_classes,
    d_neighbors
  );

  cudaError err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
  }

  float *neighbors = new float[num_neighbors * 2];
  CUDA_CALL( cudaMemcpy(
    neighbors,
    d_neighbors,
    num_neighbors * 2 * sizeof(float),
    cudaMemcpyDeviceToHost
  ));

  // Find the most common neighbor and return its class.
  int *class_counts = new int[num_classes];
  
  for (int i = 0; i < num_classes; i++) {
    class_counts[i] = 0;
  }

  for (int i = 0; i < num_neighbors; i++) {
    int idx = neighbors[IDX2C(i, 0, 2)];
    int class_id = training_set[IDX2C(idx, 0, point_size)];
    class_counts[class_id]++;
  }

  int max_class = 0;
  for (int i = 0; i < num_classes; i++) {
    if (class_counts[max_class] < class_counts[i]) {
      max_class = i;
    }
  }

  delete neighbors;

  delete class_counts;

  return max_class;
}

/**
 * Perform many GPU KNN predictions.
 */
float *gpu_knn_predict_many(
  const unsigned num_blocks,
  const unsigned num_threads_per_block,
  float *predict_points,
  int num_predict_points,
  float *training_set,
  int num_training_points,
  int point_size,
  int num_neighbors,
  int num_classes
) {
  float *res = new float[num_predict_points];

  float *temp = new float[point_size];
  
  for (int i = 0; i < num_predict_points; i++) {
    // Copy the point into temp to run it with predict_one. 
    for (int j = 0; j < point_size; j++) {
      temp[j] = predict_points[IDX2C(i, j, point_size)];
    }

    res[i] = gpu_knn_predict_one(num_blocks, num_threads_per_block, temp,
      training_set, num_training_points, point_size, num_neighbors, num_classes);
  }
  
  delete temp;

  return res;
}