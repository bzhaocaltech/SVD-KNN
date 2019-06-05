#include <cuda_runtime.h>


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
);

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
);
