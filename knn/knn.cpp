#include "knn.hpp"

#include <math.h>

KNN::KNN(
  float **training_set,
  int num_points,
  int point_size,
  int num_neighbors,
  int num_classes
) {
  this->training_set = training_set;
  this->num_points = num_points;
  this->point_size = point_size;
  this->num_neighbors = num_neighbors;
  this->num_classes = num_classes;
}

float KNN::predict(float *data) {
  int *class_counts = new int[num_classes];

  PointDistance *neighbors = find_nearest_neighbors(data);
  for (int i = 0; i < num_neighbors; i++) {
    class_counts[neighbors[i].index]++;
  }

  int max_index = 0;
  for (int i = 0; i < num_neighbors; i++) {
    if (class_counts[max_index] < class_counts[i]) {
      max_index = i;
    }
  }

  return max_index;
}

PointDistance* KNN::find_nearest_neighbors(float *data) {
  PointDistance *neighbors = new PointDistance[num_neighbors];
  
  // Fill `neighbors` with the first `num_neighbors` points from the training
  // set.
  for (int i = 0; i < num_neighbors; i++) {
    float *training_point = training_set[i];
    PointDistance pd = {
      .index = i,
      .distance = distance(data, training_point),
      .point = training_point,
    };
    neighbors[i] = pd;
  }

  // Process the rest of the points in the training set.
  for (int i = num_neighbors; i < num_points; i++) {
    float *training_point = training_set[i];
    PointDistance pd = {
      .index = i,
      .distance = distance(data, training_point),
      .point = training_point,
    };
    insert(neighbors, pd);
  }

  return neighbors;
}

/**
 * Insert the value `d` into `arr` preserving ordering of greatest to least
 * based on distance in `arr`.
 */
void KNN::insert(PointDistance *arr, PointDistance pd) {
  PointDistance *end = arr + num_neighbors;
  for (PointDistance *iter = arr; iter != end; iter++) {
    if (pd.distance < iter->distance) {
      *iter = pd;
      return;
    }
  }
}

/**
 * Calculate the euclidean distance for two points.
 */
float KNN::distance(float *x, float *y) {
  float sum = 0;
  for (int i = 0; i < point_size; i++) {
    sum += pow(x[i] - y[i], 2);
  }
  return sqrt(sum);
}