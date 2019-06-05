#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <iostream>

#include "knn/knn.hpp"
#include "knn/gpu_knn.cuh"

#define IDX2C(i, j, ld) ((i * ld) + j)

int read_data(float **training_set, std::fstream *file_fstream, int num_lines);

int main() {
  /****************************************************************************
   * LOADING DATA                                                             *
  *****************************************************************************/

  fprintf(stderr, "Loading data...\n");

  std::fstream input_stream;

  // Open files. Data is in data/training_data.csv
  input_stream.open("data/training_data.csv", std::ios::in | std::ios::binary);

  int num_training_points = 10e5;
  int point_size = 3;
  int num_neighbors = 5;
  int num_classes = 10;

  // Read data into std::vector
  fprintf(stderr, "Loading training_data.csv");
  float ** training_set = new float *[num_training_points];
  int num_training_points_read = read_data(training_set, &input_stream, num_training_points);
  fprintf(stderr, "\nLoaded %d points.\n", num_training_points_read);

  // Close the files
  input_stream.close();

  KNN *knn = new KNN(training_set, num_training_points, point_size, num_neighbors, num_classes);

  // Open the test data set
  std::fstream input_stream2;

  input_stream2.open("data/test_data.csv", std::ios::in | std::ios::binary);

  int num_test_points = 1000;
  fprintf(stderr, "Loading test_data.csv");
  float **test_set = new float *[num_test_points];
  int num_test_points_read = read_data(test_set, &input_stream2, num_test_points);
  fprintf(stderr, "\nLoaded %d points.\n", num_test_points_read);

  input_stream2.close();

  /****************************************************************************
   * Running model                                                            *
   ****************************************************************************/

  auto start_time = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "CPU Predicting on test set");
  float *res = knn->predict_many(test_set, num_test_points);
  float hits = 0;
  for (int i = 0; i < num_test_points; i++) {
    if (res[i] != test_set[i][0]) {
      hits++;
    }
  }
  fprintf(stderr, "\n");
  
  std::cerr << "Accuracy: " << hits / num_test_points << std::endl;

  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cerr << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  // Flatten the array for the GPU KNN.
  float *flat_training_set = new float[num_training_points * point_size];
  for (int i = 0; i < num_training_points; i++) {
    for (int j = 0; j < point_size; j++) {
      flat_training_set[IDX2C(i, j, point_size)] = training_set[i][j];
    }
  }

  float *flat_test_set = new float[num_test_points * point_size];
  for (int i = 0; i < num_test_points; i++) {
    for (int j = 0; j < point_size; j++) {
      flat_test_set[IDX2C(i, j, point_size)] = test_set[i][j];
    }
  }

  auto GPUstart_time = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "GPU Predicting on test set");
  float *predictions = gpu_knn_predict_many(200, 512,
    flat_test_set, num_test_points,
    flat_training_set, num_training_points,
    point_size, num_neighbors, num_classes);
  float GPUhits = 0;
  for (int i = 0; i < num_test_points; i++) {
    if (predictions[i] != test_set[i][0]) {
      GPUhits++;
    }
  }
  fprintf(stderr, "\n");

  std::cerr << "Accuracy: " << hits / num_test_points << std::endl;

  auto GPUend_time = std::chrono::high_resolution_clock::now();

  auto GPUduration =
      std::chrono::duration_cast<std::chrono::milliseconds>(GPUend_time - GPUstart_time);

  std::cerr << "Time taken: " << GPUduration.count() << " milliseconds" << std::endl;

  /****************************************************************************
   * Freeing model                                                            *
  *****************************************************************************/

  // Free the dataset
  delete training_set;
  delete flat_training_set;

  delete test_set;
  delete flat_test_set;

  return 0;
}

// Load data from given fstream into vector
int read_data(float **training_set, std::fstream *file_fstream, int num_lines) {
  std::string line;
  int line_num = 0;
  while (getline(*file_fstream, line) && line_num < num_lines) {
    // Print out a dot occasionally so that we know it's actually doing
    // something and not broken
    if (line_num % 100000 == 0)
    {
      fprintf(stderr, ".");
    }

    float *point = new float[3];
    point[0] = atof(line.substr(0, line.find(',')).c_str());
    line = line.substr(line.find(',') + 1);
    point[1] = atof(line.substr(0, line.find(',')).c_str());
    line = line.substr(line.find(',') + 1);
    point[2] = atof(line.substr(0, line.find(',')).c_str());
    line = line.substr(line.find(',') + 1);
    training_set[line_num] = point;

    line_num++;
  }

  return line_num;
}
