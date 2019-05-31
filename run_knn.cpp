#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <iostream>
#include "knn/knn.hpp"

void read_data(float **training_set, std::fstream *file_fstream);

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
  read_data(training_set, &input_stream);
  fprintf(stderr, "\n");

  // Close the files
  input_stream.close();

  KNN *knn = new KNN(training_set, num_training_points, point_size, num_neighbors, num_classes);

  // Open the test data set
  std::fstream input_stream2;

  input_stream2.open("data/test_data.csv", std::ios::in | std::ios::binary);

  int num_test_points = 1000;
  fprintf(stderr, "Loading test_data.csv");
  float **test_set = new float *[num_test_points];
  read_data(test_set, &input_stream2);
  fprintf(stderr, "\n");

  input_stream2.close();

  /****************************************************************************
   * Running model                                                            *
  *****************************************************************************/

  auto start_time = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "Predicting on test set");
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

  /****************************************************************************
   * Freeing model                                                            *
  *****************************************************************************/

  // Free the dataset
  for (int i = 0; i < num_training_points; i++) {
    free(training_set[i]);
  }

  free(training_set);

  for (int i = 0; i < num_test_points; i++) {
    free(test_set[i]);
  }

  free(test_set);

  return 0;
}

// Load data from given fstream into vector
void read_data(float **training_set, std::fstream *file_fstream) {
  std::string line;
  int line_num = 0;
  while (getline(*file_fstream, line))
  {
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
}
