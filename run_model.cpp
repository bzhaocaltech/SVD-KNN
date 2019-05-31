#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <assert.h>

void read_data_into_vector(std::vector<int*>* vec, std::fstream* file_fstream);
void read_vector_into_array(float** arr, std::vector<int*>* vec);

int main() {
  /****************************************************************************
   * LOADING DATA                                                             *
  *****************************************************************************/

  fprintf(stderr, "Loading data...\n");

  std::fstream input_stream_1;
  std::fstream input_stream_2;
  std::fstream input_stream_3;
  std::fstream input_stream_4;

  // Open files. Data is in data/combined_data_*.txt
  input_stream_1.open("data/combined_data_1.txt", std::ios::in | std::ios::binary);
  input_stream_2.open("data/combined_data_2.txt", std::ios::in | std::ios::binary);
  input_stream_3.open("data/combined_data_3.txt", std::ios::in | std::ios::binary);
  input_stream_4.open("data/combined_data_4.txt", std::ios::in | std::ios::binary);

  // Read data into std::vector
  std::vector<int*>* vec = new std::vector<int*>();
  fprintf(stderr, "Loading combined_data_1.txt");
  read_data_into_vector(vec, &input_stream_1);
  fprintf(stderr, "\n");
  fprintf(stderr, "Loading combined_data_2.txt");
  read_data_into_vector(vec, &input_stream_2);
  fprintf(stderr, "\n");
  fprintf(stderr, "Loading combined_data_3.txt");
  read_data_into_vector(vec, &input_stream_3);
  fprintf(stderr, "\n");
  fprintf(stderr, "Loading combined_data_4.txt");
  read_data_into_vector(vec, &input_stream_4);
  fprintf(stderr, "\n");

  // Close the files
  input_stream_1.close();
  input_stream_2.close();
  input_stream_3.close();
  input_stream_4.close();

  // Convert std::vector into array
  int num_points = vec->size();
  float** dataset = new float*[num_points];
  fprintf(stderr, "There are %d points\n", num_points);
  fprintf(stderr, "Converting from vector to array");
  read_vector_into_array(dataset, vec);
  fprintf(stderr, "\n");
  free(vec);

  // Free the dataset
  for (int i = 0; i < num_points; i++) {
    free(dataset[i]);
  }
  free(dataset);
}

// Load data from given fstream into vector
void read_data_into_vector(std::vector<int*>* vec, std::fstream* file_fstream) {
  std::string line;
  int movie_id;
  int line_num = 0;
  while (getline(*file_fstream, line)) {
    // Print out a dot occasionally so that we know it's actually doing
    // something and not broken
    if (line_num % 1000000 == 0) {
      fprintf(stderr, ".");
    }

    // If the last line ends in colon, update the movie id
    if (line.find(":") == line.length() - 1) {
      movie_id = atoi(line.substr(0, line.find(":")).c_str());
    }
    // If the line does not contain a colon, it represents a data point
    else if ((int) line.find(":") == -1) {
      // Get user id
      int user_id = atoi(line.substr(0, line.find(' ')).c_str());
      line = line.substr(line.find(',') + 1);
      // Get rating
      int rating = atoi(line.substr(0, line.find(' ')).c_str());
      line = line.substr(line.find(',') + 1);
      // Discard time for now

      // Format for point is (movie_id, user_id, rating)
      int* point = new int[3];
      point[0] = movie_id;
      point[1] = user_id;
      point[2] = rating;
      vec->push_back(point);
    }
    // Unknown format for line
    else {
      fprintf(stderr, "Parse error oh noessss....\n");
      fprintf(stderr, "Line is: %s\n", line.c_str());
      assert(false);
    }

    line_num++;
  }
}

// Load data from given vector into float**. Also frees the vector
void read_vector_into_array(float** arr, std::vector<int*>* vec) {
  for (unsigned int i = 0; i < vec->size(); i++) {
    // Print out a dot occasionally so that we know it's actually doing
    // something and not broken
    if (i % 1000000 == 0) {
      fprintf(stderr, ".");
    }

    arr[i] = new float[3];
    arr[i][0] = vec->at(i)[0];
    arr[i][1] = vec->at(i)[1];
    arr[i][2] = vec->at(i)[2];
    free(vec->at(i));
  }
}
