#include "svd.hpp"
#include <assert.h>
#include <stdio.h>

/* Constructor for SVD */
SVD::SVD(int latent_factors, float eta, float reg, int len_x, int len_y) {
  this->latent_factors = latent_factors;
  this->eta = eta;
  this->reg = reg;
  this->len_x = len_x;
  this->len_y = len_y;

  this->U = (float*) calloc(latent_factors * len_x, sizeof(float));
  this->V = (float*) calloc(latent_factors * len_y, sizeof(float));

  // Initialize U and V to 0.5 arbitrarily
  for (int i = 0; i < latent_factors * len_x; i++) {
    this->U[i] = 0.5;
  }
  for (int i = 0; i < latent_factors * len_y; i++) {
    this->V[i] = 0.5;
  }

  this->a = (float*) calloc(len_x, sizeof(float));
  this->b = (float*) calloc(len_y, sizeof(float));
  this->mu = 0;
}

// Access element of U
float SVD::get_u_val(int row, int col) {
  // Out of bounds
  if (row > len_x || col > latent_factors) {
    assert(false);
  }

  return U[row * latent_factors + col];
}

// Access element of V
float SVD::get_v_val(int row, int col) {
  // Out of bounds
  if (row > len_y || col > latent_factors) {
    assert(false);
  }

  return V[row * latent_factors + col];
}

// Set element of U
void SVD::set_u_val(int row, int col, float val) {
  // Out of bounds
  if (row > len_x || col > latent_factors) {
    assert(false);
  }

  U[row * latent_factors + col] = val;
}

// Set element of V
void SVD::set_v_val(int row, int col, float val) {
  // Out of bounds
  if (row > len_y || col > latent_factors) {
    assert(false);
  }

  V[row * latent_factors + col] = val;
}

float SVD::predict_one(int x, int y) {
  // Calculate the predicted value
  float predicted = 0;
  for (int j = 0; j < latent_factors; j++) {
    predicted += get_u_val(x, j) * get_v_val(y, j);
  }

  // Add biases
  predicted += a[x];
  predicted += b[y];
  predicted += mu;

  return predicted;
}

void SVD::train(float* train, int size, int num_epochs,
  float* valid, int valid_size) {
  for (int epoch_num = 0; epoch_num < num_epochs; epoch_num++) {
    fprintf(stderr, "Running epoch %d", epoch_num);
    // MSE error
    float total_error = 0;
    for (int i = 0; i < size * 3; i += 3) {
      if (i % 1000000 == 0) {
        fprintf(stderr, ".");
      }

      int x = train[i];
      int y = train[i + 1];
      float actual = train[i + 2];
      float predicted = predict_one(x, y);
      float error = predicted - actual;
      total_error += error * error;

      // Adjust a
      a[x] -= eta * (error + reg * a[x]);
      // Adjust b
      b[y] -= eta * (error + reg * b[y]);
      // Adjust mu
      mu -= eta * (error + reg * mu);
      // Adjust U and V
      for (int j = 0; j < latent_factors; j++) {
        float u_grad = eta * (error * get_v_val(y, j) + reg * get_u_val(x, j));
        float v_grad = eta * (error * get_u_val(x, j) + reg * get_v_val(y, j));

        set_u_val(x, j, get_u_val(x, j) - u_grad);
        set_v_val(y, j, get_v_val(y, j) - v_grad);
      }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "Training error for epoch: %f\n", total_error / (float) size);

    // Error for validation set
    if (valid != NULL) {
      float valid_error = 0;
      for (int i = 0; i < valid_size * 3; i += 3) {
        int x = valid[i];
        int y = valid[i + 1];
        float actual = valid[i + 2];
        float predicted = predict_one(x, y);
        valid_error += (predicted - actual) * (predicted - actual);
      }
      fprintf(stderr, "Validation error for epoch: %f\n", valid_error / (float) valid_size);
    }
  }
}

float* SVD::predict(float* test, int size) {
  float* predictions = new float[size];
  for (int i = 0; i < size * 2; i += 2) {
    predictions[i / 2] = predict_one(test[i], test[i + 1]);
  }
  return predictions;
}

SVD::~SVD() {
  free(U);
  free(V);
  free(a);
  free(b);
}
