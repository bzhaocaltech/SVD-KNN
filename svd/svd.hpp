#ifndef SVD_HPP
#define SVD_HPP

#include <stdlib.h>

class SVD {
  private:
    // Number of latent factors.
    int latent_factors;

    // The size of the leading dimension of the matrix to factorize
    int len_x;
    // The length of the other dimension of the matrix to factorize
    int len_y;

    // The matrices that the sparse training data is factorize into
    // U is size of len_x * latent_factors
    // V is size of len_y * latent_factors
    // len_x and len_y are the leading dimensions of the matrices
    // Ideally, U * V^T = sparse training matrix (neglecting biases)
    float* U;
    float* V;

    // The biases.
    // a is an array of length len_x
    // b is an array of length len_y
    float* a;
    float* b;

    // Global bias
    float mu;

    // Learning rate
    float eta;
    // Regularization factor
    float reg;

    // Access elements of U and V
    float get_u_val(int row, int col);
    float get_v_val(int row, int col);

    // Set elements of U and V
    void set_u_val(int row, int col, float val);
    void set_v_val(int row, int col, float val);

    // Predict a single point
    float predict_one(int x, int y);

  public:
    /* Constructor for SVD */
    SVD(int latent_factors, float eta, float reg, int len_x, int len_y);

    /* Train on a dataset. Each element of the input array has 3 elements.
     * (x, y, val). So the length of the training data is 3 * num_elements.
     * Note that x and y are technically ints. Includes optional
     * argument for validation set */
    void train(float* train, int size, int num_epochs,
      float* valid = NULL, int valid_size = 0);

    /* Predicts the values of a dataset. Each element of the input array has
     * 2 elements (x, y). Returns the predicted values in the order that they
     * were inserted. Note that x and y are technically ints */
    float* predict(float** test, int size);

    ~SVD();
};

#endif
