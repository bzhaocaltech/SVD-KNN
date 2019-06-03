#ifndef CUDA_GPU_SVD
#define CUDA_GPU_SVD

typedef struct {
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
  float* mu;

  // Learning rate
  float eta;
  // Regularization factor
  float reg;
} GPU_SVD;

/* Returns a GPU_SVD struct. The GPU_SVD* itself is allocated on the host but
 * its members are allocated on device (so that it can be easily passed to the
 * kernel) */
GPU_SVD* createGPUSVD(int latent_factors, float eta, float reg, int len_x,
  int len_y);

/* Frees a GPU_SVD struct created by createGPUSVD */
void freeGPUSVD(GPU_SVD* svd);

/* Kernel used to train the svd */
__global__ void SVDTrainKernel(GPU_SVD* svd, float* train, int size,
  int num_epochs, float* valid, int valid_size);

/* Calls the kernel to train the svd */
void callSVDTrainKernel(unsigned int blocks, unsigned int threadsPerBlock,
  GPU_SVD* svd, const float* train, int size, int num_epochs,
  const float* valid = NULL, int valid_size = 0);

#endif
