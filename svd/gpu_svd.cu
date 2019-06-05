#include <cuda_runtime.h>
#include <stdio.h>

#include "gpu_svd.cuh"
#include "../helper_cuda.h"
#include <cuda_runtime.h>

GPU_SVD* createGPUSVD(int latent_factors, float eta, float reg, int len_x,
  int len_y) {
  GPU_SVD* svd = (GPU_SVD*) calloc(1, sizeof(GPU_SVD));

  svd->latent_factors = latent_factors;
  svd->eta = eta;
  svd->reg = reg;
  svd->len_x = len_x;
  svd->len_y = len_y;

  // Initialize U and V
  CUDA_CALL(cudaMalloc(&svd->U, latent_factors * len_x * sizeof(float)));
  CUDA_CALL(cudaMalloc(&svd->V, latent_factors * len_y * sizeof(float)));
  float* temp_u = (float*) calloc(latent_factors * len_x, sizeof(float));
  for (int i = 0; i < latent_factors * len_x; i++) {
    temp_u[i] = 0.5;
  }
  float* temp_v = (float*) calloc(latent_factors * len_y, sizeof(float));
  for (int i = 0; i < latent_factors * len_y; i++) {
    temp_v[i] = 0.5;
  }
  CUDA_CALL(cudaMemcpy(svd->U, temp_u, latent_factors * len_x * sizeof(float),
    cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(svd->V, temp_v, latent_factors * len_y * sizeof(float),
    cudaMemcpyHostToDevice));

  // Initialize biases
  CUDA_CALL(cudaMalloc(&svd->a, len_x * sizeof(float)));
  CUDA_CALL(cudaMemset(svd->a, 0, len_x * sizeof(float)));
  CUDA_CALL(cudaMalloc(&svd->b, len_y * sizeof(float)));
  CUDA_CALL(cudaMemset(svd->b, 0, len_y * sizeof(float)));
  CUDA_CALL(cudaMalloc(&svd->mu, sizeof(float)));
  CUDA_CALL(cudaMemset(svd->mu, 0, 1 * sizeof(float)));

  free(temp_u);
  free(temp_v);

  return svd;
}

void freeGPUSVD(GPU_SVD* svd) {
  CUDA_CALL(cudaFree(svd->U));
  CUDA_CALL(cudaFree(svd->V));
  CUDA_CALL(cudaFree(svd->a));
  CUDA_CALL(cudaFree(svd->b));
  CUDA_CALL(cudaFree(svd->mu));

  free(svd);
}

__global__ void SVDTrainKernel(int latent_factors, float* U, float* V, float* a,
  float* b, float* mu, float eta, float reg, const float* train, int size,
  const float* valid, int valid_size, float* train_epoch_error,
  float* valid_epoch_error) {

  // Initialize shared memory
  // Shared memory contains 3 subarrays: (0 - blockDim.x is used for dot
  // product calculation; blockDim.x - (blockDim.x + latent_factors) is used
  // for copying U; (blockDim.x + latent_factors) + (blockDim.x +
  // latent_factors * 2) is used for copying V)
  extern __shared__ float shared[];

  // Total error for the epoch in just this block
  float total_error = 0;

  // Offsets in shared memory
  uint offsetUShared = blockDim.x;
  uint offsetVShared = blockDim.x + latent_factors;

  // Each individual block deals with a single training example. Train_index is
  // the current index of train being processed by this block.
  uint train_index = blockIdx.x * 3;
  while (train_index < size * 3) {
    // Extract out necessary variables from the training data
    int x = train[train_index];
    int y = train[train_index + 1];
    float actual = train[train_index + 2];
    // Start of the latent_factor row for U and V
    int u_row = x * latent_factors;
    int v_row = y * latent_factors;

    // Now get the predicted value at x, y. To do this, load up shared memory
    // with part of the dot product. We can also use the loop to set up the
    // parts of the shared memory which will just be copies of U and V
    uint row_index = threadIdx.x;
    float dot_part = 0;
    while (row_index < latent_factors) {
      float uval = U[u_row + row_index];
      float vval = V[v_row + row_index];
      shared[offsetUShared + row_index] = uval;
      shared[offsetVShared + row_index] = vval;
      dot_part += uval * vval;
      row_index += blockDim.x;
    }
    shared[threadIdx.x] = dot_part;

    // After everything is loaded into shared memory, sync the threads
    __syncthreads();

    // Now do a sum reduction to find the final dot product
    for (int i = blockDim.x / 2; i >= 1; i /= 2) {
      if (threadIdx.x < i) {
        float val_1 = shared[threadIdx.x];
        float val_2 = shared[threadIdx.x + i];
        float sum = val_1 + val_2;
        shared[threadIdx.x] = sum;
      }
      // Sync the threads after each iteration
      __syncthreads();
    }

    // We can now add the biases to find the predicted value given by our
    // algorithm
    float ax = a[x];
    float by = b[y];
    float m = *mu;
    float predicted = shared[0] + ax + by + m;

    float error = predicted - actual;
    total_error += (error * error);

    // Now comes the "hogwild" part. This results in race conditions between
    // blocks but is ok

    // Each thread will be responsible for updating a segment of U and V.
    // NOTE: This is done in coalesced manner
    row_index = threadIdx.x;
    while (row_index < latent_factors) {
      float old_u = shared[offsetUShared + row_index];
      float old_v = shared[offsetVShared + row_index];
      float u_error_grad = eta * error * old_v;
      float v_error_grad = eta * error * old_u;
      float u_reg_grad = eta * reg * old_u;
      float v_reg_grad = eta * reg * old_v;
      U[u_row + row_index] = old_u - u_error_grad - u_reg_grad;
      V[v_row + row_index] = old_v - v_error_grad - v_reg_grad;
      row_index += blockDim.x;
    }

    // Only one thread need adjust a, b and mu.
    if (threadIdx.x == 0) {
      a[x] = ax - eta * (error + reg * ax);
      b[y] = by - eta * (error + reg * by);
      *mu = m - eta * (error + reg * m);
    }

    // Move onto to the next training sample
    train_index += 3 * gridDim.x;

    // The next loop will change shared memory, so make sure all threads are
    // done before moving on
    __syncthreads();
  }

  // Have only a single thread per block report the error in the block
  if (threadIdx.x == 0) {
    atomicAdd(train_epoch_error, total_error);
  }

  // Now go find the validation error for the epoch
  float valid_error = 0;
  uint valid_index = blockIdx.x * 3;
  while (valid_index < valid_size * 3) {
    // Extract out necessary variables from the validation data
    int x = valid[valid_index];
    int y = valid[valid_index + 1];
    float actual = valid[valid_index + 2];
    // Start of the latent_factor row for U and V
    int u_row = x * latent_factors;
    int v_row = y * latent_factors;

    // Now get the predicted value at x, y. To do this, load up shared memory
    // with part of the dot product. No need to load U and V into shared memory
    // since we are only going to use them once in this loop
    uint row_index = threadIdx.x;
    float dot_part = 0;
    while (row_index < latent_factors) {
      float uval = U[u_row + row_index];
      float vval = V[v_row + row_index];
      dot_part += uval * vval;
      row_index += blockDim.x;
    }
    shared[threadIdx.x] = dot_part;

    // After everything is loaded into shared memory, sync the threads
    __syncthreads();

    // Now do a sum reduction to find the final dot product
    for (int i = blockDim.x / 2; i >= 1; i /= 2) {
      if (threadIdx.x < i) {
        float val_1 = shared[threadIdx.x];
        float val_2 = shared[threadIdx.x + i];
        float sum = val_1 + val_2;
        shared[threadIdx.x] = sum;
      }
      // Sync the threads after each iteration
      __syncthreads();
    }

    // We can now add the biases to find the predicted value given by our
    // algorithm
    float ax = a[x];
    float by = b[y];
    float m = *mu;
    float predicted = shared[0] + ax + by + m;

    float error = predicted - actual;
    valid_error += (error * error);
    // Move onto to the next training sample
    valid_index += 3 * gridDim.x;

    // The next loop will change shared memory, so make sure all threads are
    // done before moving on
    __syncthreads();
  }

  // Have only a single thread per block report the error in the block
  if (threadIdx.x == 0) {
    atomicAdd(valid_epoch_error, valid_error);
  }
}

void callSVDTrainKernel(unsigned int blocks, unsigned int threadsPerBlock,
  GPU_SVD* svd, const float* train, int size, int num_epochs,
  const float* valid, int valid_size) {

  // Use to keep track of the training error per epoch
  float* train_epoch_error;
  CUDA_CALL(cudaMalloc(&train_epoch_error, sizeof(float)));

  // Use to keep track of the validation error per epoch
  float* valid_epoch_error;
  CUDA_CALL(cudaMalloc(&valid_epoch_error, sizeof(float)));

  for (int epoch_num = 0; epoch_num < num_epochs; epoch_num++) {
    fprintf(stderr, "Running epoch %d\n", epoch_num);

    // Reset the errors
    CUDA_CALL(cudaMemset(train_epoch_error, 0, sizeof(float)));
    CUDA_CALL(cudaMemset(valid_epoch_error, 0, sizeof(float)));

    SVDTrainKernel<<<blocks, threadsPerBlock, (threadsPerBlock + 2 * svd->latent_factors) * sizeof(float)>>>
      (svd->latent_factors, svd->U, svd->V, svd->a, svd->b, svd->mu, svd->eta,
        svd->reg, train, size, valid, valid_size, train_epoch_error, valid_epoch_error);

    cudaError err = cudaGetLastError();
    if  (cudaSuccess != err){
      fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    }
    else {
      fprintf(stderr, "No kernel error detected\n");
    }

    // Wait for the kernel to complete
    CUDA_CALL(cudaDeviceSynchronize());

    // Copy over the errors so we can print them
    float* curr_train_error = (float*) malloc(sizeof(float));
    CUDA_CALL(cudaMemcpy(curr_train_error, train_epoch_error, sizeof(float), cudaMemcpyDeviceToHost));
    float* curr_valid_error = (float*) malloc(sizeof(float));
    CUDA_CALL(cudaMemcpy(curr_valid_error, valid_epoch_error, sizeof(float), cudaMemcpyDeviceToHost));

    fprintf(stderr, "Training error in epoch: %f\n", *curr_train_error / size);
    fprintf(stderr, "Validation error in epoch: %f\n", *curr_valid_error / valid_size);

    free(curr_train_error);
    free(curr_valid_error);
  }

  CUDA_CALL(cudaFree(train_epoch_error));
  CUDA_CALL(cudaFree(valid_epoch_error));
}

__global__ void SVDPredictKernel(int latent_factors, float* U, float* V, float* a,
  float* b, float* mu, const float* test, unsigned int size,
  float* predictions) {
    // Used for dot product calculations
    extern __shared__ float shared[];

    uint test_index = blockIdx.x * 2;
    while (test_index < size * 2) {
      // Extract out necessary variables from the test data
      int x = test[test_index];
      int y = test[test_index + 1];
      // Start of the latent_factor row for U and V
      int u_row = x * latent_factors;
      int v_row = y * latent_factors;

      // Now get the predicted value at x, y. To do this, load up shared memory
      // with part of the dot product.
      uint row_index = threadIdx.x;
      float dot_part = 0;
      while (row_index < latent_factors) {
        float uval = U[u_row + row_index];
        float vval = V[v_row + row_index];
        dot_part += uval * vval;
      row_index += blockDim.x;
    }
    shared[threadIdx.x] = dot_part;

    // After everything is loaded into shared memory, sync the threads
    __syncthreads();

    // Now do a sum reduction to find the final dot product
    for (int i = blockDim.x / 2; i >= 1; i /= 2) {
      if (threadIdx.x < i) {
        float val_1 = shared[threadIdx.x];
        float val_2 = shared[threadIdx.x + i];
        float sum = val_1 + val_2;
        shared[threadIdx.x] = sum;
      }
      // Sync the threads after each iteration
      __syncthreads();
    }

    // We can now add the biases to find the predicted value given by our
    // algorithm
    float ax = a[x];
    float by = b[y];
    float m = *mu;
    float predicted = shared[0] + ax + by + m;

    // Only the first thread needs to add the value to predictions
    if (threadIdx.x == 0) {
      predictions[test_index / 2] = predicted;
    }

    // Move onto the next training sample
    test_index += 2 * gridDim.x;

    // The next loop will change shared memory, so make sure all threads are
    // done before moving on
    __syncthreads();
  }
}

void callSVDPredictKernel(unsigned int blocks, unsigned int threadsPerBlock,
  const GPU_SVD* svd, const float* test, unsigned int size,
  float* predictions) {

  SVDPredictKernel<<<blocks, threadsPerBlock, threadsPerBlock>>>
    (svd->latent_factors, svd->U, svd->V, svd->a, svd->b, svd->mu, test, size,
      predictions);

  cudaError err = cudaGetLastError();
  if (cudaSuccess != err){
    fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
  }
  else {
    fprintf(stderr, "No kernel error detected\n");
  }

  // Wait for the kernel to complete
  CUDA_CALL(cudaDeviceSynchronize());
}
