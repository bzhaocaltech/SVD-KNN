/**
 * Class for finding K-nearest-neighbors.
 */

struct PointDistance {
  int index;
  float *point;
  float distance;
};

class KNN {
public:
  /**
   * Constructor for KNN.
   */
  KNN(
    float **training_set,
    int num_points,
    int point_size,
    int num_neighbors,
    int num_classes
  );
  float *predict_many(float **data, int num_points);
  /**
   * Takes a data point, finds the k-nearest-neighbors, then returns the most
   * likely classification.
   */
  float predict_one(float *data);
private:
  /**
   * The data that this KNN model trains on.
   */
  float **training_set;
  /**
   * The number of data points in the training set.
   */
  int num_points;
  /**
   * The size of each data point.
   */
  int point_size;
  /**
   * The number of neighbors to look for, in kNN terms, this is `k`.
   */
  int num_neighbors;
  /**
   * The number of unique classes to predict a data point will fall under.
   */
  int num_classes;
  /**
   * Compute the distance between two points. Used to find the distances
   * between a center and its neighbors.
   */
  float distance(float *x, float *y);
  /**
   * Find the nearest neighbors from the training set of a data point.
   */
  PointDistance* find_nearest_neighbors(float *data);

  /**
   * Helper function that utilizes an array of point distances as a priority
   * queue.
   */
  void insert(PointDistance *arr, PointDistance pd);
};