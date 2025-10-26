#include "api.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int main(void){
  size_t in = 3, out, n_samples = 4, n_dataset_samples = 7;
  float max_value = 213.0f; // to normalize the dataset
  NN_Model* m = nn_create_model(in);
  nn_add_dense(m, 8, NN_ACT_RELU);
  nn_add_dense(m, 1, NN_ACT_LINEAR);
  out = nn_get_output_dim(m);

  TrainerConfig cfg = { .epochs = 3000, .batch_size = 16, .shuffle = true };
  NN_Trainer* t = nn_create_trainer(m, NN_OPT_ADAMW, NN_LOSS_MSE, &cfg);

  // Dataset: 6, in=3, out=1
  float X[n_dataset_samples*in] = { 1,2,3,  2,3,4,  3,4,5,  4,5,6,  5,6,7,  31,32,33,   207, 213, 209 };
  float Y[n_dataset_samples*out] = { 1,      2,      3,      4,      5,     31,     207      };

  for (int i = 0; i < n_dataset_samples*in; ++i) X[i] /= max_value;
  for (int i = 0; i < n_dataset_samples*out; ++i) Y[i] /= max_value;


  nn_train_fit(t, X, n_dataset_samples, in, Y, out);

  float x1[in*n_samples] = { 10,11,12,   4,5,6,    20,21,22,    122, 121, 123 };
  for (int i = 0; i < n_dataset_samples*in; ++i) x1[i] /= max_value;
  float y1[out*n_samples];
  nn_predict(m, x1, n_samples, in, y1, out);
  for (size_t i = 0; i < n_samples; ++i) std::cout << "Predict(" << i << "): " << y1[i] * max_value << "\n";
  
  nn_free_trainer(t);
  nn_free_model(m);
  return 0;
}