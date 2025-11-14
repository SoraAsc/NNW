#include "api.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

int main(void){
    size_t in = 3, out = 0, n_samples = 4, n_dataset_samples = 7;
    float max_value = 213.0f; // to normalize the dataset

    NN_Model* m = nn_create_model(in);
    nn_add_dense(m, 8, NN_ACT_RELU);
    nn_add_dense(m, 1, NN_ACT_LINEAR);
    out = nn_get_output_dim(m);

    NN_TrainerConfig cfg;
    cfg.epochs = 3000;
    cfg.batch_size = 16;
    cfg.shuffle = 1;
    cfg.learning_rate = 0.01f;

    NN_Trainer* t = nn_create_trainer(m, NN_OPT_ADAMW, NN_LOSS_MSE, &cfg);

    // Dataset: 7, in=3, out=1
    std::vector<float> X = {
        1,2,3,   2,3,4,   3,4,5,
        4,5,6,   5,6,7,   31,32,33,
        207,213,209
    };
    std::vector<float> Y = { 1, 2, 3, 4, 5, 31, 207 };

    // Normalize
    for (auto& v : X) v /= max_value;
    for (auto& v : Y) v /= max_value;

    nn_train_fit(t, X.data(), n_dataset_samples, in, Y.data(), out);

    std::vector<float> x1 = {
        10,11,12,
        4,5,6,
        20,21,22,
        122,121,123
    };

    for (auto& v : x1) v /= max_value;

    std::vector<float> y1(out * n_samples);

    nn_predict(m, x1.data(), n_samples, in, y1.data(), out);

    for (size_t i = 0; i < n_samples; ++i)
      std::cout << "Predict(" << i << "): " << y1[i] * max_value << "\n";
    

    nn_free_trainer(t);
    nn_free_model(m);
    return 0;
}