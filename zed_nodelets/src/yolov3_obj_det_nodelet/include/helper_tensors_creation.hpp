//
// Created by Davide Cremona on 24/07/2023.
//

#ifndef HELPER_TENSORS_CREATION_HPP
#define HELPER_TENSORS_CREATION_HPP

#include <vector>
#include <utility>

#include <Eigen/Dense>

using namespace Eigen;

// Optimized sigmoid function for vectorized operations using Eigen library
MatrixXd _sigmoid(const MatrixXd& x);

// function ported from Python classes
std::pair<MatrixXd, VectorXd> createGridTensors(const std::vector<int>& grid_dims, int num_boxes);
MatrixXd createAnchorsTensor(const std::vector<int>& grid_dims, 
                             const std::vector<int>& anchor_boxes, int num_boxes);

#endif // HELPER_TENSORS_CREATION_HPP