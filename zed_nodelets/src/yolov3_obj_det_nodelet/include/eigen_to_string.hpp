//
// Created by Davide Cremona on 24/07/2023.
//

#ifndef EIGEN_TO_STRING_HPP
#define EIGEN_TO_STRING_HPP

#include <Eigen/Dense>

using namespace Eigen;

std::string vectorToString(const VectorXd& vec);
std::string matrixToString(const MatrixXd& mat);

#endif  // EIGEN_TO_STRING_HPP