//
// Created by Davide Cremona on 24/07/2023.
//

#include "eigen_to_string.hpp"


std::string vectorToString(const VectorXd& vec) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}


std::string matrixToString(const MatrixXd& mat) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < mat.rows(); ++i) {
        oss << "[";
        for (int j = 0; j < mat.cols(); ++j) {
            oss << mat(i, j);
            if (j < mat.cols() - 1) {
                oss << ", ";
            }
        }
        oss << "]";
        if (i < mat.rows() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}
