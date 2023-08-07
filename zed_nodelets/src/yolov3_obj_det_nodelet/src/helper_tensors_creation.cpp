//
// Created by Davide Cremona on 24/07/2023.
//

#include "helper_tensors_creation.hpp"

#include <vector>
#include <utility>

#include <Eigen/Dense>

using namespace Eigen;

// Optimized sigmoid function for vectorized operations using Eigen library
MatrixXd _sigmoid(const MatrixXd& x) {
  return 1.0 / (1.0 + (-x.array()).exp());
}


std::pair<MatrixXd, VectorXd> createGridTensors(const std::vector<int>& grid_dims, int num_boxes) {
    std::vector<MatrixXd> grid_idcs_tensor_vec;
    std::vector<VectorXd> grid_dims_tensor_vec;

    for (const auto& grid_dim : grid_dims) {
        VectorXd rows_idcs = VectorXd::LinSpaced(grid_dim, 0, grid_dim - 1);
        VectorXd cols_idcs = VectorXd::LinSpaced(grid_dim, 0, grid_dim - 1);
        MatrixXd grid_idcs(grid_dim * grid_dim * num_boxes, 2);
        int k = 0;
        for (int i = 0; i < grid_dim; ++i) {
          for (int j = 0; j < grid_dim; ++j) {
            for (int t = 0; t < num_boxes; t++) {
              grid_idcs(k, 0) = rows_idcs(i);
              grid_idcs(k, 1) = cols_idcs(j);
              ++k;
            }
          }
        }
        grid_idcs_tensor_vec.push_back(grid_idcs);

        VectorXd grid_dims_vec = VectorXd::Constant(grid_idcs.rows(), grid_dim);
        grid_dims_tensor_vec.push_back(grid_dims_vec);
    }

    //instantiate result grid_idcs_tensor
    int nb_grid_idcs_tensor_elements = 0;
    for (const auto& m: grid_idcs_tensor_vec) {
      nb_grid_idcs_tensor_elements += m.rows();
    }
    MatrixXd grid_idcs_tensor(nb_grid_idcs_tensor_elements, 2);

    // concatenate grid_idcs matrices
    int row_index = 0;
    for (const auto& m : grid_idcs_tensor_vec) {
        grid_idcs_tensor.block(row_index, 0, m.rows(), 2) = m;
        row_index += m.rows();
    }

    //instantiate result grid_dims_tensor
    int nb_grid_dims_tensor_elements = 0;
    for (const auto& v: grid_dims_tensor_vec) {
      nb_grid_dims_tensor_elements += v.size();
    }
    VectorXd grid_dims_tensor(nb_grid_dims_tensor_elements);

    // concatenate grid_dims vectors
    int idx = 0;
    for (const auto& v : grid_dims_tensor_vec) {
        grid_dims_tensor.segment(idx, v.size()) = v;
        idx += v.size();
    }

    // Save C++ output to a file
    // std::ofstream output_file("/root/test_run/cpp_output.txt");
    // std::string grid_dims_tensor_s = vectorToString(grid_dims_tensor);
    // std::string grid_idcs_tensor_s = matrixToString(grid_idcs_tensor);
    // if (output_file.is_open()) {
    //   output_file << grid_idcs_tensor_s << "\n";
    //   output_file << grid_dims_tensor_s << "\n";
    //   std::cout << "C++ output saved to cpp_output.txt\n";
    // } else {
    //   std::cerr << "Error: Unable to open the output file!\n";
    // }

    // output_file.close();
    
    return std::make_pair(grid_idcs_tensor, grid_dims_tensor);
}

MatrixXd createAnchorsTensor(const std::vector<int>& grid_dims, 
                             const std::vector<int>& anchor_boxes, int num_boxes) {
  int num_elements_per_output = num_boxes * 2; // 2 is the number of elements per box
  int num_outputs = anchor_boxes.size() / (num_elements_per_output);  

  // create the "anchors_eigen" matrix:
  // matrix containing the anchor boxes for each output tensor
  // first tensor is associated to bigger boxes
  // last tensor is associated to smaller boxes 
  MatrixXd anchors_eigen(num_outputs, num_elements_per_output);
  for (int i=num_outputs-1; i>=0; i--) {
    for (int j=0; j<num_elements_per_output; j++) {
      anchors_eigen(num_outputs-1 - i, j) = anchor_boxes[i * num_elements_per_output + j];
    }
  }

  // compute total number of boxes inside the final anchors tensor
  // for each output tensor we compute the total number of boxes that is:
  // N_BOXES for each grid cell --> N_BOXES * num_grid_cells
  int total_elems = 0;
  for (int i=0; i<grid_dims.size(); i++) {
    total_elems += 3 * grid_dims[i] * grid_dims[i];
  }

  // now build the anchors tensor
  MatrixXd anchors_tensor(total_elems, 2);
  int curr_idx = 0;
  
  // for each output tensor
  for (int i=0; i<num_outputs; i++) {
    
    // for each grid cell
    int num_cells_grid = grid_dims[i] * grid_dims[i];
    for (int j=0; j<num_cells_grid; j++) {
      
      // copy boxes widths and heights for that output tensor
      // to the correct position in the anchors tensor
      for (int k=0; k<num_boxes*2; k+=2) {
        anchors_tensor(curr_idx, 0) = anchors_eigen(i, k);
        anchors_tensor(curr_idx, 1) = anchors_eigen(i, k+1);
        curr_idx += 1;
      } 
    }
  }

  return anchors_tensor;
}

std::vector<MatrixXd> createYolosTensor(const std::vector<void*> &outputBuffers, 
                                        int nb_class, int batch_size) {



}