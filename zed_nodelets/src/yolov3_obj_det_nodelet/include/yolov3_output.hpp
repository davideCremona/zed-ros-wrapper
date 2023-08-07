//
// Created by Davide Cremona on 26/07/2023.
//

#ifndef YOLOV3_OUTPUT_HPP
#define YOLOV3_OUTPUT_HPP

#include <Eigen/Dense>

using namespace Eigen;

class YoloV3Output {

  public:

    YoloV3Output(const std::vector<int> &gridDims, int nbClass, int nbBoxes);
    void ingestModelOutput(const std::vector<void*> outputBuffers);

  private:

    int nbClass;
    int nbBoxes;
    int nValuesPerBox;

    // vector that contains the number of cells for each yolo tensor
    std::vector<int> gridDims;

    // vector that contains the values of the input buffers formatted as (Nboxes, Nvalues)
    std::vector<MatrixXd> yolos;

    // vector that contains the number of items for each input buffer from the engine
    std::vector<int> nbInputItems;

    // vector that contains the number of boxes for each yolo tensor
    std::vector<int> nbYoloBoxes;

};

#endif // YOLOV3_OUTPUT_HPP