//
// Created by Davide Cremona on 26/07/2023.
//

YoloV3Output::YoloV3Output(const std::vector<int> &gridDims, int nbClass, int nbBoxes) {
  this->gridDims = gridDims;
  this->nbClass = nbClass;
  this->nbBoxes = nbBoxes;

  this->nValuesPerBox = 5 + nbClass;

  for (int i=0; i<gridDims.size(); i++) {
    int totalBoxes = gridDims[i] * gridDims[i] * nbBoxes;
    nbYoloBoxes.push_back(nbYoloBoxes);
    
    int totalInputItems = totalBoxes * this->nValuesPerBox;
    nbInputItems.push_back(totalInputItems);

    MatrixXd yoloTensor(totalBoxes, this->nValuesPerBox);
    yolos.push_back(yoloTensor);
  }
}