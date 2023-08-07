//
// Created by Davide Cremona on 20/07/2023.
//

#include "yolov3_obj_det.hpp"
#include "eigen_to_string.hpp"
#include "helper_tensors_creation.hpp"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv4/opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <fstream>
#include <sstream>

#ifndef NDEBUG
#include <ros/console.h>
#endif


namespace zed_nodelets
{
  Yolov3ObjDetNodelet::~Yolov3ObjDetNodelet() {
    // Release the allocated memory
    cudaFree(inputBuffer);

    for (int i=0; i<outputBuffers.size(); i++){
      cudaFree(outputBuffers[i]);
    }

    // Clean up resources
    // cudaFree(d_input);
    // cudaFree(d_output);
    trtContext->destroy();
    trtEngine->destroy();
    trtRuntime->destroy();
  }

  void Yolov3ObjDetNodelet::onInit() {
    // Node handlers
    mNh = getNodeHandle();
    mNhP = getPrivateNodeHandle();
    it_ = std::make_shared<image_transport::ImageTransport>(mNh);

    NODELET_INFO("********** Starting nodelet '%s' **********", getName().c_str());

    image_sub_ = it_->subscribe("/zed2/zed_node/left/image_rect_color", 1, &Yolov3ObjDetNodelet::imgCallback, this);
    NODELET_INFO_STREAM(" * Subscribed to topic: " << image_sub_.getTopic().c_str());

    // -------------- CONFIGURATION READ ----------------
    // TODO: to be read from config file / parameters
    enginePath = "/home/DATI/insulators/YOLOv3/models/YOLO_06_noSyn/bs1_608x608_fp16_engine.trt";
    numClasses = 1;
    gridDims = {19, 38, 76};
    anchorBoxes = {12, 97, 40, 391, 44, 116, 63, 31, 114, 450, 121, 206, 223, 74, 253, 389, 342, 156};

    // -------------- STATIC TENSORS HELPERS ----------------
    auto res = createGridTensors(gridDims, NUM_BOXES);
    gridIdcsTensor = res.first;
    gridDimsTensor = res.second;

    anchorsTensor = createAnchorsTensor(gridDims, anchorBoxes, NUM_BOXES);

    // -------------- TRT INITIALIZATION ----------------
    // create inference runtime
    trtRuntime = nvinfer1::createInferRuntime(gLogger);

    // Read the TensorRT engine file
    const std::string engineData = readEngineFile();

    // Deserialize the TensorRT engine
    trtEngine = trtRuntime->deserializeCudaEngine(engineData.data(), engineData.size());

    if (!trtEngine) {
        NODELET_ERROR_STREAM("Error deserializing the engine.");
        exit(EXIT_FAILURE);
    }
    else {
      NODELET_INFO_STREAM("Correctly Loaded engine: " << enginePath);
    }

    // Read input and output shapes from the TensorRT engine
    const int numBindings = trtEngine->getNbBindings();

    for (int i = 0; i < numBindings; ++i) {
      nvinfer1::Dims dims = trtEngine->getBindingDimensions(i);

      nvinfer1::DataType dataType = trtEngine->getBindingDataType(i);
      std::string dataTypeStr;
      switch (dataType) {
        case nvinfer1::DataType::kFLOAT: dataTypeStr = "kFLOAT"; break;
        case nvinfer1::DataType::kHALF: dataTypeStr = "kHALF"; break;
        case nvinfer1::DataType::kINT8: dataTypeStr = "kINT8"; break;
        case nvinfer1::DataType::kINT32: dataTypeStr = "kINT32"; break;
        default: dataTypeStr = "UNKNOWN"; break;
      }

      if (trtEngine->bindingIsInput(i)) {
        NODELET_INFO_STREAM("Model Binding " << i << " type: INPUT");
        
        // get model input tensor shape
        batchSize = dims.d[0];
        inputH = dims.d[1];
        inputW = dims.d[2];
        inputC = dims.d[3];

        // compute input buffer size
        inputSize = batchSize * inputH * inputW * inputC * sizeof(float);

        // allocate memory for input
        inputBuffer = nullptr;
        cudaMalloc(&inputBuffer, inputSize);

        NODELET_INFO_STREAM("Shape dim 0: "<<dims.d[0]);
        NODELET_INFO_STREAM("Shape dim 1: "<<dims.d[1]);
        NODELET_INFO_STREAM("Shape dim 2: "<<dims.d[2]);
        NODELET_INFO_STREAM("Shape dim 3: "<<dims.d[3]);
        NODELET_INFO_STREAM("Size: "<<inputSize);
        NODELET_INFO_STREAM("Data Type: "<<dataTypeStr);
      } 
      else {
        NODELET_INFO_STREAM("Model Binding " << i << " type: OUTPUT");

        std::vector<int> shape(dims.nbDims);
        size_t shapeSize = 1;

        // compute shape and tensor size
        for (int j = 0; j < dims.nbDims; ++j) {
          shape[j] = dims.d[j];
          shapeSize *= dims.d[j];
          NODELET_INFO_STREAM("Shape dim "<<j<<": "<<dims.d[j]);
        }
        shapeSize *= sizeof(float);
        NODELET_INFO_STREAM("Size: "<<shapeSize);
        NODELET_INFO_STREAM("Data Type: "<<dataTypeStr);
        outputShapes.push_back(shape);

        // allocate output buffer
        void *outBuffer = nullptr;
        cudaMalloc(&outBuffer, shapeSize);
        outputBuffers.push_back(outBuffer);
      }
    }

    // Populate the buffers vector with input and output pointers
    buffers.push_back(inputBuffer);
    for (size_t i = 0; i < outputBuffers.size(); ++i) {
        buffers.push_back(outputBuffers[i]);
    }

    // Create TensorRT execution context
    trtContext = trtEngine->createExecutionContext();

    if (!trtContext) {
        NODELET_ERROR_STREAM("Error creating execution context.");
        exit(EXIT_FAILURE);
    }
    else{
      NODELET_INFO_STREAM("Correctly created execution context.");
    }

    // compute decoding helpers
    nbElementsPerBox = (NUM_VALS_PER_BOX + numClasses);
    for (int i=0; i<gridDims.size(); i++) {
      nbBoxesPerBuffer.push_back(gridDims[i] * gridDims[i] * NUM_BOXES);
      nbElemsPerBuffer.push_back(nbBoxesPerBuffer[i] * nbElementsPerBox);
    }

    nodeletInitialized = true;
  }

  void Yolov3ObjDetNodelet::imgCallback(const sensor_msgs::ImageConstPtr &msg) {

    // do not execute callback if the nodelet is not initialized
    if (!nodeletInitialized) {
      return;
    }
    
    NODELET_INFO_STREAM(" => Received image msg with shape: (" << msg->width << ", " << msg->height << ")");

    // convert image msg to cv image
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      NODELET_INFO_STREAM("Image Orig size: " << cv_ptr->image.cols << "x" << cv_ptr->image.rows);
    } 
    catch (cv_bridge::Exception& e) {
      NODELET_ERROR_STREAM("cv_bridge exception: " << e.what());
      return;
    }

    // preprocess image
    cv::Mat inputImage = preprocessImage(cv_ptr);
    NODELET_INFO_STREAM("Processed image size: (" << inputImage.cols << ", " << inputImage.rows << ")");

    // Copy preprocessed image to the input buffer
    cudaMemcpy(inputBuffer, inputImage.data, inputSize, cudaMemcpyHostToDevice);

    // Run inference using TensorRT
    trtContext->executeV2(buffers.data());


    // decoding buffers into matrixes
    std::vector<std::vector<MatrixXf>> outputBatch;

    // for each batch
    NODELET_INFO_STREAM("------------- DECODING ------------------");
    NODELET_INFO_STREAM("BATCH SIZE: "<<batchSize);
    for (int i=0; i<batchSize; i++) {
      
      std::vector<MatrixXf> output_i;
      
      // for each yolo layer
      NODELET_INFO_STREAM("BATCH: "<<i);
      for (int j=0; j<outputBuffers.size(); j++) {

        //initialize the eigen matrix for this layer
        int nbBoxes = nbBoxesPerBuffer[j];
        MatrixXf yoloTensor_j(nbBoxes, nbElementsPerBox);
        
        NODELET_INFO_STREAM("YOLO TENSOR "<<j<<" SHAPE: ("<<nbBoxes<<","<<nbElementsPerBox<<")");

        // copy values from output buffer to eigen matrix
        float *buffer_j = (float*) outputBuffers[j];
        int nbElems_j = nbElemsPerBuffer[j];
        for (int b=0; b<nbBoxes; b++) {
          
          // The box index inside the output buffer.
          // i*nbElems_j selects the correct batch
          // b*nbElementsPerBox selects the correct box
          int boxIdx = i * nbElems_j + b * nbElementsPerBox;

          // NODELET_INFO_STREAM("BOX IDX: "<<boxIdx<<", b="<<b<<"/"<<nbBoxes);

          for (int boxElementIdx=0; boxElementIdx<nbElementsPerBox; boxElementIdx++) {
            // NODELET_INFO_STREAM("ELEMENT IDX: "<<boxIdx + boxElementIdx<<" MATRIX IDX: ["
            //                     <<b<<","<<boxElementIdx<<"]/["<<nbBoxes<<","
            //                     <<nbElementsPerBox<<"]");
            NODELET_INFO_STREAM("ELEM: "<<buffer_j[boxIdx + boxElementIdx]);
            // yoloTensor_j(b, boxElementIdx) = buffer_j[boxIdx + boxElementIdx]; 
          }
        }
        output_i.push_back(yoloTensor_j);
      }
      outputBatch.push_back(output_i);
    }

    

    // show image
    // cv::imshow("OPENCV_WINDOW", inputImage);
    // cv::waitKey(1);
  }

  std::string Yolov3ObjDetNodelet::readEngineFile() {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        NODELET_ERROR_STREAM("Error opening engine file: " << enginePath);
        exit(EXIT_FAILURE);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  cv::Mat Yolov3ObjDetNodelet::preprocessImage(const cv_bridge::CvImageConstPtr &cv_ptr) {
    int new_h = cv_ptr->image.rows;
    int new_w = cv_ptr->image.cols;

    // Determine the new size of the image
    if (static_cast<float>(inputW) / new_w < static_cast<float>(inputH) / new_h) {
        new_h = (new_h * inputW) / new_w;
        new_w = inputW;
    } else {
        new_w = (new_w * inputH) / new_h;
        new_h = inputH;
    }

    // Resize the image to the new size
    cv::Mat resized_image;
    cv::resize(cv_ptr->image, resized_image, cv::Size(new_w, new_h));

    // Convert color from BGR to RGB
    cv::Mat image_rgb;
    cv::cvtColor(resized_image, image_rgb, cv::COLOR_BGR2RGB);

    // Convert image to float and normalize
    image_rgb.convertTo(image_rgb, CV_32FC3);
    image_rgb /= 255.0f;

    // Embed the image into the standard letterbox
    cv::Mat new_image = cv::Mat::ones(inputH, inputW, CV_32FC3) * 0.5f;
    new_image.setTo(cv::Scalar(0.5f, 0.5f, 0.5f));
    cv::Rect roi((inputW - new_w) / 2, (inputH - new_h) / 2, new_w, new_h);
    image_rgb.copyTo(new_image(roi));

    return std::move(new_image);
  }

}  // namespace zed_nodelets