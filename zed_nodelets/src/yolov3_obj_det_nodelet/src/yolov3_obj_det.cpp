//
// Created by Davide Cremona on 20/07/2023.
//

#include "yolov3_obj_det.hpp"
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

    // -------------- TRT INITIALIZATION ----------------
    engine_path = "/home/DATI/insulators/YOLOv3/models/YOLO_06_noSyn/bs1_608x608_fp16_engine.trt";
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
      NODELET_INFO_STREAM("Correctly Loaded engine: " << engine_path);
    }

    // read model parameters
    batchSize = trtEngine->getBindingDimensions(0).d[0];
    inputH = trtEngine->getBindingDimensions(0).d[1];
    inputW = trtEngine->getBindingDimensions(0).d[2];
    inputC = trtEngine->getBindingDimensions(0).d[3];

    NODELET_INFO_STREAM("Model Input Shape: (" << batchSize << ", " << inputH << ", " << inputW << ", " << inputC << ")");

    // Create TensorRT execution context
    trtContext = trtEngine->createExecutionContext();

    if (!trtContext) {
        NODELET_ERROR_STREAM("Error creating execution context.");
        exit(EXIT_FAILURE);
    }
    else{
      NODELET_INFO_STREAM("Correctly created execution context.");
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
    } 
    catch (cv_bridge::Exception& e) {
      NODELET_ERROR_STREAM("cv_bridge exception: " << e.what());
      return;
    }

    // show image
    cv::imshow("OPENCV_WINDOW", cv_ptr->image);
    cv::waitKey(1);
  }

  std::string Yolov3ObjDetNodelet::readEngineFile() {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        NODELET_ERROR_STREAM("Error opening engine file: " << engine_path);
        exit(EXIT_FAILURE);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
}  // namespace zed_nodelets