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
      NODELET_INFO_STREAM("Image Orig size: " << cv_ptr->image.cols << "x" << cv_ptr->image.rows);
    } 
    catch (cv_bridge::Exception& e) {
      NODELET_ERROR_STREAM("cv_bridge exception: " << e.what());
      return;
    }

    // preprocess image
    cv::Mat inputImage = preprocessImage(cv_ptr);
    NODELET_INFO_STREAM("Procedded image size: (" << inputImage.cols << ", " << inputImage.rows << ")");

    // show image
    cv::imshow("OPENCV_WINDOW", inputImage);
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