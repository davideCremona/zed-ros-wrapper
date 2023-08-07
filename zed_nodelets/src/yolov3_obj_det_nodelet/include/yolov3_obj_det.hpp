//
// Created by Davide Cremona on 20/07/2023.
//

#ifndef YOLOV3_OBJ_DET_HPP
#define YOLOV3_OBJ_DET_HPP

// headers to define nodelet
#include <nodelet/nodelet.h>

// ros
#include <ros/ros.h>
#include <ros/subscriber.h>
#include <image_transport/image_transport.h>

// messages
#include <sensor_msgs/Image.h>

// trt
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/opencv.hpp>

#include <Eigen/Dense>

using namespace Eigen;


class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;


// define the nodelet inside a namespace
namespace zed_nodelets
{
  // define nodelet as subclass of Nodelet class
  class Yolov3ObjDetNodelet : public nodelet::Nodelet
  {
    public:
      virtual ~Yolov3ObjDetNodelet();

    protected:
      /*! \brief Initialization function called by the Nodelet base class
      */
      virtual void onInit();

      /*! \brief Callback for full topics synchronization
      */
      void imgCallback(const sensor_msgs::ImageConstPtr &msg);

    private:

      // helper functions
      std::string readEngineFile();
      cv::Mat preprocessImage(const cv_bridge::CvImageConstPtr &cv_ptr);

      // Node handlers
      ros::NodeHandle mNh;   // Node handler
      ros::NodeHandle mNhP;  // Private Node handler

      // Node flags
      bool nodeletInitialized = false;

      // image transport
      std::shared_ptr<image_transport::ImageTransport> it_;
      image_transport::Subscriber image_sub_;

      // model params
      const int NUM_BOXES = 3;  // yolov3 predicts 3 boxes for each grid cell
      const int NUM_VALS_PER_BOX = 5; // x, y, w, h and confidence
      std::string enginePath;
      int batchSize;
      int inputW;
      int inputH;
      int inputC;
      std::vector<std::vector<int>> outputShapes;
      int numClasses;
      std::vector<int> gridDims;
      std::vector<int> anchorBoxes;

      // static inference helpers
      MatrixXd gridIdcsTensor;
      VectorXd gridDimsTensor;
      MatrixXd anchorsTensor;

      // inference
      cv::Mat inputImg;
      size_t inputSize;
      void *inputBuffer;
      std::vector<size_t> outputSizes;
      std::vector<void*> outputBuffers;
      std::vector<void*> buffers;

      //decoding
      // number of elements in corresponding buffer per image
      int nbElementsPerBox;
      std::vector<int> nbElemsPerBuffer;
      std::vector<int> nbBoxesPerBuffer;

      // trt stuffs
      // todo: using smart pointers makes me cry. I'm using raw ptrs. Is it safe? Do I have to del?
      nvinfer1::IRuntime *trtRuntime;
      nvinfer1::ICudaEngine *trtEngine;
      nvinfer1::IExecutionContext *trtContext;

  };
}  // namespace zed_nodelets

// export nodelet class
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(zed_nodelets::Yolov3ObjDetNodelet, nodelet::Nodelet)

#endif  // YOLOV3_OBJ_DET