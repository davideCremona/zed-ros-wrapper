//
// Created by Davide Cremona on 20/07/2023.
//

#include "yolov3_obj_det.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv4/opencv2/opencv.hpp>

#ifndef NDEBUG
#include <ros/console.h>
#endif

namespace zed_nodelets
{
  Yolov3ObjDetNodelet::~Yolov3ObjDetNodelet() {}

  void Yolov3ObjDetNodelet::onInit() {
    // Node handlers
    mNh = getNodeHandle();
    mNhP = getPrivateNodeHandle();
    it_ = std::make_shared<image_transport::ImageTransport>(mNh);

    NODELET_INFO("********** Starting nodelet '%s' **********", getName().c_str());

    image_sub_ = it_->subscribe("/zed2/zed_node/left/image_rect_color", 1, &Yolov3ObjDetNodelet::imgCallback, this);
    NODELET_INFO_STREAM(" * Subscribed to topic: " << image_sub_.getTopic().c_str());
  }

  void Yolov3ObjDetNodelet::imgCallback(const sensor_msgs::ImageConstPtr &msg) {
    
    NODELET_INFO_STREAM(" => Received image msg");

    // convert image msg to cv image
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } 
    catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // show image
    cv::imshow("OPENCV_WINDOW", cv_ptr->image);
    cv::waitKey(1);
  }

}  // namespace zed_nodelets