//
// Created by Davide Cremona on 20/07/2023.
//

#ifndef SIMPLE_IMG_SUB_HPP
#define SIMPLE_IMG_SUB_HPP

// headers to define nodelet
#include <nodelet/nodelet.h>

// ros
#include <ros/ros.h>
#include <ros/subscriber.h>
#include <image_transport/image_transport.h>

// messages
#include <sensor_msgs/Image.h>

// define the nodelet inside a namespace
namespace zed_nodelets
{
  // define nodelet as subclass of Nodelet class
  class SimpleImageSubNodelet : public nodelet::Nodelet
  {
    public:
      virtual ~SimpleImageSubNodelet();

    protected:
      /*! \brief Initialization function called by the Nodelet base class
      */
      virtual void onInit();

      /*! \brief Callback for full topics synchronization
      */
      void imgCallback(const sensor_msgs::ImageConstPtr &msg);

    private:
      // Node handlers
      ros::NodeHandle mNh;   // Node handler
      ros::NodeHandle mNhP;  // Private Node handler

      // image transport
      std::shared_ptr<image_transport::ImageTransport> it_;
      image_transport::Subscriber image_sub_;

  };
}  // namespace zed_nodelets

// export nodelet class
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(zed_nodelets::SimpleImageSubNodelet, nodelet::Nodelet)

#endif  // SIMPLE_IMG_SUB_HPP