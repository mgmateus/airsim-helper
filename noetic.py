import rospy
import cv2

import numpy as np

from numpy.typing import NDArray
from typing import Tuple

from .airsim_base.client import MultirotorClient
from .airsim_base.types import Vector3r, Quaternionr
from .airsim_base.utils import to_quaternion

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Quaternion

from airsim_ros_pkgs.msg import VelCmd, GimbalAngleQuatCmd
from airsim_ros_pkgs.srv import Takeoff, Land

from cv_bridge import CvBridge, CvBridgeError


def image_transport(img_msg):
    try:
        return CvBridge().imgmsg_to_cv2(img_msg, "passthrough")

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

class QuarotorStereoROS(MultirotorClient):            
    def __init__(self, ip : str, vehicle_name : str, camera_name : str, observation : str):
        MultirotorClient.__init__(self, ip)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/stereo/Scene", \
                         Image, self._callback_rgb)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/stereo/DepthPlanar", \
                         Image, self._callback_depth_raw)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/stereo/DepthVis", \
                         Image, self._callback_depth)
        
        self.__velocity_pub = rospy.Publisher("/airsim_node/"+vehicle_name+"/vel_cmd_world_frame", \
                                        VelCmd, queue_size=1)
        self.__gimbal_pub = rospy.Publisher("/airsim_node/gimbal_angle_quat_cmd", \
                                        GimbalAngleQuatCmd, queue_size=1)
        self.__pub_info = rospy.Publisher("uav_info", \
                                          String, queue_size=10)        

        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)

        self.__vehicle_name = vehicle_name
        self.__camera_name = camera_name
        self.__observation = observation
        self.__gimbal_orientation = to_quaternion(0, 0, 0)
        self.__rgb = np.array([])
        self.__depth = np.array([])
        self.__depth_vis = np.array([])

    @property
    def vehicle_name(self):
        return self.__vehicle_name
    
    @property
    def camera_name(self):
        return self.__camera_name
    
    @property
    def observation(self):
        return self.__observation
    
    @property
    def rgb(self):
        return self.__rgb 

    @rgb.setter
    def rgb(self, data):
        self.__rgb = data

    @property
    def depth(self):
        return self.__depth   
    
    @depth.setter
    def depth(self, data):
        self.__depth = data

    @property
    def depth_vis(self):
        return self.__depth_vis  
    
    @depth_vis.setter
    def depth_vis(self, data):
        self.__depth_vis = data
                
    ## Callbacks
    # def callback_image(func):
    #     def callback(self, *args, **kwargs):
    #         data, img_type = func(self, *args, **kwargs)
    #         if data:
    #             cv_img = image_transport(data)
    #             resized_img = cv2.resize(cv_img.copy(), (100, 100), interpolation = cv2.INTER_AREA)
    #             self.__setattr__(img_type, resized_img)
    #         else:
    #             info = f"Error in {img_type} cam!"
    #             self.__pub_info.publish(info)

    #     return callback
    
    # @callback_image
    # def _callback_rgb(self, data):
    #     return data, "rgb"
    
    # @callback_image
    # def _callback_depth(self, data):
    #     return data, "depth"
        
    def _callback_rgb(self, data):
        cv_img = image_transport(data)
        self.rgb = cv2.resize(cv_img.copy(), (100, 100), interpolation = cv2.INTER_AREA)

    def _callback_depth_raw(self, data):
        cv_img = image_transport(data)
        self.depth = cv_img
        # nan_location = np.isnan(cv_img)
        # cv_img[nan_location] = np.nanmax(cv_img)
        # norm_image =  (cv_img)*255./5.
        # norm_image[0,0] = 255.
        # norm_image = norm_image.astype('uint8')
        # norm_image = cv2.resize(norm_image.copy(), (100, 100), interpolation = cv2.INTER_AREA)
        # self.depth = norm_image
    
    def _callback_depth(self, data):
        cv_img = image_transport(data)
        self.depth_vis = cv2.resize(cv_img.copy(), (100, 100), interpolation = cv2.INTER_AREA)
    
    ## Services
    def take_off(self):
        try:
            service = rospy.ServiceProxy("/airsim_node/"+self.__vehicle_name+"/takeoff", Takeoff)
            rospy.wait_for_service("/airsim_node/"+self.__vehicle_name+"/takeoff")

            service()

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)

    def land(self):
        try:
            service = rospy.ServiceProxy("/airsim_node/"+self.__vehicle_name+"/land", Land)
            rospy.wait_for_service("/airsim_node/"+self.__vehicle_name+"/land")

            service()

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)
            
    ##Functions
    def _velocity(self, linear_x : float, linear_y : float, linear_z : float, angular_z : float):
        
        vel = VelCmd()
        vel.twist.linear.x = linear_x
        vel.twist.linear.y = linear_y
        vel.twist.linear.z = linear_z
        vel.twist.angular.x = 0
        vel.twist.angular.y = 0
        vel.twist.angular.z = angular_z
        
        self.__velocity_pub.publish(vel)
        
    def _gimbal(self, airsim_quaternion : Quaternionr):
        rotation = self.__gimbal_orientation * airsim_quaternion
        self.__gimbal_orientation = rotation
        quaternion = Quaternion()
        quaternion.x = self.__gimbal_orientation.x_val
        quaternion.y = self.__gimbal_orientation.y_val
        quaternion.z = self.__gimbal_orientation.z_val
        quaternion.w = self.__gimbal_orientation.w_val

        gimbal = GimbalAngleQuatCmd()
        gimbal.camera_name = "stereo"
        gimbal.vehicle_name = "Hydrone"
        gimbal.orientation = quaternion

        self.__gimbal_pub.publish(gimbal)

    def _observation(self) -> NDArray:
        return np.array([self.__rgb.transpose(2, 0, 1).copy(), self.__depth.transpose(2, 0, 1).copy()])

        
    def get_state(self, action : NDArray) -> Tuple[NDArray, bool]:
        linear_x, linear_y, linear_z, angular_x, angular_y, angular_z = action
        vx = np.clip(linear_x, -.25, .25)
        vy = np.clip(linear_y, -.25, .25)
        vz = np.clip(linear_z, -.25, .25)
        omegaz = np.clip(angular_z, -.25, .25)

        roll = np.clip(angular_x, -90, 90)
        pitch = np.clip(angular_y, -45, 45)
        yaw = np.clip(angular_z, -45, 45)
        airsimq = to_quaternion(pitch, roll, yaw)
        
        self._velocity(vx, vy, vz, omegaz)
        self._gimbal(airsimq)

        done = False # condition to finish
        if done:
            self.reset()
            return self._observation(), done
        
        done = False
        return self._observation(), done
        
        
        
    





