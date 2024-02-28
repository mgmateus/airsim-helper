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

class QuarotorROS(MultirotorClient):            
    def __init__(self,
                  ip : str, 
                  vehicle_name : str, 
                  camera_name : str, 
                  observation_type : str):
        
        MultirotorClient.__init__(self, ip)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/"+camera_name+"/Scene", \
                         Image, self._callback_rgb)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/"+camera_name+"/DepthPlanar", \
                         Image, self._callback_depth)
        
        if observation_type == 'panoptic':
            rospy.Subscriber("/airsim_node/"+vehicle_name+"/"+camera_name+"/Segmentation", \
                         Image, self._callback_segmentation)
            
        
        self.__gimbal_pub = rospy.Publisher("/airsim_node/gimbal_angle_quat_cmd", \
                                        GimbalAngleQuatCmd, queue_size=1)

        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)

        
        self.__vehicle_name = vehicle_name
        self.__camera_name = camera_name
        self.__observation_type = observation_type

        self.gimbal_orientation = to_quaternion(0, 0, 0)
        self.rgb = np.array([])
        self.depth = np.array([])
        self.segmentation = np.array([])


    @property
    def vehicle_name(self):
        return self.__vehicle_name
    
    @property
    def camera_name(self):
        return self.__camera_name
    
    @property
    def observation_type(self):
        return self.__observation_type
                
    # Callbacks
    def callback_image(func):
        def callback(self, *args, **kwargs):
            data, img_type = func(self, *args, **kwargs)
            if data:
                cv_img = image_transport(data)
                resized_img = cv2.resize(cv_img.copy(), (100, 100), interpolation = cv2.INTER_AREA)
                self.__setattr__(img_type, resized_img)
            else:
                info = f"Error in {img_type} cam!"
                self.__pub_info.publish(info)

        return callback
    
    @callback_image
    def _callback_rgb(self, data):
        return data, "rgb"
    
    @callback_image
    def _callback_depth(self, data):
        return data, "depth"
    
    @callback_image
    def _callback_segmentation(self, data):
        return data, "segmentation"
    
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
    def observation(self) -> NDArray:
        return [self.rgb, self.depth] if self.__observation_type == "stereo" else [self.rgb, self.depth, self.segmentation]
    
    def gimbal(self, airsim_quaternion : Quaternionr):
        rotation = self.gimbal_orientation * airsim_quaternion
        self.gimbal_orientation = rotation
        quaternion = Quaternion()
        quaternion.x = self.gimbal_orientation.x_val
        quaternion.y = self.gimbal_orientation.y_val
        quaternion.z = self.gimbal_orientation.z_val
        quaternion.w = self.gimbal_orientation.w_val

        gimbal = GimbalAngleQuatCmd()
        gimbal.camera_name = self.camera_name
        gimbal.vehicle_name = self.vehicle_name
        gimbal.orientation = quaternion

        self.__gimbal_pub.publish(gimbal)
    

class Trajectory(QuarotorROS):
    def __init__(self,
                  ip : str, 
                  vehicle_name : str, 
                  camera_name : str, 
                  observation_type : str):
        
        super().__init__(ip, vehicle_name, camera_name, observation_type)

        self.__velocity_pub = rospy.Publisher("/airsim_node/"+vehicle_name+"/vel_cmd_body_frame", \
                                        VelCmd, queue_size=1)
        
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
        
    def get_state(self, action : NDArray) -> Tuple[NDArray, bool]:
        linear_x, linear_y, linear_z, angular_z, gimbal_pitch = action
        vx = np.clip(linear_x, -.25, .25)
        vy = np.clip(linear_y, -.25, .25)
        vz = np.clip(linear_z, -.25, .25)
        omegaz = np.clip(angular_z, -.25, .25)

        pitch = np.clip(gimbal_pitch, -45, 45)
        airsimq = to_quaternion(pitch, 0, 0)
        
        self._velocity(vx, vy, vz, omegaz)
        self.gimbal(airsimq)

        done = False # condition to finish
        if done:
            self.reset()
            return self.observation(), done
        
        done = False
        return self.observation(), done
        
        
        
class Position(QuarotorROS):
    def __init__(self,
                  ip : str, 
                  vehicle_name : str, 
                  camera_name : str, 
                  observation_type : str):
        
        super().__init__(ip, vehicle_name, camera_name, observation_type)

    ##Functions





