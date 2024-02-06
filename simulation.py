#!/usr/bin/env python3

import rospy
import numpy as np

from typing import List, Tuple, NewType
from threading import Thread

from airsim_base.client import MultirotorClient, VehicleClient
from airsim_base.types import Vector3r, KinematicsState

from utils import sum_vector, dot_scalar_vector

from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from airsim_ros_pkgs.msg import Image, VelCmd
from airsim_ros_pkgs.srv import Takeoff, Land

numpy_array = NewType('numpy_array', np.array)

def set_client(ip : str, client : str):
     if client == "computer_vision":
          return VehicleClient(ip)
     
     else: 
        return MultirotorClient(ip)

class _Ue4Briedge:
    """Starts communication's Engine.

    Args:
        HOST: 
            -For set ip address from host, use ipconfig result's on host
            -For set ip address from docker network, use a ip from ping result's between containers on host
            -For set ip address from WSL, os.environ['WSL_HOST_IP'] on host.
    """ 
    def __init__(self, ip : str, client : str):
        self.__client = set_client(ip, client)
        print(type(self.__client))
        self.__client.confirmConnection()

    def reset(self, debbug = True):
        self.__client.reset()
        if debbug:
            f"Restart Connection : {self.__client.ping()}"

    def get_ground_truth_kinematics(self) -> KinematicsState:
        return self.__client.simGetGroundTruthKinematics()
    
    def get_kinematics(self):
        kinematics = self.__client.getMultirotorState().kinematics_estimated
        return kinematics if type(self.__client) == MultirotorClient else KinematicsState()
    
    def set_Kinematics(self, state):
        self.__client.simSetKinematics(state, ignore_collision=False)
        
    def rotation_rate(self, roll : float, pitch : float, yaw : float, z : float):
        self.__client.moveByAngleRatesZAsync(roll, pitch, yaw, z, duration=10)
        
    def take_off(self):
        self.__client.enableApiControl(True)
        self.__client.takeoffAsync().join()
    
class ComputerVision(_Ue4Briedge):
    def __init__(self, ip : str):
        _Ue4Briedge.__init__(self, ip, client="computer_vision")
       
    def trajectory(self, vel_rate : np.array, dt : float = 0.01):
        vx, vy, vz, pitch, yaw = vel_rate

        linear_velocity = Vector3r(vx, vy, vz)
        angular_velocity = Vector3r(pitch, 0, yaw)

        state = KinematicsState()
        state.position = Vector3r(vx*dt, vx*dt, vx*dt)
        state.linear_velocity = linear_velocity
        state.angular_velocity = angular_velocity

        self.set_Kinematics(state)
        
class QuadrotorClient(_Ue4Briedge):
    def __init__(self, ip : str):
        _Ue4Briedge.__init__(self, ip, client="")
        ...
        
class QuarotorROS:
    
    @staticmethod
    def image_transport(img_msg):
        rospy.logwarn(img_msg.header)
        try:
            return CvBridge().imgmsg_to_cv2(img_msg, "passthrough")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            
    def __init__(self, vehicle_name : str):
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/Stereo_Cam/Scene", \
                         Image, self._callback_rgb)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/Stereo_Cam/DepthPerspective", \
                         Image, self._callback_depth)
        
        self.__vel_pub = rospy.Publisher("/airsim_node/"+vehicle_name+"/vel_cmd_world_frame", \
                                        VelCmd, queue_size=1)
        self.__pub_info = rospy.Publisher("uav_info", \
                                          String, queue_size=10)
        
        self.__vehicle_name = vehicle_name
    
    def __str__(self) -> str:
        return self.__vehicle_name
            
    ## Callbacks
    def callback_image(func):
        def callback(self, *args, **kwargs):
            data, img_type = func(self, *args, **kwargs)
            if data:
                cv_rgb = self.image_transport(data)
                self.__setattr__("__"+img_type, cv_rgb)
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
    
    ## Services
    def takeOff(self):
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
    def set_velocity(self, linear_x : float, linear_y : float, linear_z : float, \
                    angular_x : float, angular_y : float, angular_z : float):
        
        vel = VelCmd()
        vel.twist.linear.x = linear_x
        vel.twist.linear.y = linear_y
        vel.twist.linear.z = linear_z
        vel.twist.angular.x = angular_x
        vel.twist.angular.y = angular_y
        vel.twist.angular.z = angular_z
        
        self.__vel_pub.publish(vel)
        
    def get_state(self, action : numpy_array) -> Tuple[numpy_array, bool]:
        linear_x, linear_y, linear_z, angular_x, angular_y, angular_z = action
        linear_x = np.clip(linear_x, -.25, .25)
        linear_y = np.clip(linear_y, -.25, .25)
        linear_z = np.clip(linear_z, -.25, .25)
        angular_x = np.clip(angular_x, -.25, .25)
        angular_y = np.clip(angular_y, -.25, .25)
        angular_z = np.clip(angular_z, -.25, .25)
        self.set_velocity(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)

        
        
        
    





