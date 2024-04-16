import rospy
import cv2
import message_filters
import os

import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple

from .airsim_base.client import MultirotorClient
from .airsim_base.types import Vector3r, Quaternionr, Pose
from .airsim_base.utils import to_quaternion


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Quaternion, TransformStamped

from airsim_ros_pkgs.msg import VelCmd, GimbalAngleQuatCmd
from airsim_ros_pkgs.srv import Takeoff, Land

from cv_bridge import CvBridge, CvBridgeError



       
class QuarotorROS(MultirotorClient):            
    @staticmethod
    def image_transport(img_msg):
        try:
            return CvBridge().imgmsg_to_cv2(img_msg, "passthrough")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
    
    @staticmethod
    def rcv_image(cv_img : NDArray, dw : int, dh : int):
        return cv2.resize(cv_img.copy(), (dw, dh), interpolation = cv2.INTER_AREA)
    
    @staticmethod
    def tf_to_list(tf : TransformStamped):
        x, y, z = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
        qx, qy, qz, qw = tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w
        return [x, y, z, qx, qy, qz, qw]
        
    def __init__(self,
                  ip : str, 
                  vehicle_name : str, 
                  camera_name : str, 
                  observation_type : str,
                  resize_img : Tuple):
        
        MultirotorClient.__init__(self, ip)
        rgb_sub = message_filters.Subscriber("/airsim_node/"+vehicle_name+"/stereo/Scene", Image)
        depth_sub = message_filters.Subscriber("/airsim_node/"+vehicle_name+"/stereo/DepthPlanar", Image)
        tf_cam_sub = message_filters.Subscriber("/airsim_node/"+vehicle_name+"/"+camera_name+"/tf", TransformStamped)
        
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, tf_cam_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self._callback_stereo)
        
        if observation_type == 'panoptic':
            seg_sub = message_filters.Subscriber("/airsim_node/"+vehicle_name+"/stereo/Segmentation", Image)
            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, seg_sub, tf_cam_sub], 10, 0.1, allow_headerless=True)
            ts.registerCallback(self._callback_panoptic)
            
        
        self.gimbal_pub = rospy.Publisher("/airsim_node/gimbal_angle_quat_cmd", \
                                        GimbalAngleQuatCmd, queue_size=1)

        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)

        
        self.__vehicle_name = vehicle_name
        self.__camera_name = camera_name
        self.__observation_type = observation_type
        self.__resize_img = resize_img
        
        self.rgb = np.array([])
        self.depth = np.array([])
        self.segmentation = np.array([])
        self.tf = TransformStamped()
        self.gimbal_orientation = to_quaternion(0, 0, 0)
        
        self.position = None
        self.past_position = None
        
        


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
    def _callback_stereo(self, rgb_data, depth_data, tf_data):
        self.rgb = self.image_transport(rgb_data)
        self.depth = self.image_transport(depth_data)
        self.tf = self.tf_to_list(tf_data)
        
    def _callback_panoptic(self, rgb_data, depth_data, segmentation_data, tf_data):
        self.rgb = self.image_transport(rgb_data)
        self.depth = self.image_transport(depth_data)
        self.segmentation = self.image_transport(segmentation_data)
        self.tf = self.tf_to_list(tf_data)
        
        
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
    def get_views(self) -> List[NDArray]:
        return [self.rgb, self.depth] if self.__observation_type == "stereo" else [self.rgb, self.depth, self.segmentation]
        
    def get_observation(self) -> NDArray:
        w, h = self.__resize_img
        rgb = self.rcv_image(self.rgb, w, h)
        depth = self.rcv_image(self.depth, w, h)
        obs = [rgb, depth] + self.tf
        
        if self.__observation_type == 'panoptic':
            segmentation = self.rcv_image(self.segmentation, w, h)
            obs == [rgb, depth, segmentation] + self.tf
            
        return obs
    
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

        self.gimbal_pub.publish(gimbal)
    

class Trajectory(QuarotorROS):
    def __init__(self,
                  ip : str, 
                  vehicle_name : str, 
                  camera_name : str, 
                  observation_type : str,
                  resize_img : Tuple):
        
        super().__init__(ip, vehicle_name, camera_name, observation_type, resize_img)

        self.velocity_pub = rospy.Publisher("/airsim_node/"+vehicle_name+"/vel_cmd_body_frame", \
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
        
        self.velocity_pub.publish(vel)
        
    def get_state(self, action : NDArray) -> Tuple[NDArray, bool]:
        linear_x, linear_y, linear_z, angular_z, gimbal_pitch = action
        
        vx = np.clip(linear_x, -10, 10)
        vy = np.clip(linear_y, -10, 10)
        vz = np.clip(linear_z, -10, 10)
        
        omegaz = np.clip(angular_z, -5, 5)

        pitch = np.clip(gimbal_pitch, -45, 45)
        action_pitch = to_quaternion(pitch, 0, 0)
        
        self._velocity(vx, vy, vz, omegaz)
        self.gimbal(action_pitch)

        done = False # condition to finish
        if done:
            self.reset()
            return self.get_observation(), done
        
        done = False
        return self.get_observation(), done
        
        
        
class Position(QuarotorROS):
    def __init__(self,
                  ip : str, 
                  vehicle_name : str, 
                  camera_name : str, 
                  observation_type : str,
                  resize_img : Tuple):
        
        super().__init__(ip, vehicle_name, camera_name, observation_type, resize_img)
    
        self.vehicle_pose = self.simGetVehiclePose(vehicle_name)
        
    ##Functions
    def _pose(self, airsim_pose : Pose):
        next_position = self.vehicle_pose.position + airsim_pose.position
        next_orientation = self.vehicle_pose.orientation * airsim_pose.orientation
        self.vehicle_pose.position = next_position
        self.vehicle_pose.orientation = next_orientation
        self.simSetVehiclePose(Pose(next_position, next_orientation), False)
        
    def get_state(self, action : NDArray) -> Tuple[NDArray, bool]:
        x, y, z, yaw, gimbal_pitch= action
        
        px = np.clip(x, -60, 60)
        py = np.clip(y, -155, 155)
        pz = np.clip(z, -60, 60)
        yaw = np.clip(yaw, -45, 45)
        
        action_pose = Pose(Vector3r(px, py, pz), to_quaternion(0,0,yaw))
        self._pose(action_pose)
        
        pitch = np.clip(gimbal_pitch, -45, 45)
        action_pitch = to_quaternion(pitch, 0, 0)
        self.gimbal(action_pitch)
        
        done = False # condition to finish
        if done:
            self.reset() #reset of airsim
            return self.get_observation(), done
        
        done = False
        return self.get_observation(), done
        




