import rospy
import copy
import cv2
import message_filters
import os

import numpy as np
import open3d as o3d

from functools import singledispatchmethod
from numpy.typing import NDArray
from typing import List

from airsim_base.client import MultirotorClient
from airsim_base.types import Vector3r, Quaternionr, Pose, ImageType
from airsim_base.utils import to_quaternion, to_eularian_angles, random_choice, theta


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped

from airsim_ros_pkgs.msg import GimbalAngleEulerCmd
from airsim_ros_pkgs.srv import Takeoff, Land

from cv_bridge import CvBridge, CvBridgeError


class RotorPyROS(MultirotorClient):       
         
    @staticmethod
    def image_transport(img_msg):
        try:
            return CvBridge().imgmsg_to_cv2(img_msg, "passthrough")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
    
    @staticmethod
    def o3d_intrinsic_matrix(image_dim : tuple, camera_fov : float):
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        image_width, image_height = image_dim
        fov_rad = camera_fov * np.pi/180
        fd = (image_width/2.0) / np.tan(fov_rad/2.0)
        intrinsic.set_intrinsics(image_width, image_height, fd, fd, image_width/2 - 0.5, image_height/2 - 0.5)
        
        return intrinsic
    
    @staticmethod
    def tf_to_list(tf : TransformStamped):
        x, y, z = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z
        qx, qy, qz, qw = tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w
        return np.array([x, y, z, qx, qy, qz, qw])
    
    @staticmethod
    def vector3_from_list(v : list):
        x, y, z = v
        return Vector3r(x, y, z)
    
    @staticmethod
    def quaternionr_from_list(q : list):
        x, y, z, w = q
        return Quaternionr(x, y, z, w)
    
    @staticmethod
    def pose_from_positon_euler_list(pose : list):
        x, y, z, roll, pitch, yaw= pose
        return Pose(Vector3r(x, y, z,), to_quaternion(roll, pitch, yaw))
    
    @staticmethod
    def pose_from_positon_quat_list(pose : list):
        x, y, z, qx, qy, qz, qw = pose
        return Pose(Vector3r(x, y, z,), Quaternionr(qx, qy, qz, qw))
    
    @staticmethod
    def pose_to_list(pose : Pose):
        position = pose.position.to_numpy_array().tolist()
        orientation = pose.orientation.to_numpy_array().tolist()
        return position+orientation
    
    @staticmethod
    def pose_to_position_quat(pose : Pose):
        position = pose.position.to_numpy_array().tolist()
        orientation = pose.orientation.to_numpy_array().tolist()
        return position, orientation
    
    @staticmethod
    def pose_to_position_euler(pose : Pose):
        position = pose.position.to_numpy_array().tolist()
        orientation = list(to_eularian_angles(pose.orientation))
        return position, orientation
        
    def __init__(self,
                  ip : str,
                  image_dim : tuple,
                  camera_fov : float):
        
        MultirotorClient.__init__(self, ip)         

        self.confirmConnection()

        self.__intrinsic = self.o3d_intrinsic_matrix(image_dim, camera_fov)
        self.__gimbal = to_quaternion(0,0,0)
        self.__gimbal_pub = rospy.Publisher("/airsim_node/gimbal_angle_euler_cmd", \
                                        GimbalAngleEulerCmd, queue_size=1)
        
        

    @property
    def gimbal(self):
        return self.__gimbal
    
    @gimbal.setter
    def gimbal(self, angles : list):
        roll, pitch, yaw = angles
        q = to_quaternion(roll=roll, pitch=pitch, yaw=yaw)
        self.__gimbal *= q
    
    
    ## Services
    def take_off(self, vehicle_name : str = '', flag : bool = False):
        try:
            service = rospy.ServiceProxy("/airsim_node/"+vehicle_name+"/takeoff", Takeoff)
            rospy.wait_for_service("/airsim_node/"+vehicle_name+"/takeoff")

            service(flag)

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)

    def land(self, vehicle_name : str = ''):
        try:
            service = rospy.ServiceProxy("/airsim_node/"+vehicle_name+"/land", Land)
            rospy.wait_for_service("/airsim_node/"+vehicle_name+"/land")

            service(True)

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)
    
    ##Functions        
    def rgb_image(self, vehicle_name : str ='', camera_name : str =''):
        image_data = None
        cv_image = None
        while image_data is None :
            try:
                image_data = rospy.wait_for_message("/airsim_node/"+vehicle_name+"/"+camera_name+"/Scene", Image, timeout=5)
                cv_image = self.image_transport(image_data)  
            except:
                pass

        return cv_image
    
            
    def depth_image(self, vehicle_name : str ='', camera_name : str =''):
        image_data = None
        cv_image = None
        while image_data is None :
            try:
                image_data = rospy.wait_for_message("/airsim_node/"+vehicle_name+"/"+camera_name+"/DepthPlanar", Image, timeout=5)
                cv_image = self.image_transport(image_data)              
            except:
                pass

        return cv_image
    
    
    def segmentation_image(self, vehicle_name : str ='', camera_name : str =''):
        image_data = None
        cv_image = None
        while image_data is None :
            try:
                image_data = rospy.wait_for_message("/airsim_node/"+vehicle_name+"/"+camera_name+"/Segmentation", Image, timeout=5)
                cv_image = self.image_transport(image_data)                
            except:
                pass

        return cv_image
    
    def rgbd_image(self, image : NDArray, depth : NDArray):
        o3d_image = o3d.geometry.Image(image)
        o3d_depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, 
                                                                  o3d_depth, 
                                                                  depth_scale=1.0, 
                                                                  depth_trunc=1000, 
                                                                  convert_rgb_to_intensity=False)
            
        return rgbd
    
    def point_cloud(self, image : NDArray, depth : NDArray):
        rgbd = self._rgbd(image, depth)
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.__intrinsic)
    
    def tf(self, vehicle_name : str ='', camera_name : str =''):
        tf_data = None
        tf_ = None
        while tf_data is None :
            try:
                tf_data = rospy.wait_for_message("/airsim_node/"+vehicle_name+"/"+camera_name+"/tf", TransformStamped, timeout=5)
                tf_ = self.tf_to_list(tf_data)         
            except:
                pass

        return tf_
    
    
    def get_object_pose(self, object_name: str):
        return self.simGetObjectPose(object_name)
    
    def get_object_pose_as_list(self, object_name: str):
        obj_pose = self.get_object_pose(object_name)
        
        position = list(obj_pose.position.to_numpy_array())
        orientation = list(to_eularian_angles(obj_pose.orientation))
        return position + orientation
    
    def get_pose(self, vehicle_name : str = ''):
        pose = self.simGetVehiclePose(vehicle_name)
        return self.pose_to_position_quat(pose)
        
    def set_gimbal(self, roll : float, pitch : float, yaw : float, vehicle_name : str ='', camera_name : str =''):
        gimbal_msg = GimbalAngleEulerCmd()
        gimbal_msg.vehicle_name = vehicle_name
        gimbal_msg.camera_name = camera_name
        gimbal_msg.roll = roll
        gimbal_msg.pitch = pitch
        gimbal_msg.yaw = yaw
        self.__gimbal_pub.publish(gimbal_msg)
    
    def set_object_pose(self, pose : list, object_name : str = ''):
        x, y, z, roll, pitch, yaw = pose
        start_pose = Pose(Vector3r(x, y, z), to_quaternion(pitch, roll, yaw))
        self.simSetObjectPose(object_name, start_pose)
        
        return True
    
    def set_pose(self, pose: list, vehicle_name : str = ''):
        x, y, z, roll, pitch, yaw = pose
        start_pose = Pose(Vector3r(x, y, z), to_quaternion(pitch, roll, yaw))
        
        self.simSetVehiclePose(start_pose, ignore_collision=True, vehicle_name=vehicle_name)
        return True
    
    def start(self, vehicle_name : str = ''):
        self.enableApiControl(True, vehicle_name)
        self.armDisarm(True, vehicle_name)
        
    
    
    
class DualActPose(RotorPyROS):
    def __init__(self, ip: str, vehicle_cfg : dict, shadow_cfg : dict, observation_type : str):        
        super().__init__(ip, vehicle_cfg['camera']['dim'], vehicle_cfg['camera']['fov'])
        
        self.__vehicle_cfg = vehicle_cfg
        self.__shadow_cfg = shadow_cfg
        self.__obs_type = observation_type
        self.__home = self.pose_from_positon_euler_list(vehicle_cfg['start_pose'])
        self.__dual_pose = self.pose_from_positon_euler_list(shadow_cfg['start_pose'])
        
        self._start()
    
    @property
    def vehicle_and_camera_name(self):
        return self.__vehicle_cfg['name'], self.__vehicle_cfg['camera']['name']   
    
    @property
    def home(self):
        return self.pose_to_position_quat(self.__home)
    
    @home.setter
    def home(self, new_pose : list):
        pose = self.pose_from_positon_euler_list(new_pose)
        self.__home = copy.deepcopy(pose)
        self.__dual_pose = copy.deepcopy(pose)   
    
    @property
    def altitude(self):
        return -1 * self.__dual_pose.position.z_val
    
    @property
    def altitude_min(self):
        altitude = self.__vehicle_cfg['altitude'] 
        altitude_min = altitude if type(altitude) == float else altitude[0]
        return altitude_min
    
    @property
    def altitude_max(self):
        return self.__vehicle_cfg['altitude'][1] 
    
    @property
    def rgb(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        return {'rgb' : self.rgb_image(vehicle_name, camera_name), 'tf' : self.tf(vehicle_name, camera_name)}
    
    @property
    def depth(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        return {'depth' : self.depth_image(vehicle_name, camera_name), 
                'tf' : self.tf(vehicle_name, camera_name)}
    
    @property
    def segmentation(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        return {'segmentation' : self.segmentation_image(vehicle_name, camera_name), 
                'tf' : self.tf(vehicle_name, camera_name)}
    
    @property
    def stereo(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        return {'rgb' : self.rgb_image(vehicle_name, camera_name), 
                'depth' : self.depth_image(vehicle_name, camera_name), 
                'tf' : self.tf(vehicle_name, camera_name)}
    
    @property
    def panoptic(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        return {'rgb' : self.rgb_image(vehicle_name, camera_name), 
                'depth' : self.depth_image(vehicle_name, camera_name), 
                'segmentation' : self.segmentation_image(vehicle_name, camera_name), 
                'tf' : self.tf(vehicle_name, camera_name)}
    
    @property
    def stereo_occupancy(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        rgb = self.rgb_image(vehicle_name, camera_name)
        depth = self.depth_image(vehicle_name, camera_name)
        pcd = self.point_cloud(rgb[...,::-1].copy(), depth.astype(np.float32))
        
        return {'rgb' : rgb, 'depth' : depth, 'point_cloud' : pcd, 'tf' : self.tf(vehicle_name, camera_name)}
    
    
    @property
    def panoptic_occupancy(self):
        vehicle_name = self.__vehicle_cfg['name']
        camera_name = self.__vehicle_cfg['camera']['name']
        rgb = self.rgb_image(vehicle_name, camera_name)
        depth = self.depth_image(vehicle_name, camera_name)
        pcd = self.point_cloud(rgb[...,::-1].copy(), depth.astype(np.float32))
        
        return {'rgb' : rgb, 
                'depth' : depth, 
                'segmentation' : self.segmentation_image(vehicle_name, camera_name), 
                'point_cloud' : pcd, 
                'tf' : self.tf(vehicle_name, camera_name)}
    
    @property
    def observation(self) -> NDArray:
        return self.__getattribute__(self.__obs_type)
    
    
    def _start(self):
        self.start(self.__vehicle_cfg['name'])
        self.start(self.__shadow_cfg['name'])
        
        self.simSetDetectionFilterRadius(self.__shadow_cfg['camera']['name'], 
                                                 ImageType.Scene, 200 * 100, 
                                                 vehicle_name=self.__shadow_cfg['name'])
        
    def _gimbal(self, pitch : float, deg2rad : bool = True):
        self.gimbal = [0, np.deg2rad(pitch), 0] if deg2rad else [0, pitch, 0]
        pose = Pose(Vector3r(0.3,0,0.3), self.gimbal)
        self.simSetCameraPose(self.__vehicle_cfg['camera']['name'], pose, self.__vehicle_cfg['name'])
        self.simSetCameraPose(self.__shadow_cfg['camera']['name'], pose, self.__shadow_cfg['name'])
        
        return True
        
    def _pose(self, new_pose : list):
        pose = self.pose_from_positon_euler_list(new_pose)
        self.__dual_pose.position += pose.position
        self.__dual_pose.orientation *= pose.orientation
        
        self.simSetVehiclePose(self.__dual_pose, True)
        self.simSetVehiclePose(self.__dual_pose, True, vehicle_name=self.__shadow_cfg['name'])
        
        return True
    
    def go_home(self):
        _, rpitch, _ = to_eularian_angles(self.gimbal)
        self._gimbal(-rpitch)
        self.simSetVehiclePose(self.__home, True)
        self.simSetVehiclePose(self.__home, True, vehicle_name=self.__shadow_cfg['name'])
        self.__dual_pose = copy.deepcopy(self.__home)
        return True
        
    def next_point_of_view(self, five_DoF : NDArray):
        px, py, pz, yaw, gimbal_pitch= five_DoF
        
        self._pose([px, py, pz, 0, 0, yaw])
        self._gimbal(gimbal_pitch)
        
        return True
    
    def set_detection(self, detect : str):
        self.simAddDetectionFilterMeshName(self.__shadow_cfg['camera']['name'], 
                                              ImageType.Scene, f"{detect}*", 
                                              vehicle_name=self.__shadow_cfg['name'])
        
    def detection_distance(self):
        wx, wy, wz, _, _, _ = self.__shadow_cfg['global_pose']
        position, _ = self.__dual_pose
        rx, ry, rz = position
        x, y, z = wx + rx, wy + ry, wz + rz
        
        return np.sqrt(x**2 + y**2 + z**2)
        
    def detections(self):
        return self.simGetDetectedMeshesDistances(self.__shadow_cfg['camera']['name'], 
                                                                           ImageType.Scene,
                                                                           vehicle_name=self.__shadow_cfg['name'])
        
    def random_pose(self, range_x : dict, range_y : dict, safe_range_x : dict, safe_range_y : dict, target : str):
        xmin, xmax = range_x
        ymin, ymax = range_y
        
        sxmin, sxmax = safe_range_x
        symin, symax = safe_range_y
        
        px = random_choice((xmin - sxmin, sxmin), (sxmax, sxmax + xmax))
        py = random_choice((ymin - symin, symin), (symax, symax + ymax))
        
        centroide_pose = self.get_object_pose_as_list(target)
        yaw = theta([px, py], centroide_pose[:2])
        pose = [px, py, self.__home.position.z_val, 0, 0, yaw]

        self._pose(pose)
        
        return pose

    @singledispatchmethod    
    def random_base_pose(self, range_x : dict, range_y : dict, safe_range_x : dict, safe_range_y : dict, target : str):
        pov_pose = self.random_pose(range_x, range_y, safe_range_x, safe_range_y, target)  
        position = pov_pose[:3]
        orientation = pov_pose[3:]      
        
        position[1] -= 13
        position[2] = self.__vehicle_cfg['base']['altitude']
        orientation[2:] = [np.deg2rad(-90), 0]
        pose = position + orientation
        self.set_object_pose(pose, self.__vehicle_cfg['base']['name'])
        
        np_position = np.array(self.__shadow_cfg['global_pose'][:3]) + np.array(position)
        pose = np_position.tolist() + orientation
        self.set_object_pose(pose, self.__shadow_cfg['base']['name'])
        
        return pov_pose
    
    @random_base_pose.register
    def random_base_pose(self, pov_pose : list):
        position = pov_pose[:3]
        orientation = pov_pose[3:]      
        
        position[1] -= 13
        position[2] = self.__vehicle_cfg['base']['altitude']
        orientation = [0, np.deg2rad(-90), 0]
        
        pose = position + orientation
        self.set_object_pose(pose, self.__vehicle_cfg['base']['name'])
        
        np_position = np.array(self.__shadow_cfg['global_pose'][:3]) + np.array(position)
        pose = np_position.tolist() + orientation
        self.set_object_pose(pose, self.__shadow_cfg['base']['name'])
        
        return pov_pose