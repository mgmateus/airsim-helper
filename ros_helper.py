import copy
import cv2
import message_filters
import rospy
import os

import numpy as np
import open3d as o3d

from functools import singledispatchmethod
from numpy.typing import NDArray
from typing import List

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped

from airsim_ros_pkgs.msg import GimbalAngleEulerCmd
from airsim_ros_pkgs.srv import Takeoff, Land

from cv_bridge import CvBridge, CvBridgeError

from airsim_base.client import MultirotorClient
from airsim_base.types import Vector3r, Quaternionr, Pose, ImageType, ImageRequest
from airsim_base.utils import to_quaternion, to_eularian_angles, random_choice, theta, string_to_uint8_array


class DictToClass:
    def __init__(self, dictionary):
        self.update(dictionary)
    
    def __repr__(self):
        return str(self.__dict__)
    
    def __getitem__(self, index):
        return self.__dict__[index]
    
    def update(self, dictionary : dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Se o valor for um dicionário, criar um atributo de classe recursivamente
                setattr(self, key, DictToClass(value))
            else:
                # Se o valor não for um dicionário, criar um atributo diretamente
                setattr(self, key, value)

        return self
    


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
    
    @staticmethod
    def request(camera_name = ""):
        return {'rgb' : [ImageRequest(camera_name or "0", ImageType.Scene, False, False)], 
                'stereo': [ImageRequest(camera_name or "0", ImageType.Scene, False, False), 
                           ImageRequest(camera_name or "1", ImageType.DepthPlanar, True)]}

    @classmethod
    def wait_for_ros_msg(cls, topic : str, msg_type : np.uint8):
        data = None
        while data is None :
            try:
                if msg_type is np.uint8:
                    data = rospy.wait_for_message(topic, Image, timeout=5)
                    data = cls.image_transport(data)
                    data = data if data.size else None

                else:
                    data = rospy.wait_for_message(topic, TransformStamped, timeout=5) 
                    data = cls.tf_to_list(data)   
            except:
                pass
        
        return data  

        
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
    def gimbal(self, data):
        if isinstance(data, list):
            roll, pitch, yaw = data
            q = to_quaternion(roll=roll, pitch=pitch, yaw=yaw)
        else:
            q = data
        self.__gimbal *= q
    
    @property
    def gimbal_reset(self):
        self.__gimbal = to_quaternion(0,0,0)
        return self.__gimbal
    
    @property
    def intrinsic(self):
        return self.__intrinsic
    
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
        # image_data = None
        # cv_image = None
        # while image_data is None :
        #     try:
                
        #         image_data = rospy.wait_for_message("/airsim_node/"+vehicle_name+"/"+camera_name+"/Scene", Image, timeout=5)
        #         cv_image = self.image_transport(image_data)  
        #         cv_image = cv_image if cv_image.size else None
        #     except:
        #         pass
        
        # return cv_image
        return self.wait_for_ros_msg("/airsim_node/"+vehicle_name+"/"+camera_name+"/Scene")
    
            
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
    
    def call(self, image_key, vehicle_name : str ='', camera_name : str =''):
    
        def responses(key):
            key = key.split('_')[0] if key.count('_') else key
            request = self.request(camera_name)[key]
            # print(request)
            responses = self.simGetImages(request, vehicle_name)
            # print(responses)
            pre_call = []
            for response in responses:
                if response.image_data_uint8:
                    img1d = string_to_uint8_array(response.image_data_uint8)
                    image = (
                        np.flipud(img1d.reshape(response.height, response.width, 3)) 
                        if response.height * response.width * 3 == img1d.size 
                        else np.zeros((response.width, response.height, 3), dtype=np.uint8)
                        )
                else:
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    image = img1d.reshape(response.height, response.width)
                    
                pre_call.append(image)

            return pre_call
        
        def occupancy(raw_depth):
            raw_depth = copy.deepcopy(raw_depth) * 100
            raw_depth = np.clip(raw_depth, 0, 20000)

            height, width = raw_depth.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))

            u = u.flatten()
            v = v.flatten()
            depth = raw_depth.flatten()

            image_width, image_height = (672, 376)
            fov_rad = 90 * np.pi/180
            fd = (image_width/2.0) / np.tan(fov_rad/2.0)
            print(fd)
            fx, fy = fd, fd
            cx, cy = image_width/2, image_height/2
            
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            
            points = np.vstack((-y, -z, -x)).T

            valid_indices = depth > 0
            return points[valid_indices]
        
        calls = None
        check = True
        while True: 
            calls = responses(image_key)
            for called_image in calls:
                if called_image.size == 0:
                    check = False
                    break
                check = True
                
            if check:
                break
                    
        if image_key.endswith('occupancy'):
            calls.append(occupancy(calls[1]))
        
        pose = self.simGetCameraInfo(camera_name, vehicle_name).pose
        TF = [*pose.position.to_numpy_array(), *pose.orientation.to_numpy_array()]

        calls.append(TF)
        return calls
         
                              
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
        # print(f"---- pose of object: {pose}")
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
        

class PointOfViewTwins(RotorPyROS):
    def __init__(self, ip: str, vehicle_cfg : DictToClass, twin_cfg : DictToClass, observation_type : str):        
        super().__init__(ip, vehicle_cfg.camera.dim, vehicle_cfg.camera.fov)
        
        self.__vehicle_cfg = vehicle_cfg
        self.__twin_cfg = twin_cfg
        self.__obs_type = observation_type
        self.__home = self.pose_from_positon_euler_list(vehicle_cfg.start_pose)
        self.__twin_pose = self.pose_from_positon_euler_list(twin_cfg.start_pose)
        self.__mode = 'Py'
        
        self._start()
            
    @property
    def vehicle_and_camera_name(self):
        vehicle_name = self.__vehicle_cfg.name
        camera_name = self.__vehicle_cfg.camera.name
        return vehicle_name, camera_name
    
    @property
    def home(self):
        home = self.__home
        return self.pose_to_position_quat(home)
    
    @home.setter
    def home(self, new_pose : list):
        pose = self.pose_from_positon_euler_list(new_pose)
        self.__home = copy.deepcopy(pose)
        self.__twin_pose = copy.deepcopy(pose)   
    
    @property
    def altitude(self):
        altitude = self.__twin_pose.position.z_val
        return -1 * altitude
    
    @property
    def altitude_min(self):
        altitude = self.__vehicle_cfg.altitude
        altitude_min = altitude if type(altitude) == float else altitude[0]
        return altitude_min
    
    @property
    def altitude_max(self):
        altitude_max = self.__vehicle_cfg.altitude[1]
        return altitude_max
    
    @property
    def rgb(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            rgb = None
            while not isinstance(rgb, np.ndarray):
                rgb = self.rgb_image(vehicle_name, camera_name)
            tf = self.tf(vehicle_name, camera_name)
        else:
            rgb, tf = self.call('rgb', vehicle_name, camera_name)
        return {'rgb' : rgb, 'tf' : tf}
    
    @property
    def depth(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            depth = self.depth_image(vehicle_name, camera_name)
            tf = self.tf(vehicle_name, camera_name)
        else:
            depth, tf = self.call('depth',vehicle_name, camera_name)
        return {'depth' : depth, 
                'tf' : tf}
    
    @property
    def segmentation(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            seg = self.segmentation_image(vehicle_name, camera_name)
            tf = self.tf(vehicle_name, camera_name)
        else:
            seg, tf = self.call('segmentation',vehicle_name, camera_name)
        return {'segmentation' : seg, 
                'tf' : tf}
    
    @property
    def stereo(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            rgb = self.rgb_image(vehicle_name, camera_name)
            depth = self.depth_image(vehicle_name, camera_name)
            tf = self.tf(vehicle_name, camera_name)
        else:
            rgb, depth, tf = self.call('stereo',vehicle_name, camera_name)
        return {'rgb' : rgb, 
                'depth' : depth, 
                'tf' : tf}
    
    @property
    def panoptic(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            rgb = self.rgb_image(vehicle_name, camera_name)
            depth = self.depth_image(vehicle_name, camera_name)
            segmentation = self.segmentation_image(vehicle_name, camera_name)
            tf = self.tf(vehicle_name, camera_name)
        else:
            rgb, depth, segmentation, tf = self.call('panoptic',vehicle_name, camera_name)
        return {'rgb' : rgb, 
                'depth' : depth, 
                'segmentation' : segmentation,
                'tf' : tf}

    @property
    def stereo_occupancy(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            rgb = self.rgb_image(vehicle_name, camera_name)
            depth = self.depth_image(vehicle_name, camera_name)
            point_cloud = self.point_cloud(rgb[...,::-1].copy(), depth.astype(np.float32))
            tf = self.tf(vehicle_name, camera_name)
        else:
            rgb, depth, point_cloud, tf = self.call('stereo_occupancy',vehicle_name, camera_name)
        return {'rgb' : rgb, 
                'depth' : depth, 
                'point_cloud' : point_cloud,
                'tf' : tf}
            
    @property
    def panoptic_occupancy(self):
        vehicle_name, camera_name = self.vehicle_and_camera_name
        if self.__mode.endswith('ROS'):
            rgb = self.rgb_image(vehicle_name, camera_name)
            depth = self.depth_image(vehicle_name, camera_name)
            segmentation = self.segmentation_image(vehicle_name, camera_name)
            point_cloud = self.point_cloud(rgb[...,::-1].copy(), depth.astype(np.float32))
            tf = self.tf(vehicle_name, camera_name)
        else:
            rgb, depth, segmentation, point_cloud, tf = self.call('panoptic_occupancy',vehicle_name, camera_name)
        return {'rgb' : rgb, 
                'depth' : depth, 
                'segmentation' : segmentation,
                'point_cloud' : point_cloud,
                'tf' : tf}

    
    @property
    def observation(self) -> NDArray:
        return self.__getattribute__(self.__obs_type)
    
    
    def _start(self):
        vehicle_name = self.__vehicle_cfg.name
        twin_name = self.__twin_cfg.name
        twin_camera_name = self.__twin_cfg.camera.name
        self.start(vehicle_name)
        self.start(twin_name)
        
        self.simSetDetectionFilterRadius(twin_camera_name, 
                                        ImageType.Scene, 200 * 100, 
                                        vehicle_name=twin_name)
        
    def _gimbal(self, pitch : float, deg2rad : bool = True, reset= False):
        if reset:
            pose = Pose(Vector3r(0.3,0,0.3), self.gimbal_reset)
        else:
            self.gimbal = [0, np.deg2rad(pitch), 0] if deg2rad else [0, pitch, 0]
            pose = Pose(Vector3r(0.3,0,0.3), self.gimbal)

        vehicle_name = self.__vehicle_cfg.name
        vehicle_camera_name = self.__vehicle_cfg.camera.name
        twin_name = self.__twin_cfg.name
        twin_camera_name = self.__twin_cfg.camera.name

        self.simSetCameraPose( vehicle_camera_name, pose, vehicle_name)
        self.simSetCameraPose(twin_camera_name, pose, twin_name)
        
        return True
        
    def _pose(self, new_pose : list):
        pose = self.pose_from_positon_euler_list(new_pose)
        twin_name = self.__twin_cfg.name
        self.__twin_pose.position += pose.position
        self.__twin_pose.orientation *= pose.orientation
        pose = self.__twin_pose
        
        self.simSetVehiclePose(pose, True)
        self.simSetVehiclePose(pose, True, vehicle_name=twin_name)
        
        return True
    
    def go_home(self):
        _, rpitch, _ = to_eularian_angles(self.gimbal)
        home = self.__home
        twin_name = self.__twin_cfg.name

        self._gimbal(0, reset=True)
        self.simSetVehiclePose(home, True)
        self.simSetVehiclePose(home, True, vehicle_name=twin_name)
        self.__twin_pose = copy.deepcopy(home)
        return True
        
    def next_point_of_view(self, five_DoF : NDArray):
        px, py, pz, yaw, gimbal_pitch= five_DoF
        
        self._pose([px, py, pz, 0, 0, yaw])
        self._gimbal(gimbal_pitch)
        
        return True
    
    def set_detection(self, detect : str):
        twin_name = self.__twin_cfg.name
        twin_camera_name = self.__twin_cfg.camera.name
        self.simAddDetectionFilterMeshName(twin_camera_name, 
                                            ImageType.Scene, f"{detect}*", 
                                            vehicle_name=twin_name)
        
    def detection_distance(self):
        wx, wy, wz, _, _, _ = self.__twin_cfg['global_pose']
        position, _ = self.__twin_pose
        rx, ry, rz = position
        x, y, z = wx + rx, wy + ry, wz + rz
        
        return np.sqrt(x**2 + y**2 + z**2)
        
    def detections(self):
        twin_name = self.__twin_cfg.name
        twin_camera_name = self.__twin_cfg.camera.name
        return self.simGetDetectedMeshesDistances(twin_camera_name, 
                                                    ImageType.Scene,
                                                    vehicle_name=twin_name)
        
    def _random_pose(self, range_x : dict, range_y : dict, safe_range_x : 
                    dict, safe_range_y : dict, target : str):
        xmin, xmax = range_x
        ymin, ymax = range_y
        
        sxmin, sxmax = safe_range_x
        symin, symax = safe_range_y
        
        px = random_choice((xmin - sxmin, sxmin), (sxmax, sxmax + xmax))
        py = random_choice((ymin - symin, symin), (symax, symax + ymax))
        
        centroide_pose = self.get_object_pose_as_list(target)
        yaw = theta([px, py], centroide_pose[:2])
        pose = [px, py, self.__home.position.z_val, 0, 0, yaw]
        
        return pose
    
    @singledispatchmethod 
    def random_pose(self, range_x : dict, range_y : dict, safe_range_x : 
                    dict, safe_range_y : dict, target : str, spawn_heliport : bool):
        
        pov_pose = self._random_pose(range_x, range_y, safe_range_x, safe_range_y, target)  
        self._pose(pov_pose)

        # print(self.__vehicle_cfg.base, self.__twin_cfg.base)

        if spawn_heliport:
            position = pov_pose[:3]
            position[1] -= 13
            position[2] = self.__vehicle_cfg.base.altitude
            orientation = [0, np.deg2rad(-90), 0]
            pose = position + orientation
            # print(f"aqui----------------{position} {orientation}")
            self.set_object_pose(pose, self.__vehicle_cfg.base.name)
            
            np_position = np.array(self.__twin_cfg.global_pose[:3]) + np.array(position)
            pose = np_position.tolist() + orientation
            self.set_object_pose(pose, self.__twin_cfg.base.name)

        return pov_pose
            

    @random_pose.register
    def _(self, pov_pose : str, spawn_heliport : bool):  
        self._pose(pov_pose)

        if spawn_heliport:
            position = pov_pose[:3]
            orientation = pov_pose[3:]

            position[1] -= 13
            position[2] = self.__vehicle_cfg.base.altitude
            orientation[2:] = [np.deg2rad(-90), 0]
            pose = position + orientation
            self.set_object_pose(pose, self.__vehicle_cfg.base.name)
            
            np_position = np.array(self.__twin_cfg.global_pose[:3]) + np.array(position)
            pose = np_position.tolist() + orientation
            self.set_object_pose(pose, self.__twin_cfg.base.name)

        return pov_pose
    




def raw_pcd2(cv, camera_name, vehicle_name):
    caminfo = airsim.simGetCameraInfo(camera_name, vehicle_name)
    
    responses = cv.simGetImages([ImageRequest(camera_name, ImageType.DepthPerspective)], vehicle_name)
    response = responses[0]
    raw_depth = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), cv2.IMREAD_UNCHANGED)
    depth = cv2.cvtColor(raw_depth, cv2.COLOR_BGR2GRAY)
    pcd = cv2.reprojectImageTo3D(depth, np.array(caminfo.proj_mat.matrix))
    mask = np.isfinite(pcd).all(axis=2)  # Ensure all coordinates are finite
    valid_points = pcd[mask] 
    return valid_points, airsim_pose_to_open3d_transformation(caminfo.pose)

def tf(pose):
    position = pose.position
    orientation = pose.orientation
    x, y, z = position.x_val, position.y_val, position.z_val
    T = np.eye(4)
    T[:3,3] = [-y, z, x]
    
    qw, qx, qy, qz = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    R = np.eye(4)
    R[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((qw, -qy, qz, qx))
    
    C = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

    return R.T @ T

def o3d_pcd2(cv, camera_name, vehicle_name):
    obs = {'rgb' : ImageType.Scene, 'depth' : ImageType.DepthPlanar}
    caminfo = airsim.simGetCameraInfo(camera_name, vehicle_name)
    
    responses = cv.simGetImages([ImageRequest(camera_name, key) for key in obs.values()], vehicle_name)

    raw_rgb = cv2.imdecode(string_to_uint8_array(responses[obs['rgb']].image_data_uint8), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
    raw_depth = cv2.imdecode(np.frombuffer(responses[obs['depth']].image_data_uint8, np.uint8), cv2.IMREAD_UNCHANGED)
    depth = cv2.cvtColor(raw_depth, cv2.COLOR_BGR2GRAY)

    rgb_ = o3d.geometry.Image(np.asarray(rgb.copy()))
    depth_ = o3d.geometry.Image(depth.copy())
    tf_ = tf(caminfo.pose)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_, 
                                                                  depth_, 
                                                                  depth_scale=1.0, 
                                                                  depth_trunc=1000, 
                                                                  convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cv.intrinsic)
    print(pcd)

    return pcd, airsim_pose_to_open3d_transformation(caminfo.pose)

from scipy.spatial.transform import Rotation as R
def airsim_pose_to_open3d_transformation(pose):
    # Extract position and orientation from AirSim pose
    position = pose.position
    x, y ,z = position.x_val, position.y_val, position.z_val
    orientation = pose.orientation
    qx, qy, qz, qw = orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val

    # Position: AirSim (X, Y, Z) -> Open3D (X, -Z, Y)
    #translation = np.array([position.x_val, -position.z_val, position.y_val])
    #translation = np.array([position.x_val, position.y_val, position.z_val])
    translation = np.array([-y, z, x])

    # Orientation (quaternion): AirSim (X, Y, Z, W) -> Open3D (X, -Z, Y, W)
    #rotation = R.from_quat([orientation.x_val, -orientation.z_val, orientation.y_val, orientation.w_val])
    #rotation = R.from_quat([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
    rotation = R.from_quat([qy, qz, qx, qw])

    rotation_matrix = rotation.as_matrix() @ rx(np.deg2rad(90)) @ rx(np.deg2rad(90)) @ ry(np.deg2rad(-90)) # 3x3 rotation matrix

    # Construct the 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix  # Insert rotation
    transformation[:3, 3] = translation  # Insert translation


    return transformation





def raw_pcd3(cv, camera_name, vehicle_name):

    def rx(angle):
        return np.array([
            [1, 0, 0 , 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    def ry(angle):
        return  np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    def rz(angle):
        return  np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

     

    caminfo = airsim.simGetCameraInfo(camera_name, vehicle_name)
    
    responses = cv.simGetImages([ImageRequest(camera_name, ImageType.Scene, False, False),
                                 ImageRequest(camera_name, ImageType.DepthPlanar, True)], vehicle_name)

    raw_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    raw_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width) * 100

    raw_depth = np.clip(raw_depth, 0, 20000)

    height, width = raw_depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    u = u.flatten()
    v = v.flatten()
    depth = raw_depth.flatten()

    image_width, image_height = (376, 672)
    fov_rad = 90 * np.pi/180
    fd = (image_width/2.0) / np.tan(fov_rad/2.0)

    fx, fy = fd, fd
    cx, cy = image_width/2 - 0.5, image_height/2 - 0.5
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    points = np.vstack((x, y, z)).T

    valid_indices = depth > 0
    points = points[valid_indices]

    # Color the points using the RGB image
    colors = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0  # Normalize colors to range [0, 1]
    colors = colors[valid_indices]

    # Create Open3D Point Cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    
    tt = airsim_pose_to_open3d_transformation(caminfo.pose)
    point_cloud.transform(rz(np.deg2rad(-180)) )
    return point_cloud, tt


def register_point_clouds(pcds):
    """
    Registers multiple point clouds into a single point cloud.
    
    Parameters:
    pcds: List of Open3D PointCloud objects
    
    Returns:
    combined_point_cloud: Registered and combined point cloud
    """
    # Initialize the first point cloud as the reference
    pcd_combined = pcds[0]

    for i in range(1, len(pcds)):
        # Pre-process: Voxel downsampling and feature extraction
        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=1)
        pcd_combined_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        pcd_curr = pcds[i].voxel_down_sample(voxel_size=1)
        pcd_curr.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Registration using ICP (Iterative Closest Point)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_curr, pcd_combined_down, max_correspondence_distance=0.1,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        # Transform current point cloud
        pcd_curr.transform(reg_p2p.transformation)

        # Merge point clouds
        pcd_combined += pcd_curr
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=1)  # Merge and downsample to reduce redundancy

    return pcd_combined

def rx(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
def ry(angle):
        return  np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
def rz(angle):
        return  np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def icp(a, b):
    print(a, b)
    pcd_combined = a

    # Pre-process: Voxel downsampling and feature extraction
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=1)
    pcd_combined_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    pcd_curr = b.voxel_down_sample(voxel_size=1)
    pcd_curr.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Registration using ICP (Iterative Closest Point)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_curr, pcd_combined_down, max_correspondence_distance=0.1,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    # Transform current point cloud
    pcd_curr.transform(reg_p2p.transformation)

    # Merge point clouds
    pcd_combined += pcd_curr
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=1)  # Merge and downsample to reduce redundancy

    return pcd_combined

def airsim_conversion2(pose, points):
    correct_points = np.array([])
    orientation = pose.orientation
    q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                    [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                    [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
    
    position = pose.position
    for i in range(0, len(points), 3):
        xyz = points[i]

        
        corrected_x, corrected_y, corrected_z = rotation_matrix @ xyz
        final_x = corrected_x + position.x_val
        final_y = corrected_y + position.y_val
        final_z = corrected_z + position.z_val
        npoints = np.array([final_x, final_y, final_z])
        correct_points = npoints if not len(correct_points) else np.vstack((correct_points, npoints))

    return correct_points


    


def o3d_pcd(cv, camera_name, vehicle_name):
    caminfo = airsim.simGetCameraInfo(camera_name, vehicle_name)
    
    responses = cv.simGetImages([ImageRequest(camera_name, ImageType.Scene, False, False),
                                 ImageRequest(camera_name, ImageType.Scene, True)], vehicle_name)

    raw_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    raw_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width) * 100

    rgb_ = o3d.geometry.Image(cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB))
    depth_ = o3d.geometry.Image(raw_depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_, 
                                                            depth_, 
                                                            depth_scale=1.0, 
                                                            depth_trunc=10000, 
                                                            convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cv.intrinsic)    
    
    print('oi')

    pcd.transform(airsim_conversion(caminfo.pose))
    return pcd

def airsim_conversion(pose):
    orientation = pose.orientation
    q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                    [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                    [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
    
    position = pose.position
    x, y, z = position.x_val, position.y_val, position.z_val

    T = np.eye(4)
    T[:3,3] = [-y, -z, -x]

    R = np.eye(4)
    # R[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((q0, q2, q3, q1))
    R[:3,:3] = rotation_matrix

    C = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

    return R.T @ T @ C

def raw_pcd(cv, camera_name, vehicle_name):
    
    caminfo = airsim.simGetCameraInfo(camera_name, vehicle_name)
    
    responses = cv.simGetImages([ImageRequest(camera_name, ImageType.Scene, False, False),
                                 ImageRequest(camera_name, ImageType.DepthPlanar, True)], vehicle_name)

    raw_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    raw_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width) * 100

    raw_depth = np.clip(raw_depth, 0, 20000)

    height, width = raw_depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    u = u.flatten()
    v = v.flatten()
    depth = raw_depth.flatten()

    image_width, image_height = (672, 376)
    fov_rad = 90 * np.pi/180
    fd = (image_width/2.0) / np.tan(fov_rad/2.0)
    print(fd)
    fx, fy = fd, fd
    cx, cy = image_width/2, image_height/2
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    points = np.vstack((-y, -z, -x)).T

    valid_indices = depth > 0
    points = points[valid_indices]
    # points = airsim_conversion(caminfo.pose, points)
    # points = np.hstack((points, np.ones((points.shape[0], 1))))

    # transformation_matrix = np.array([
    #     [0, 1, 0],
    #     [0, 0, -1],
    #     [-1, 0, 0]
    # ])
        
    #print(np.array(caminfo.proj_mat.matrix), np.linalg.inv(np.array(caminfo.proj_mat.matrix)))
    #points = points @ C

    # print(points.T.shape)
    orientation = caminfo.pose.orientation
    q0, q1, q2, q3 = -orientation.y_val, -orientation.z_val, -orientation.x_val, orientation.w_val
    rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                    [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                    [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
    position = caminfo.pose.position
    x, y, z = position.x_val, position.y_val, position.z_val
    t = np.array([2*y, x, x])
    
    points = points @ rotation_matrix

    # Color the points using the RGB image
    colors = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0  # Normalize colors to range [0, 1]
    colors = colors[valid_indices]

    return points, colors

def raw_pcd2(cv, camera_name, vehicle_name):
    caminfo = cv.simGetCameraInfo(camera_name, vehicle_name)
    projectionMatrix = np.array([[10.501202762, 0.000000000, 0.000000000, 0.000000000],
                              [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                              [0.000000000, 0.000000000, 1.00000000, 100.00000000],
                              [0.000000000, 0.000000000, -10.0000000, 0.000000000]])
    
    responses = cv.simGetImages([ImageRequest(camera_name, ImageType.Scene, False, False),
                                 ImageRequest(camera_name, ImageType.DepthPerspective, True)], vehicle_name)
    
    raw_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    raw_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width)

    pcd = cv2.reprojectImageTo3D(raw_depth, np.array(caminfo.proj_mat.matrix))
    # pcd = cv2.reprojectImageTo3D(raw_depth, np.array(projectionMatrix))
    mask = np.isfinite(pcd).all(axis=2)

    valid_points = pcd[mask] 

    colors = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0  # Normalize colors to range [0, 1]
    # colors = colors[mask]

    # raw_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    # raw_depth = cv2.imdecode(responses[1].image_data_float, cv2.IMREAD_UNCHANGED)
    # print(raw_depth)
    # depth = cv2.cvtColor(raw_depth, cv2.COLOR_BGR2GRAY)
    # pcd = cv2.reprojectImageTo3D(depth, np.array(projectionMatrix))
    # mask = np.isfinite(pcd).all(axis=2)  # Ensure all coordinates are finite
    # valid_points = pcd[mask] 


    return valid_points, colors, airsim_conversion(caminfo.pose)



if __name__ == "__main__":
    

    ip = '172.19.0.3'
    camera_name = "stereo"
    vehicle_name = "Hydrone"
    twin_name = "Twin"
    airsim = RotorPyROS(ip, (672, 376), 90)

    agents = ['curl-rgb-air', 'curl-rgb-underwater']
    twins = [agent+'-twin' for agent in agents]

    added_agents = airsim.listVehicles()
    agent = 'agent-curl'
    for agent, twin in zip(agents, twins):
        if not agent in added_agents and not twin in added_agents:
            airsim.simAddVehicle(agent, 'simpleflight', airsim.simGetVehiclePose(vehicle_name))
            airsim.simAddVehicle(twin, 'simpleflight', airsim.simGetVehiclePose(twin_name))
    
    y = 0
    t = 0

    resolution = 0.01
    print(np.array([1, 2, 3]).dtype)
    for i in range(20):

        y += 10
        t-=20
        new_pose = [10, y, -20, 0, 0 ,np.deg2rad(t)]
        airsim.set_pose(new_pose, vehicle_name=vehicle_name)
        c = airsim.call('stereo_occupancy', vehicle_name, camera_name)
        pcd, _, _ = raw_pcd2(airsim, camera_name, vehicle_name)
        
        
    
    # position = airsim.get_object_pose('target').position
    # xyz = [500, 100, 500]
    # f = os.path.join(os.getcwd(), "map.binvox")

    # print(airsim.simCreateVoxelGrid(position, *xyz, 0.5, "/home/ue4/Documents/AirSim/map.binvox"))