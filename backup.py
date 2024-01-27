#!/usr/bin/env python3

import rospy
import os
import random
import time
import cv2
import rospkg
import rospy
import datetime

import numpy as np

from typing import List

from airsim_base import MultirotorClient
from airsim_base import Pose, ImageRequest, ImageType, ImageResponse
from airsim_base import to_eularian_angles, write_file

from std_msgs.msg import String

from utils import pose, angular_distance

class Ue4Briedge:
    """Starts communication's Engine.

    Args:
        HOST: 
            -For set ip address from host, use ipconfig result's on host
            -For set ip address from docker network, use a ip from ping result's between containers on host
            -For set ip address from WSL, os.environ['WSL_HOST_IP'] on host.
    """        
    
    def __init__(self, project_name : str) -> None:
        self.client = MultirotorClient(os.environ['UE4_IP'])
        self.client.confirmConnection()
        rospy.logwarn(f"\nConnection: {self.client.ping()}")     
        self.spawn_resources = _SpawResources(self.client)
        self.debbug_resources = _DebbugResources(self.client, project_name)

    def restart(self) -> None:
        """
        Reset the ue4 client conection.
        """        
        rospy.logwarn(f"\nRestart Connection: {self.client.ping()}")
        self.client.reset()
    

class _SpawResources:
    
    def __init__(self, ue4_connection : MultirotorClient):
        self.__conn = ue4_connection
        self.__pub_info = rospy.Publisher("simulation_info", String, queue_size=10)
        
    def _get_object_pose(self, object_name : str) -> Pose:
        """Returns a scene element pose.

        Args:
            object_name (str): Element's name.

        Returns:
            Pose: AirSim native pose.
        """        
        pose = self.__conn.simGetObjectPose(object_name)
        return pose
    
    def get_vehicle_pose(self, vehicle_name : str) -> Pose:
        """Returns a scene element pose.

        Args:
            vehicle_name (str): Vehicle's name.

        Returns:
            Pose: AirSim native pose.
        """        
        pose = self.__conn.simGetVehiclePose(vehicle_name)
        return pose
            
    def _set_vehicle_pose(self, vehicle_name : str, position : tuple, eularian_orientation : tuple, debbug : bool = False) -> None:
        """Define a new pose at one vehicle.

        Args:
            vehicle_name (str): The same was defined on settings.json .
            position (tuple): (x,y,z).
            eularian_orientation (tuple): (pitch, roll, yaw).
        """           
        pose_ = pose(position, eularian_orientation)
        self.__conn.simSetVehiclePose(pose_, True, vehicle_name)
        
        if debbug:
            info = f"New Vehicle pose was defined: {vehicle_name} - [{position}, {eularian_orientation}]"
            self.__pub_info.publish(info) 
            
    def _set_object_pose(self, object_name : str, position : tuple, eularian_orientation : tuple, debbug : bool = False) -> None:
        """Define a new pose at one scene object.

        Args:
            object_name (str): Element's name.
            position (tuple): (x,y,z).
            eularian_orientation (tuple): (pitch, roll, yaw).
        """           
        pose_ = pose(position, eularian_orientation)
        self.__conn.simSetObjectPose(pose_, True, object_name)
        
        if debbug:
            info = f"New Object pose was defined: {object_name} - [{position}, {eularian_orientation}]"
            self.__pub_info.publish(info) 

    def _oriented_target_vision_to_vehicle(self, vehicle_name : str, target : str, radius : float, dist : float, theta : float):
        """Define a random vehicle pose at target based on radius range and secure distance to avoid collision.

        Args:
            vehicle_name (str) : Vehicle's settings name.
            target (str) : Name of target's object.
            radius (float): Range around the target to define vehicle's pose.
            dist (float): Secure distance to spawn.
            theta (float): Angle (0, 2pi) to direction of radius.

        Returns:
            tuple: New vehicle pose (x, y) and orientation (phi). 
        """ 
       
        _, _, current_phi = self.get_current_eularian_vehicle_angles(vehicle_name)
        pose_object = self._get_object_pose(target)

        ox = pose_object.position.x_val
        oy = pose_object.position.y_val

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        px = ox + (dist*sin_theta + radius*sin_theta)
        py = oy + (dist*cos_theta + radius*cos_theta)
        phi = current_phi + angular_distance((px, py, 0), (ox, oy, 0), degrees=True)
        return (px, py, phi)

    def _air_random_oriented_target_vision_to_vehicle(self, vehicle_name : str, target : str, radius : float, dist : float) -> tuple:
        """Define a random vehicle pose at target based on radius range and secure distance to avoid collision.

        Args:
            radius (float): Range around the target to define vehicle's pose.
            dist (float): Secure distance to spawn.
            target (str): Scene object name.

        Returns:
            tuple: (new x, new y, new yaw)
        """        
        
        theta = random.random() * 2*np.pi
        rand_range = np.arange(0, radius, 0.5)
        r = random.choice(rand_range)

        px, py, phi = self._oriented_target_vision_to_vehicle(vehicle_name, target, r, dist, theta)

        return (px, py, phi)
          
    def get_current_eularian_vehicle_angles(self, vehicle_name : str) -> tuple:
        """Get current eularian angles from vehicle.

        Args:
            vehicle_name (str): The same was defined on settings.json .

        Returns:
            tuple: (pitch, roll, yaw)
        """        
        pose = self.get_vehicle_pose(vehicle_name)

        pitch, roll, yaw = to_eularian_angles(pose.orientation)
        return (pitch, roll, yaw)
    

    def set_oriented_target_vision_to_vehicle(self, vehicle_name : str, target : str, radius : float, dist : float, theta : float) -> None:
        """Define a random vehicle pose at target based on radius range and secure distance to avoid collision.

        Args:
            vehicle_name (str) : Vehicle's settings name.
            target (str) : Name of target's object.
            radius (float): Range around the target to define vehicle's pose.
            dist (float): Secure distance to spawn.
            theta (float): Angle (0, 2pi) to direction of radius.

        Returns:
            tuple: New vehicle pose (x, y) and orientation (phi). 
        """ 

        pose = self.get_vehicle_pose(vehicle_name)
        z = pose.position.z_val

        x, y, phi = self._oriented_target_vision_to_vehicle(vehicle_name, target, radius, dist, theta)
        position = (x, y, z)
        eularian_orientation = (0, 0, phi)
        self._set_vehicle_pose(vehicle_name, position, eularian_orientation)

    def set_air_random_oriented_target_vision_to_vehicle(self, vehicle_name : str, target : str, radius : float, dist : float, theta : float) -> None:
        """Define a random vehicle pose at target based on radius range and secure distance to avoid collision.

        Args:
            vehicle_name (str) : Vehicle's settings name.
            target (str) : Name of target's object.
            radius (float): Range around the target to define vehicle's pose.
            dist (float): Secure distance to spawn.
            theta (float): Angle (0, 2pi) to direction of radius.

        Returns:
            tuple: New vehicle pose (x, y) and orientation (phi). 
        """ 

        pose = self.get_vehicle_pose(vehicle_name)
        z = pose.position.z_val

        x, y, phi = self._air_random_oriented_target_vision_to_vehicle(vehicle_name, target, radius, dist, theta)
        position = (x, y, z)
        eularian_orientation = (0, 0, phi)
        self._set_vehicle_pose(vehicle_name, position, eularian_orientation)

class _DebbugResources:
    def __init__(self, ue4_connection : MultirotorClient, package_name : str):
        self.__conn = ue4_connection
        self.__package_name = package_name
        self.usual_cams = {0 : "rgb", 1 : 'depth', 5 : 'segmentation'}
        self.rospack = rospkg.RosPack()
        
    def save_assync_images(self, camera_names : List[str], image_types : List[ImageType]):
        for cam, image in zip(camera_names, image_types):
            response = self.__conn.simGetImages([ImageRequest(cam, image, False, False)], 'Hydrone')[0]
            filename = self.rospack.get_path(self.__package_name) + '/debbug/images/' +\
                       self.usual_cams[image] + '/' + str(datetime.datetime.now().strftime('%d-%m-%Y--%H:%M:%S'))
            
            rospy.logwarn(f"filename -> {filename}")
            self.save_image(response, filename)

    def save_image(self, response : ImageResponse, filename : str):
        if response.pixels_as_float:
            # airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            depth = np.array(response.image_data_float, dtype=np.float64)
            depth = depth.reshape((response.height, response.width, -1))
            depth = np.array(depth * 255, dtype=np.uint8)
            # save pic
            cv2.imwrite(os.path.normpath(filename + '.png'), depth)

        elif response.compress: #png format
            write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)

        else: #uncompressed array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png


class ComputerVision:
    def __init__(self):
        pass
