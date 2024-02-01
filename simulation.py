#!/usr/bin/env python3

import os
import pprint
import json
import time
import itertools

import numpy as np

from typing import List, Tuple
from threading import Thread

from airsim_base.client import MultirotorClient, VehicleClient
from airsim_base.types import Vector3r, DrivetrainType, YawMode, Pose
from airsim_base.utils import to_quaternion, to_eularian_angles

from utils import angular3D_difference, angular2D_difference

class _Ue4Briedge:
    """Starts communication's Engine.

    Args:
        HOST: 
            -For set ip address from host, use ipconfig result's on host
            -For set ip address from docker network, use a ip from ping result's between containers on host
            -For set ip address from WSL, os.environ['WSL_HOST_IP'] on host.
    """ 
    def __init__(self):
        ip = os.environ['UE4_IP']
        self.__client = MultirotorClient(ip)
        self.__client.confirmConnection()

    @property
    def client(self):
        return self.__client

    def reset(self):
        self.__client.reset()

class _Debbug:
    pp = pprint.PrettyPrinter(indent=4)
    restart_conn = "Restart Connection : "

    def __init__(self, client) -> None:
        self.__client = client

    def log_restart_connection(self):
        full_info = f"{self.restart_conn} {self.__client.ping()}"
        print(full_info)

class Gimbal:
    def __init__(self, cams : List[str]):
        """Gimbal

        Args:
            cams (List[str]): camera names in settings.json
        """        
        self.__cams = cams
        
    @property
    def cams(self):
        return self.__cams
    
    def discretize_rotation(self, camera : Pose, pitch : float, roll : float, yaw : float) -> Tuple:
        """A discretization for angle difference

        Args:
            camera (Pose): Current camera pose
            pitch (float): pitch in randians
            roll (float): roll in radians
            yaw (float): yaw in radians

        Returns:
            Tuple: discrezed positions for each DoF
        """        
        pitch_, roll_, yaw_ = to_eularian_angles(camera.orientation)
                
        d_pitch = np.arange(pitch_, pitch, 0.01) 
        d_roll = np.arange(roll_, roll, 0.01) 
        d_yaw = np.arange(yaw_, yaw, 0.01) 
        
        max_ = max(len(d_pitch), len(d_roll), len(d_yaw))        
        
        d_pitch = np.zeros(max_) if d_pitch is None else d_pitch
        d_roll = np.zeros(max_) if d_roll is None else d_roll
        d_yaw = np.zeros(max_) if d_yaw is None else d_yaw
        
        return d_pitch, d_roll, d_yaw
    
    def rotation(self, client : MultirotorClient, vehicle_position : Vector3r, camera_orientation : Vector3r):
        """Apply rotations for each camera

        Args:
            client (VehicleClient): vehicle type connection
            vehicle_name (str): vehicle name in settings.json
            pitch (float): pitch in randians
            roll (float): roll in radians
            yaw (float): yaw in radians
        """
        pitch, roll, yaw = camera_orientation        
        xv, yv, zv = vehicle_position

        camera_pose = client.simGetCameraInfo(self.__cams[0]).pose
        d_pitch, d_roll, d_yaw = self.discretize_rotation(camera_pose, pitch, roll, yaw)

        adjustments = list(itertools.zip_longest(d_pitch, d_roll, d_yaw, fillvalue= 0))
        for adjust in adjustments:
            p, r, y = adjust
            pose = Pose(Vector3r(xv, yv, -zv), to_quaternion(p, r, y))
            time.sleep(0.01)
            for cam in self.__cams:
                client.simSetCameraPose(cam, pose)
                
        
        
            
        

class ClientController(_Ue4Briedge, _Debbug):
    def __init__(self) -> None:
        _Ue4Briedge.__init__(self)
        _Debbug.__init__(self, self.client)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.__vehicle_name = list(json.loads(self.client.getSettingsString())['Vehicles'].keys())[0]
        
        cams = list(json.loads(self.client.getSettingsString())['Vehicles'][self.__vehicle_name]['Cameras'].keys())
        self.__cams = cams
        self.__gimbal = Gimbal(cams)
        
        self.__target = None
        
    @property
    def vehicle_name(self):
        return self.__vehicle_name
    
    @vehicle_name.setter
    def vehicle_name(self, name):
        self.__vehicle_name = name
        
    @property
    def target(self):
        return self.__target
    
    @target.setter
    def target(self, name):
        self.__target = name
        

    def action_move(self, x : float, y : float, z : float, pitch : float, yaw : float) -> bool:
        target = Vector3r(x, y, z)
        orientation = Vector3r(pitch, 0, yaw)
        gimbal = Thread(target= self.__gimbal.rotation, args=(self.client, self.__vehicle_name, target, orientation, ))
        gimbal.start()
        
        self.client.moveToPositionAsync(x, y, z, 2, drivetrain=DrivetrainType.MaxDegreeOfFreedom, yaw_mode=YawMode(False, yaw))
        
    def look_to_target(self, next_position : Vector3r, target_name : str):
        rtk_v = self.client.getMultirotorState().kinematics_estimated
        rtk_t = self.client.simGetObjectPose(target_name)

        pitch = angular3D_difference(rtk_v.position, rtk_t.position)
        yaw = angular2D_difference(rtk_v.orientation, rtk_v.position, rtk_t.position)
        p, _, y = to_eularian_angles(rtk_v.orientation) 
        yaw += y
        pitch += p
        distance = ( (rtk_v.position.x_val-next_position.x_val)**2 + (rtk_v.position.y_val-next_position.y_val)**2 +\
            (rtk_v.position.z_val - (-next_position.z_val)**2 ))**0.5
                
        while distance >= 0.45:
            rtk_v = self.client.getMultirotorState().kinematics_estimated
            pitch = angular3D_difference(rtk_v.position, rtk_t.position)
            yaw = angular2D_difference(rtk_v.orientation, rtk_v.position, rtk_t.position)
            
            yaw = 1.5*yaw
            camera_pose = Pose()
            camera_pose.position = rtk_v.position
            camera_pose.position.z_val = -camera_pose.position.z_val
            camera_pose.orientation = to_quaternion(pitch, 0, yaw)
            
            for camera in self.__cams:
                self.client.simSetCameraPose(camera, camera_pose)
            
            distance = ( (rtk_v.position.x_val-next_position.x_val)**2 + (rtk_v.position.y_val-next_position.y_val)**2 +\
            (rtk_v.position.z_val- (-next_position.z_val))**2 )**0.5
            
            
    def move_look_to_target(self, next_position : Vector3r, target_name : str):
        x, y, z = next_position
        gimbal = Thread(target=self.look_to_target, args=(next_position, target_name, ))
        gimbal.start()
        self.client.moveToPositionAsync(x, y, z, 2, drivetrain=DrivetrainType.MaxDegreeOfFreedom)
        
                


if __name__ == "__main__":
    c = ClientController()
    c.client.takeoffAsync().join()

    pose = Vector3r(30, 10, -30)
    c.move_look_to_target(pose, 'eolic')
    
    # pose = Vector3r(30, 10, -30)
    # t = Thread(target=c.move_look_to_target, args=(pose, 'centroide', ))
    # t.start()
    # t.join()
    
    # pose = Vector3r(30, 80, -30)
    # t = Thread(target=c.move_look_to_target, args=(pose, 'centroide', ))
    # t.start()
    # t.join()
    # print(c.vehicle_name)
    # c.client.takeoffAsync().join()

    # pv = c.client.simGetVehiclePose('Hydrone')
    # xv, yv, zv = pv.position

    # po = c.client.simGetObjectPose('centroide')
    # xo, yo, zo = po.position
    
    
    # pitch = angular3D_difference(pv.position, po.position)
    # yaw = angular2D_difference(pv.position, po.position)
    # p, _, y = to_eularian_angles(pv.orientation) 
    # yaw += y
    # pitch += p
    # c.action_move(10, 10, -10, pitch, yaw)
    
    
    # pv = c.client.simGetVehiclePose('Hydrone')
    # xv, yv, zv = pv.position

    # po = c.client.simGetObjectPose('centroide')
    # xo, yo, zo = po.position
    
    
    # pitch = angular3D_difference(pv.position, po.position)
    # yaw = angular2D_difference(pv.position, po.position)
    # p, _, y = to_eularian_angles(pv.orientation) 
    # yaw += y
    # pitch += p
    # c.action_move(-30, -90, -30, pitch, yaw)
    
    