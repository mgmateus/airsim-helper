#!/usr/bin/env python3

import time
import copy
import pprint

import numpy as np

from threading import Thread
import concurrent.futures


from typing import List, Tuple, NewType
from numpy.typing import NDArray


from airsim_base.client import MultirotorClient, VehicleClient
from airsim_base.types import Vector3r, KinematicsState, ImageType, ImageRequest, ImageResponse


from utils import transform_response, cv_image

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
        self.__client.confirmConnection()
        self.__client.enableApiControl(True)
        self.__client.armDisarm(True)

    def reset(self, debbug = True):
        self.__client.reset()
        if debbug:
            f"Restart Connection : {self.__client.ping()}"

    def get_ground_truth_kinematics(self) -> KinematicsState:
        return self.__client.simGetGroundTruthKinematics()
    
    def get_kinematics(self):
        kinematics = self.__client.getMultirotorState().kinematics_estimated
        return kinematics if type(self.__client) == MultirotorClient else KinematicsState()
    
    def get_pose(self):
        return self.__client.simGetVehiclePose()
    
    def set_Kinematics(self, state):
        self.__client.simSetKinematics(state, ignore_collision=False)

    def set_pose(self, pose):
        self.__client.simSetVehiclePose(pose, ignore_collision=True)
        
    def rotation_rate(self, roll : float, pitch : float, yaw : float, z : float):
        self.__client.moveByAngleRatesZAsync(roll, pitch, yaw, z, duration=10)
        
    def take_off(self):
        self.__client.enableApiControl(True)
        self.__client.takeoffAsync().join()

    # def set_pose(self, t):
    #     self.__client.simSetVehiclePose()
    
class ComputerVision(_Ue4Briedge):
    def __init__(self, ip : str):
        _Ue4Briedge.__init__(self, ip, client="computer_vision")
        self.__t0 = time.time()
        self.__home = self.get_pose()
        
    def _moveon(self, linear_x : float, linear_y : float, linear_z : float, \
                    angular_x : float, angular_y : float, angular_z : float):
        
        dt = time.time() - self.__t0

        x = linear_x * dt
        y = linear_y * dt
        z = linear_z * dt

        roll = angular_x * dt
        pitch = angular_y * dt
        yaw = angular_z * dt

        pose = self.get_pose()
        position = pose.position + Vector3r(x, y ,z)
        pose.position = position
        pose.orientation += Vector3r(roll, pitch, yaw).to_Quaternionr()
        self.set_pose(pose)

    def _domain(self, z, vel, domain):
        if domain == "air" or domain == "hybrid":
            return abs(z), -vel
        if domain == "underwater":
            return z, vel

    # def take_off(self, h, vel= 0.5, domain= "air"):
    #     pose = self.get_pose()
    #     z, v = self._domain(pose.position.z_val, vel, domain)
    #     relative_h = z + h

    #     while abs(z - relative_h) > 0.01:
            
    #         dt = time.time() - self.__t0
    #         nz  = v * dt
    #         pose.position.z_val = nz
    #         self.set_pose(pose)
    #         pose = self.get_pose()
    #         z, v = self._domain(pose.position.z_val, vel, domain)
        
        
    def vertical_movement(f):
        def wrap(*args, **kwargs):
            self = args[0]
            h, vel = f(*args, **kwargs)
            
            pose = self.get_pose()
            
            z = pose.position.z_val
            relative_h = z + h

            next_position = copy.deepcopy(pose.position)
            next_position.z_val = relative_h
            
            distance = pose.position.distance_to(next_position)
            t0 = time.time()
            self.set_pose(pose)
            print(f'z : {z}, h : {h}, relative : {relative_h}, distance : {distance}')
            while distance > 0.009:                
                dt = time.time() - t0
                nz  = vel * dt
                pose.position.z_val = nz
                print(pose.position.z_val, distance)
                self.set_pose(pose)
                pose = self.get_pose()
                distance = pose.position.distance_to(next_position)
            print(f'z : {z}, h : {h}, relative : {relative_h}, distance : {distance}')
        return wrap

    @vertical_movement
    def _up(self, h, vel):
        return -h, -vel
    
    @vertical_movement
    def _down(self, h, vel):
        return h, vel

    def take_off(self, h, vel, domain):
        self.__home = self.get_pose()
        print(f"Home : {self.__home.position.to_numpy_array()}")
        self._down(h, vel) if domain == "underwater" else self._up(h, vel)

    def land(self, vel, domain):
        pose = self.get_pose()
        h = pose.position.distance_to(self.__home.position)
        print(f"Position : {pose.position.to_numpy_array()} - Target Land : {self.__home.position.to_numpy_array()} - Altitude : {h}")
        self._up(h, vel) if domain == "underwater" else self._down(h, vel)

    def get_state(self, action : NDArray) -> Tuple[NDArray, bool]:
        linear_x, linear_y, linear_z, angular_x, angular_y, angular_z = action
        linear_x = np.clip(linear_x, -0.5, 0.5)
        linear_y = np.clip(linear_y, -0.5, 0.5)
        linear_z = np.clip(linear_z, -0.5, 0.5)
        angular_x = np.clip(angular_x, -0.25, 0.25)
        angular_y = np.clip(angular_y, -0.25, 0.25)
        angular_z = np.clip(angular_z, -0.25, 0.25)
        
        self._moveon(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
        self.__t0 = time.time()

        

class QuadrotorClient(MultirotorClient):
    def __init__(self, ip : str, vehicle_name : str, camera_name : str, observation : str):
        MultirotorClient.__init__(self, ip)
        self.confirmConnection()
        self.enableApiControl(True)
        self.armDisarm(True)

        self.__vehicle_name = vehicle_name
        self.__camera_name = camera_name
        self.__observation = observation

    @property
    def vehicle_name(self):
        return self.__vehicle_name
    
    @property
    def camera_name(self):
        return self.__camera_name
    
    @property
    def observation(self):
        return self.__observation
    
    def trajectory(self, linear_x : float, linear_y : float, linear_z : float, \
                    angular_x : float, angular_y : float, angular_z : float, dt : float = 0.001):
        

        linear_velocity = Vector3r(linear_x, linear_y, linear_z)
        linear_acc = Vector3r(linear_x/dt, linear_y/dt, linear_z/dt)
        angular_velocity = Vector3r(angular_x, angular_y, angular_z)
        angular_acc = Vector3r(angular_x/dt, angular_y/dt, angular_z/dt)

        state = KinematicsState()
        state.position = Vector3r(linear_x*dt, linear_x*dt, linear_x*dt)
        state.linear_velocity = linear_velocity
        state.angular_velocity = angular_velocity
        state.linear_acceleration = linear_acc
        state.angular_acceleration = angular_acc

        self.simSetKinematics(state)
    
    def moveon(self, linear_x : float, linear_y : float, linear_z : float, \
                    angular_x : float, angular_y : float, angular_z : float):
        dt = 1
        pose = self.simGetVehiclePose()
        position = pose.position + Vector3r(linear_x*dt, linear_y*dt ,linear_z*dt)
        pose.position = position
        pose.orientation += Vector3r(angular_x*dt, angular_y*dt, angular_z*dt).to_Quaternionr()
        self.simSetVehiclePose(pose, False)

    def _observation(self) -> List:
        t = Thread()
        if self.observation == 'rgb':
            # return self.simGetImages([ImageRequest(self.camera_name, ImageType.Scene)])
            return transform_response(self.simGetImages([ImageRequest(self.camera_name, ImageType.Scene)]))
        if self.observation == 'depth':
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     response = self.simGetImages([ImageRequest(self.camera_name, ImageType.DepthPlanar, True)])
            #     future = executor.submit(transform_response, response)
            #     return future.result()
            # return cv_image(self.simGetImage(self.camera_name, ImageType.DepthPlanar))
            return transform_response(self.simGetImages([ImageRequest(self.camera_name, ImageType.DepthPlanar, True)]))
        
        if self.observation == 'segmentation':
            return self.simGetImages([ImageRequest(self.camera_name, ImageType.Segmentation)])
        if self.observation == 'stereo':
            return self.simGetImages([ImageRequest(self.camera_name, ImageType.Scene),
                                      ImageRequest(self.camera_name, ImageType.DepthPlanar, True)])
        if self.observation == 'panoptic':
            return self.simGetImages([ImageRequest(self.camera_name, ImageType.Scene),
                                      ImageRequest(self.camera_name, ImageType.DepthPlanar, True),
                                      ImageRequest(self.camera_name, ImageType.Segmentation)])

    def get_state(self, action : NDArray) -> Tuple[NDArray, bool]:
        linear_x, linear_y, linear_z, angular_x, angular_y, angular_z = action
        linear_x = np.clip(linear_x, -0.25, 0.25)
        linear_y = np.clip(linear_y, -0.25, 0.25)
        linear_z = np.clip(linear_z, -0.25, 0.25)
        angular_x = np.clip(angular_x, -0.25, 0.25)
        angular_y = np.clip(angular_y, -0.25, 0.25)
        angular_z = np.clip(angular_z, -0.25, 0.25)
        
        # self.moveon(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
        self.moveByVelocityBodyFrameAsync(linear_x, linear_y, linear_x, 5)
        # self._observation()
    