#!/usr/bin/env python3

import os
import pprint
import json

import numpy as np

from airsim_base.client import MultirotorClient
from airsim_base.types import Vector3r, Quaternionr, Pose, KinematicsState
from airsim_base.utils import to_quaternion

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
        self._client = client

    def log_restart_connection(self):
        full_info = f"{self.restart_conn} {self._client.ping()}"
        print(full_info)



class ClientController(_Ue4Briedge, _Debbug):
    def __init__(self) -> None:
        _Ue4Briedge.__init__(self)
        _Debbug.__init__(self, self.client)
        self.vehicle_name = list(json.loads(self.client.getSettingsString())['Vehicles'].keys())[0]
        

    def moveTo(self, x : float, y : float, z : float, pitch : float, yaw : float) -> bool:
        position = Vector3r(x, y, z)
        orientation = to_quaternion(pitch, 0, yaw)

        rtk = self.client.getMultirotorState().kinematics_estimated

        nrtk = KinematicsState()
        nrtk.position = position
        nrtk.orientation = orientation
        nrtk.linear_velocity = rtk.linear_velocity
        nrtk.angular_velocity = rtk.angular_velocity
        nrtk.linear_acceleration = rtk.linear_acceleration
        nrtk.angular_acceleration = rtk.angular_acceleration

        self.client.simSetKinematics(nrtk, ignore_collision=False, vehicle_name=self.vehicle_name)

    def calcular_diferenca_angulos(self, vetor1, vetor2):
        # Normalizar os vetores
        u = vetor1 / np.linalg.norm(vetor1)
        v = vetor2 / np.linalg.norm(vetor2)

        # Calcular o cosseno do ângulo de yaw
        cos_yaw = np.dot(u[:2], v[:2])

        # Calcular o ângulo de yaw
        yaw = np.arccos(cos_yaw)

        # Modificar os vetores para excluir a componente de yaw
        u_prime = np.array([u[0], u[1], 0])
        v_prime = np.array([v[0], v[1], 0])

        # Calcular o cosseno dos ângulos de roll e pitch
        #cos_roll = np.dot(u_prime, v_prime) / (np.linalg.norm(u_prime) * np.linalg.norm(v_prime))
        cos_pitch = np.dot(u, v)

        # Calcular os ângulos de roll e pitch
        #roll = np.arccos(cos_roll)
        pitch = np.arccos(cos_pitch)

        # Converter de radianos para graus
        yaw_graus = np.degrees(yaw)
        #roll_graus = np.degrees(roll)
        pitch_graus = np.degrees(pitch)

        return pitch_graus, yaw_graus

    # def _oriented_target_vision_to_vehicle(self, target : str, radius : float, dist : float, theta : float):
    #     """Define a random vehicle pose at target based on radius range and secure distance to avoid collision.

    #     Args:
    #         vehicle_name (str) : Vehicle's settings name.
    #         target (str) : Name of target's object.
    #         radius (float): Range around the target to define vehicle's pose.
    #         dist (float): Secure distance to spawn.
    #         theta (float): Angle (0, 2pi) to direction of radius.

    #     Returns:
    #         tuple: New vehicle pose (x, y) and orientation (phi). 
    #     """ 
       
    #     _, _, current_phi = self.get_current_eularian_vehicle_angles(vehicle_name)
    #     pose_object = self._get_object_pose(target)

    #     ox = pose_object.position.x_val
    #     oy = pose_object.position.y_val

    #     cos_theta = np.cos(theta)
    #     sin_theta = np.sin(theta)

    #     px = ox + (dist*sin_theta + radius*sin_theta)
    #     py = oy + (dist*cos_theta + radius*cos_theta)
    #     phi = current_phi + angular_distance((px, py, 0), (ox, oy, 0), degrees=True)
    #     return (px, py, phi)


    # def set_oriented_target_vision_to_vehicle(self, vehicle_name : str, target : str, radius : float, dist : float, theta : float) -> None:
    #     """Define a random vehicle pose at target based on radius range and secure distance to avoid collision.

    #     Args:
    #         vehicle_name (str) : Vehicle's settings name.
    #         target (str) : Name of target's object.
    #         radius (float): Range around the target to define vehicle's pose.
    #         dist (float): Secure distance to spawn.
    #         theta (float): Angle (0, 2pi) to direction of radius.

    #     Returns:
    #         tuple: New vehicle pose (x, y) and orientation (phi). 
    #     """ 

    #     pose = self.get_vehicle_pose(vehicle_name)
    #     z = pose.position.z_val

    #     x, y, phi = self._oriented_target_vision_to_vehicle(vehicle_name, target, radius, dist, theta)
    #     position = (x, y, z)
    #     eularian_orientation = (0, 0, phi)
    #     self._set_vehicle_pose(vehicle_name, position, eularian_orientation)



if __name__ == "__main__":
    c = ClientController()
    position = Vector3r(10, 10, 10)
    orientation = to_quaternion(10, 0, 10)

    pv = c.client.simGetVehiclePose('Hydrone')
    xv, yv, zv = pv.position

    po = c.client.simGetObjectPose('teste')
    xo, yo, zo = po.position
    
    r = np.sqrt((xv - xo)**2 + (yv - yo)**2 + (zv - zo)**2)
    pitch = 90 - np.rad2deg(np.arcsin(zo/r))
    yaw = 90 - np.rad2deg(np.arccos(yo/(r * np.cos(pitch))))

    pv.orientation = to_quaternion(pitch, 180, yaw)

    c.client.simSetCameraPose("Stereo_Cam", pv, "Hydrone")
    
