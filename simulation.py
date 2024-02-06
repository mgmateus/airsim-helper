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
from airsim_base.types import Vector3r, Quaternionr, DrivetrainType, YawMode, Pose, KinematicsState
from airsim_base.utils import to_quaternion, to_eularian_angles

from utils import set_client

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

    def reset(self, debbug = True):
        self.__client.reset()
        if debbug:
            f"Restart Connection : {self.__client.ping()}"

    def get_ground_truth_Kinematics(self) -> KinematicsState:
        return self.__client.simGetGroundTruthKinematics()
    
    def set_Kinematics(self, state):
        self.__client.simSetKinematics(state, ignore_collision=False)


class ComputerVision(_Ue4Briedge):
    def __init__(self, ip : str):
        _Ue4Briedge.__init__(self, ip, client="computer_vision")

    def trajectory(self, vel_rate : np.array):
        vx, vy, vz, pitch, yaw = vel_rate

        linear_velocity = Vector3r(vx, vy, vz)
        angular_velocity = Vector3r(pitch, 0, yaw)

        state = KinematicsState()
        state.linear_velocity = linear_velocity
        state.angular_velocity = angular_velocity

        self.set_Kinematics(state)

if __name__ == "__main__":
    IP = os.environ['UE4_IP'] or ""
    cv = ComputerVision(IP)

    v = 0.5
    rate = 10
    kinematics = cv.get_ground_truth_Kinematics()
    for i in range(10):
        vk = kinematics.linear_velocity.to_numpy_array() * v
        ak = kinematics.angular_velocity.to_numpy_array() * rate

        vel_rate = np.hstack((vk,ak))
        cv.trajectory(vel_rate)
        kinematics = cv.get_ground_truth_Kinematics()






