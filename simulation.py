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
from abc import ABC, abstractmethod

from airsim_base import MultirotorClient, VehicleClient
from airsim_base import Pose, ImageRequest, ImageType, ImageResponse
from airsim_base import to_eularian_angles, write_file

from std_msgs.msg import String

from utils import pose, angular_distance

class _Debbug(ABC):
    restart_conn = "Restart Connection : "

    @abstractmethod
    def loginfo(self, info):
        ...


class Ue4Briedge(_Debbug):
    """Starts communication's Engine.

    Args:
        HOST: 
            -For set ip address from host, use ipconfig result's on host
            -For set ip address from docker network, use a ip from ping result's between containers on host
            -For set ip address from WSL, os.environ['WSL_HOST_IP'] on host.
    """        

    @staticmethod
    def define_client(type_client : str, ip : str = os.environ['UE4_IP']) -> VehicleClient or MultirotorClient:
        """Set client

        Args:
            type_client (str): rotor or cv
            ip (str, optional): ip to conect airsim with UE4. Defaults to os.environ['UE4_IP'].

        Returns:
           VehicleClient or MultirotorClient: cv or rotor client
        """
        return MultirotorClient(ip) if type_client == 'rotor' else VehicleClient(ip)
    
    def __init__(self, type_client : str, project_name : str) -> None:
        self._client = self.define_client(type_client=type_client)
        self._client.confirmConnection() 

    def loginfo(self, info):
        full_info = _Debbug.restart_conn + f"{info}"
        print(full_info)

    def restart(self) -> None:
        """
        Reset the ue4 client conection.
        """        
        self._client.reset()




        
    
    
























