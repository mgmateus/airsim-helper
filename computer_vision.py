import copy

from typing import Tuple, List, Dict
from numpy.typing import NDArray

import numpy as np

from airsim_base import (Vector3r, Quaternionr, Pose, ImageRequest, \
                           ImageResponse, ImageType)
from airsim_base import to_quaternion, to_eularian_angles, string_to_uint8_array
from airsim_base import VehicleClient

from utils import DictToClass

class ComputerVision(VehicleClient):

    @staticmethod
    def pose_from_positon_euler_list(pose : list) -> Pose:
        x, y, z, roll, pitch, yaw= pose
        return Pose(Vector3r(x, y, z,), 
               to_quaternion(pitch=pitch, roll=roll, yaw=yaw)) 
    
    @staticmethod
    def pose_to_position_quat(pose : Pose) -> Tuple[NDArray, NDArray]:
        position = pose.position.to_numpy_array()
        orientation = pose.orientation.to_numpy_array()
        return position, orientation
    
    @staticmethod
    def pose_to_ndarray(pose : Pose) -> Tuple[NDArray, NDArray]:
        position = pose.position.to_numpy_array()
        orientation = pose.orientation.to_numpy_array()
        return np.hstack((position, orientation))

    @staticmethod
    def request(camera_name : str = ""):
        """
        Generates a dictionary of image requests for the specified camera.

        Args
        ----------
        - camera_name (str): The name of the camera. If not provided, it defaults to "0" for the first camera and "1" for the second camera.

        Returns
        ----------
        - dict: A dictionary containing two keys: "rgb" and "stereo". The "rgb" key contains a list of ImageRequest objects for the RGB camera, and the "stereo" key contains a list of ImageRequest objects for the stereo camera.
        """
        rgb = [ImageRequest(camera_name or "0", ImageType.Scene, False, False)]
        stereo = [ImageRequest(camera_name or "0", ImageType.Scene, False, False),
                  ImageRequest(camera_name or "1", ImageType.DepthPlanar, True)]
        return dict(rgb= rgb, stereo= stereo)
    
    def __init__(self,
                  ip : str):
        VehicleClient.__init__(self, ip)    
        self.confirmConnection()  

        self.__image_dim = (0, 0)
        self.__camera_fov = 0.0
        self.__gimbal = to_quaternion(0,0,0)

    @property
    def image_dim(self):
        return self.__image_dim
    
    @image_dim.setter
    def image_dim(self, value):
        self.__image_dim = value

    @property
    def camera_fov(self):
        return self.__camera_fov

    @camera_fov.setter
    def camera_fov(self, value):
        self.__camera_fov = value
    
    @property
    def gimbal(self):
        return self.__gimbal
    
    @gimbal.setter
    def gimbal(self, data):
        assert isinstance(data, list) or isinstance(data, Quaternionr)

        if isinstance(data, list):
            roll, pitch, yaw = data
            q = to_quaternion(roll=roll, pitch=pitch, yaw=yaw)
        else:
            q = data
        self.__gimbal *= q

    def _call(self, requests : Dict[str, List[ImageRequest]], vehicle_name : str = "") \
                                                                -> List[ImageResponse]:
        """
        Retrieves image with non-zero responses from the simulator.

        Args
        ----------
        - key (str): A valid key to send a request.
        - camera_name (str): The name of the camera. (default is "")
        - vehicle_name (str): The name of the vehicle. (default is "")

        Returns
        ----------
        - list: A list of images, where each image is a 3D numpy array (height, width, 3).
        """
        retry = True
        responses = None
        while True:
            responses = self.simGetImages(requests, vehicle_name)
            if responses:
                for response in responses:
                    if not response.size:
                        retry = True
                        break
                    retry = False
                
                if not retry:
                    break

        return responses


    def _image_responses(self, key : str, camera_name : str = "", vehicle_name : str = "") \
                                                                                    -> list:
        """
        Retrieves image responses from the simulator based on the provided key.

        Args
        ----------
        - key (str): A valid key to send a request.
        - camera_name (str): The name of the camera. (default is "")
        - vehicle_name (str): The name of the vehicle. (default is "")

        Returns
        ----------
        - list: A list of images, where each image is a 3D numpy array (height, width, 3).
        """

        request = self.request(camera_name)[key]
        responses = self._call(request, vehicle_name)
        call = list()

        for response in responses:
            if response.image_data_uint8:
                img1d = string_to_uint8_array(response.image_data_uint8) # convert binary image to numpy array
                image = np.flipud(img1d.reshape(response.height, response.width, 3)) # convert to RGB
            else:
                img1d = np.array(response.image_data_float, dtype=np.float32)
                image = img1d.reshape(response.height, response.width)

            call.append(image)

        return call

    def _depth2pcd(self, raw_depth : NDArray) -> NDArray:
        """
        Maps a 2D depth image to a 3D point cloud.

        Args
        ----------
        raw_depth (NDArray): A 2D numpy array representing the depth image.

        Returns
        ----------
        NDArray: A 2D numpy array representing the point cloud.
        """
        
        raw_depth = copy.deepcopy(raw_depth) * 100
        raw_depth = np.clip(raw_depth, 0, 30000) #limit of the depth sensor

        height, width = raw_depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height)) #reference to the pixel coordinates

        u = u.flatten()
        v = v.flatten()
        depth = raw_depth.flatten()  #flatten the depth image

        # Intrinsic camera parameters
        image_width, image_height = self.__image_dim
        fov_rad = 90 * np.pi/180
        fd = (image_width/2.0) / np.tan(fov_rad/2.0)
        print(fd)
        fx, fy = fd, fd
        cx, cy = image_width/2, image_height/2
        
        x = (u - cx) * depth / fx # calculate the x coordinate projected in the 3D space
        y = (v - cy) * depth / fy # calculate the y coordinate projected in the 3D space
        z = depth
        
        points = np.vstack((-y, -z, -x)).T  # stack the coordinates in a 2D array

        valid_indices = z > 0 # filter out invalid points (depth = 0)
        return points[valid_indices] 

    def call(self, key : str, camera_name : str ='', vehicle_name : str ='') -> list:
        """
        Calls the service to get the data from the camera.

        Args
        ----------
        - key (str): The key of the data to be retrieved.
        - camera_name (str): The name of the camera. Defaults to an empty string.
        - vehicle_name (str): The name of the vehicle. Defaults to an empty string.

        Returns
        ----------
        - list: A list of data retrieved from the camera.
        """

        def _parse_key(_key : str):
            return  _key.split('_') if _key.count('_') else _key, None

        key, occupancy = _parse_key(key) # split the key into the image name and the occupancy flag
        calls = self._image_responses(key, camera_name, vehicle_name) 
        if occupancy:
            calls.append(self._depth2pcd(calls[1]))
        
        camera_pose = self.simGetCameraInfo(camera_name, vehicle_name).pose
        calls.append(self.pose_to_ndarray(camera_pose))

        return calls
    
    def start(self, vehicle_name : str = ''):
        self.enableApiControl(True, vehicle_name)
        self.armDisarm(True, vehicle_name)



if __name__ == "__main__":
    cv = ComputerVision('172.19.0.2')
    import time
    vp, cp = cv.simGetVehiclePose(), cv.simGetCameraInfo('0').pose
    print(f"Vehicle position : {vp.position.to_numpy_array()} -- Camera position : {cp.position.to_numpy_array()}")
    time.sleep(2)

    relative_cp = cv.pose_from_positon_euler_list([0.3, 0, 0.3, 0, 0, 0])

    cv.simSetCameraPose('0',relative_cp)
    vp, cp = cv.simGetVehiclePose(), cv.simGetCameraInfo('0').pose
    print(f"Vehicle position : {vp.position.to_numpy_array()} -- Camera position : {cp.position.to_numpy_array()}")
    time.sleep(2)

    cv.simSetVehiclePose(cv.pose_from_positon_euler_list([0, 0, -50, 0, 0, 0]), True)
    time.sleep(2)
    cv.simSetCameraPose('0',relative_cp)
    vp, cp = cv.simGetVehiclePose(), cv.simGetCameraInfo('0').pose
    print(f"Vehicle position : {vp.position.to_numpy_array()} -- Camera position : {cp.position.to_numpy_array()}")
    time.sleep(3)
