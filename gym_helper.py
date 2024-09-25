from airsim_base import ImageType

from computer_vision import ComputerVision
from .utils import DictToClass, make_settings



class PointOfViewTwins(ComputerVision):
    @staticmethod
    def make_cameras(camera_names : list[str]):
        cameras = make_settings(camera_names)
        cameras.update({f"{name}_twin" : cameras.CaptureSettings for name in camera_names})
        
    def __init__(self, ip: str,  camera_names : list[str]):        
        ComputerVision().__init__(ip)
        self.cameras = self.make_cameras(camera_names)
        

        
     


 