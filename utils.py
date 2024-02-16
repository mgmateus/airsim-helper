import math
import cv2

import numpy as np
from numpy.typing import NDArray

from typing import List, Tuple
from PIL import Image

from .airsim_base.types import Pose, Vector3r, ImageType, ImageResponse
from .airsim_base.utils import to_eularian_angles, string_to_uint8_array


#from cv_bridge import CvBridge, CvBridgeError

def angular_diference(current : float, to : float) -> float:
    heading= to - current
    if heading > math.pi:
        heading -= 2 * math.pi

    elif heading < -math.pi:
        heading += 2 * math.pi

    return heading 

def eularian_diference(vehicle_pose : Pose, target_position : Vector3r) -> Tuple[float, float, float]:
    vx, vy, vz = vehicle_pose.position
    vpitch, vroll, vyaw = to_eularian_angles(vehicle_pose.orientation)

    tx, ty, tz = target_position
    
    u = Vector3r(tx, ty, vz)
    v = Vector3r(tx, ty, tz)
    
    pitch = np.arccos(u.dot(v)/(u.get_length() * v.get_length()))
    pitch = angular_diference(vpitch, pitch)

    roll = np.arctan2(ty - vy, tz - vz)
    roll = angular_diference(vroll, roll)
    
    yaw = np.arctan2(ty - vy, tx - vx)
    yaw = angular_diference(vyaw, yaw)

    return pitch, roll, yaw

def cv_image(raw_image) -> List:
    return cv2.imdecode(string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)

def color_from_response(response : ImageResponse) -> NDArray:
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_rgb = img_rgb[..., :3][..., ::-1]
    return img_rgb

def depth_from_response(response : ImageResponse) -> NDArray:
    img_depth = np.array(response.image_data_float, dtype=np.float64)
    img_depth = img_depth.reshape(response.height, response.width)
    return img_depth

def transform_response(responses):
    images = []
    for raw in responses:
        if raw.image_type == ImageType.Scene or raw.image_type == ImageType.Segmentation:
            images.append(color_from_response(raw))
        else:
            images.append(depth_from_response(raw))
    return images

    
     
      

