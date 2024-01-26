import math
import rospy

import numpy as np

from airsim_base.types import Pose
from airsim_base.utils import to_quaternion

from cv_bridge import CvBridge, CvBridgeError


def angular_distance(position : tuple, target_position : tuple, degrees : bool) -> float:
    """Calculate the angular distance between origin and target position

    Args:
        position (tuple): (x, y, z) current position.
        target_position (tuple): (x, y, z) target's position.
        degrees: Define if radian's or degree's at result type 

    Returns:
        float: distance
    """    
    x, y, _= position
    tx, ty, _= target_position

    to= np.arctan2(ty - y, tx - x)

    heading= to - 0
    if heading > math.pi:
        heading -= 2 * math.pi

    elif heading < -math.pi:
        heading += 2 * math.pi

    return heading if not degrees else np.degrees(heading)

def pose(position : tuple, eularian_orientation : tuple) -> Pose:
            """_summary_

            Args:
                position (tuple): position (x, y ,z).
                eularian_orientation (tuple): (roll, pitch, yaw).

            Returns:
                Pose: AirSim pose type.
            """        
            x, y, z = position
            pitch, roll, yaw =  np.deg2rad(eularian_orientation[0]),\
                                np.deg2rad(eularian_orientation[1]),\
                                np.deg2rad(eularian_orientation[2])
            pose_ = Pose()
            pose_.position.x_val = x
            pose_.position.y_val = y
            pose_.position.z_val = z
            
            pose_.orientation = to_quaternion(pitch, roll, yaw)
            
            return pose_

def image_transport(img_msg):
        rospy.logwarn(img_msg.header)
        try:
            return CvBridge().imgmsg_to_cv2(img_msg, "passthrough")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            

