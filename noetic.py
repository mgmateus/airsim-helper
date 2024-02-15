
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from airsim_ros_pkgs.msg import VelCmd, GimbalAngleEulerCmd
from airsim_ros_pkgs.srv import Takeoff, Land

class QuarotorROS:
    
    @staticmethod
    def image_transport(img_msg):
        try:
            return CvBridge().imgmsg_to_cv2(img_msg, "passthrough")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            
    def __init__(self, vehicle_name : str):
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/Stereo_Cam/Scene", \
                         Image, self._callback_rgb)
        rospy.Subscriber("/airsim_node/"+vehicle_name+"/Stereo_Cam/DepthPerspective", \
                         Image, self._callback_depth)
        
        self.__velocity_pub = rospy.Publisher("/airsim_node/"+vehicle_name+"/vel_cmd_world_frame", \
                                        VelCmd, queue_size=1)
        self.__gimbal_pub = rospy.Publisher("/airsim_node/gimbal_angle_euler_cmd", \
                                        GimbalAngleEulerCmd, queue_size=1)
        self.__pub_info = rospy.Publisher("uav_info", \
                                          String, queue_size=10)
        
        self.__vehicle_name = vehicle_name
    
    def __str__(self) -> str:
        return self.__vehicle_name
            
    ## Callbacks
    def callback_image(func):
        def callback(self, *args, **kwargs):
            data, img_type = func(self, *args, **kwargs)
            if data:
                cv_rgb = self.image_transport(data)
                self.__setattr__("__"+img_type, cv_rgb)
            else:
                info = f"Error in {img_type} cam!"
                self.__pub_info.publish(info)

        return callback
    
    @callback_image
    def _callback_rgb(self, data):
        return data, "rgb"
    
    @callback_image
    def _callback_depth(self, data):
        return data, "depth"
    
    ## Services
    def take_off(self):
        try:
            service = rospy.ServiceProxy("/airsim_node/"+self.__vehicle_name+"/takeoff", Takeoff)
            rospy.wait_for_service("/airsim_node/"+self.__vehicle_name+"/takeoff")

            service()

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)

    def land(self):
        try:
            service = rospy.ServiceProxy("/airsim_node/"+self.__vehicle_name+"/land", Land)
            rospy.wait_for_service("/airsim_node/"+self.__vehicle_name+"/land")

            service()

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)
            
    ##Functions
    def _velocity(self, linear_x : float, linear_y : float, linear_z : float, \
                    angular_x : float, angular_y : float, angular_z : float):
        
        vel = VelCmd()
        vel.twist.linear.x = linear_x
        vel.twist.linear.y = linear_y
        vel.twist.linear.z = linear_z
        vel.twist.angular.x = angular_x
        vel.twist.angular.y = angular_y
        vel.twist.angular.z = angular_z
        
        self.__velocity_pub.publish(vel)
        
    def _gimbal(self, pitch : float, yaw : float):
        gimbal = GimbalAngleEulerCmd()
        gimbal.camera_name = "Stereo_Cam"
        gimbal.vehicle_name = "Hydrone"
        gimbal.roll = 0.0
        gimbal.pitch = pitch
        gimbal.yaw = yaw
        
        self.__gimbal_pub.publish(gimbal)
        
        
    def get_state(self, action : numpy_array) -> Tuple[numpy_array, bool]:
        linear_x, linear_y, linear_z, angular_x, angular_y, angular_z = action
        linear_x = np.clip(linear_x, -.25, .25)
        linear_y = np.clip(linear_y, -.25, .25)
        linear_z = np.clip(linear_z, -.25, .25)
        angular_x = np.clip(angular_x, -.25, .25)
        angular_y = np.clip(angular_y, -.25, .25)
        angular_z = np.clip(angular_z, -.25, .25)
        
        self._velocity(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
        
    def get_state2(self, action : numpy_array) -> Tuple[numpy_array, bool]:
        pitch, yaw = action
        
        pitch = np.clip(pitch, -np.pi, np.pi)
        yaw = np.clip(yaw, -np.pi, np.pi)
        
        rospy.logwarn("ooooooooooooooooooooooooooooooooooooooooooooooooooooooiiiiiiiiiiiiiiiiiiiiiiiii")
        
        self._gimbal(pitch, yaw)

        
        
        
    





