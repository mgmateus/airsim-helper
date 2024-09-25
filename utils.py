import json
import numpy


def rotation_matrix_from_angles(pry):
    pitch = pry[0]
    roll = pry[1]
    yaw = pry[2]
    sy = numpy.sin(yaw)
    cy = numpy.cos(yaw)
    sp = numpy.sin(pitch)
    cp = numpy.cos(pitch)
    sr = numpy.sin(roll)
    cr = numpy.cos(roll)
    
    Rx = numpy.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    
    Ry = numpy.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    
    Rz = numpy.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    
    #Roll is applied first, then pitch, then yaw.
    RyRx = numpy.matmul(Ry, Rx)
    return numpy.matmul(Rz, RyRx)

def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    #Turn the camera position into a column vector.
    camPosition = numpy.transpose([camXYZ])

    #Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = utils.to_eularian_angles(camQuaternion)
    
    #Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = rotation_matrix_from_angles(pitchRollYaw)
    
    #Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = numpy.transpose([subjectXYZ])
    XYZW = numpy.add(XYZW, -camPosition)
    print("XYZW: " + str(XYZW))
    XYZW = numpy.matmul(numpy.transpose(camRotation), XYZW)
    print("XYZW derot: " + str(XYZW))
    
    #Recreate the perspective projection of the camera.
    XYZW = numpy.concatenate([XYZW, [[1]]])    
    XYZW = numpy.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]
    
    #Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2
    
    return numpy.array([
        imageWidthHeight[0] * normX,
        imageWidthHeight[1] * normY
    ]).reshape(2,)

class DictToClass:
    def __init__(self, dictionary : dict = dict()):

        self.update(dictionary)
    @property
    def as_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToClass):
                # Recursively convert nested DictToClass objects
                result[key] = value.as_dict
            else:
                # Directly add non-DictToClass values
                result[key] = value
        return result
    
    @property
    def as_json(self):
        return json.dumps(self.as_dict, indent=4)
    
    @property
    def keys(self):
        return self.__dict__.keys()
    
    def __repr__(self):
        return str(self.__dict__)
    
    def __getitem__(self, index):
        return self.__dict__[index]
    
    def update(self, dictionary):
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, DictToClass(value))
                else:
                    setattr(self, key, value)

            return self

        for key, value in vars(dictionary).items():
                if isinstance(value, dict):
                    setattr(self, key, DictToClass(value))
                else:
                    setattr(self, key, value)

        return self
    
    def save_as_json(self,  filename : str):
        with open(filename, 'w') as file:
            json.dump(self.as_dict, file, indent=4)
    

    
def make_settings(camera_names: list = list()) -> DictToClass:
    settings = {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "ComputerVision",
        "ViewMode": "NoDisplay",
        "ClockSpeed": 1,
        "ApiServerPort": 41451,
        "RecordUIVisible": False,
        "LogMessagesVisible": False,
        "ShowLosDebugLines": False,
        "RpcEnabled": True,
        "EngineSound": True,
        "PhysicsEngineName": "",
        "SpeedUnitFactor": 1.0,
        "SpeedUnitLabel": "m/s",
        "Wind": { "X": 0, "Y": 0, "Z": 0 }
    }
    
    camera_settings  = {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 672,
                "Height": 376,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            },
            {
                "ImageType": 1,
                "Width": 672,
                "Height": 376,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0,
                "PixelsAsFloat": True
            },
            {
                "ImageType": 5,
                "Width": 672,
                "Height": 376,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
            }
        ],
        "X": 0, "Y": 0, "Z": 0, 
        "Pitch":0.0, "Roll": 0.0, "Yaw": 0.0
    }
  
    cameras = dict({name : camera_settings for name in camera_names})
    
    settings.update({'Cameras' : cameras})
    return DictToClass(settings)
