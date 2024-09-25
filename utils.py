import json

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
