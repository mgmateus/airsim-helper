import os
import argparse

import numpy as np

from airsim_base.client import MultirotorClient
from airsim_base.types import GeoPoint

class Storage(MultirotorClient):
    @staticmethod
    def geopoint_from_np(point):
        lat, lg, alt = point
        geopoint = GeoPoint()
        geopoint.latitude = lat
        geopoint.longitude = lg
        geopoint.altitude = alt
        return geopoint
    
    def __init__(self, ip : str, path : str):
        super().__init__(ip)
        self.__path = path
        
    def get_airsim_mesh(self, mesh : str):
        meshes = self.simGetMeshPositionVertexBuffers()
        
        for mesh_ in meshes:
            if mesh_.name == mesh:
                size_ = int(len(mesh_.vertices)/3)
                vertices = np.array(mesh_.vertices).reshape(size_,3)
                print(vertices)
                np.save(self.__path + "/"+ mesh_.name, vertices)
                break
           
    
    def get_mesh(self, mesh : str):
        return np.load(self.__path + "/"+ mesh +".npy")
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Store AirSim Meshes as .npy')
    parser.add_argument('--ip', type= str, required=False, default=os.environ['UE4_IP'], help='Configurated ip to airsim-ue4 comunication.')
    parser.add_argument('--path_to_save', type= str, required=True, help='/home/path/to/your/dir')
    parser.add_argument('--mesh_name', type= str, required=True, help="Number of squares on width in chessboard")

    args = parser.parse_args()

    ip = args.ip
    storage = Storage(ip, args.path_to_save)
    vertices = storage.get_mesh(args.mesh_name)
    
    print(vertices.shape)
    
    print(storage.simTestLineOfSightBetweenPoints(storage.geopoint_from_np([11360.0, 940, 3380.0]), "Hydrone"))
    
    # vv = vertices[:1000]
    # print(vv)
   
    # see = [storage.simTestLineOfSightToPoint(storage.geopoint_from_np(v), "Hydrone") for v in vv]
    # print(see)

    p = storage.getMultirotorState().gps_location