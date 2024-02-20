import os
import argparse

import numpy as np

from numpy.typing import NDArray

from simulation import QuadrotorClient

class Storage(QuadrotorClient):

    @staticmethod
    def process(vertices : NDArray):
        return vertices.reshape(3, 3)

    def __init__(self, ip : str, vehicle_name : str, camera_name : str, observation : str, path : str):
        super().__init__(ip, vehicle_name, camera_name, observation)
        self.__path = path
        
    def get_mesh(self, mesh : str):
        for mesh_ in self.simGetMeshPositionVertexBuffers():
            if mesh_.name == mesh:
                vertices = np.array(mesh_.vertices)
                np.save(self.__path + "/"+ mesh_.name, vertices)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Store AirSim Meshes as .npy')
    parser.add_argument('--ip', type= str, required=True, help='Configurated ip to airsim-ue4 comunication.')
    parser.add_argument('--path_to_save', type= str, required=True, default=os.environ['UE4_IP'], help='/home/path/to/your/dir')
    parser.add_argument('--mesh_name', type= int, required=True, help="Number of squares on width in chessboard")

    args = parser.parse_args()

    ip = args.ip
    storage = Storage(ip, args.path_to_save)
    storage.get_mesh(args.mesh_name)
    

    