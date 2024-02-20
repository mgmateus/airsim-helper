import os
from simulation import QuadrotorClient

class Storage(QuadrotorClient):
    def __init__(self, ip : str, vehicle_name : str, camera_name : str, observation : str):
        super().__init__(ip, vehicle_name, camera_name, observation)
        
    def get_mesh(self, mesh : str):
        for mesh_ in self.simGetMeshPositionVertexBuffers():
            if mesh_.name == mesh:
                print(mesh_)
        
if __name__ == "__main__":
    ip = os.environ['UE4_IP']
    storage = Storage(ip, 'Hydrone', 'stereo', 'depth')
    storage.get_meshe('semi_sub')