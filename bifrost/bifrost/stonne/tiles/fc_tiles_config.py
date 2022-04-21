import os

# T_M=[x]: Number of output neurons
# T_N=[x]: Batch size
# T_K=[x]: Number of input neurons

class FCTileConfig(object):
    def __init__(self):
        self.path: str
        self.T_N : int 
        self.T_S : int 
        self.T_K : int 

    def generate_basic_tile_config(self):
        return self.edit_tile_config(1,1)

    def edit_tile_config(self,S,K):
        self.T_S = S 
        self.T_K = K 

        if not os.path.exists("bifrost_temp"):
            os.mkdir("bifrost_temp")
        elif not os.path.exists("bifrost_temp/fc_tiles"):
            os.mkdir("bifrost_temp/fc_tiles")

        self.path = os.path.join(
            os.getcwd() ,
            "bifrost_temp/fc_tiles/fc_tile_config_"
                + str(self.T_S) 
                + str(self.T_K) 
                + ".txt" 
        )
        
        with open(self.path, "w+") as f:
            f.write(f'tile_type="FC"\n')
            f.write(f"T_N=1\n")
            f.write(f"T_S={self.T_S}\n")
            f.write(f"T_K={self.T_K}\n")
        return self.path
    


fc_tiles = FCTileConfig()
