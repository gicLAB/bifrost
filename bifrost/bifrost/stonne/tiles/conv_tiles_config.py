import os

class ConvTileConfig(object):

    def __init__(self):
        self.T_R : int 
        self.T_S : int 
        self.T_C : int 
        self.T_G : int 
        self.T_K : int 
        self.T_N : int 
        self.T_X : int 
        self.T_Y : int 

    def edit_tile_config(
        self,
        T_R : int,
        T_S : int,
        T_C : int,
        T_G : int,
        T_K : int,
        T_N : int,
        T_X : int,
        T_Y : int,
    ) -> str:
        """
        Edits a tile config file and returns its path

        Parameters
        ----------

        Returns
        ----------
        """
        # Make tile paths if they don't exist
        if not os.path.exists("bifrost_temp"):
            os.mkdir("bifrost_temp")
        elif not os.path.exists("bifrost_temp/conv_tiles"):
            os.mkdir("bifrost_temp/conv_tiles")

        path =os.path.join(
            os.getcwd(),
            "bifrost_temp/conv_tiles/conv_tile_config_"
                + str(T_R)
                + str(T_S)
                + str(T_C)
                + str(T_G)
                + str(T_K)
                + str(T_N)
                + str(T_X)
                + str(T_Y)   
                +".txt"
        )
        self.T_R = T_R
        self.T_S = T_S
        self.T_C = T_C
        self.T_G = T_G
        self.T_K = T_K
        self.T_N = T_N
        self.T_X = T_X
        self.T_Y = T_Y   
        
        with open(path, "w+") as f:
            f.write(f'tile_type="CONV"\n')
            f.write(f"T_R={self.T_R}\n")
            f.write(f"T_S={self.T_S}\n")
            f.write(f"T_C={self.T_C}\n")
            f.write(f"T_G={self.T_G}\n")
            f.write(f"T_K={self.T_K}\n")
            f.write(f"T_N={self.T_N}\n")
            f.write(f"T_X'={self.T_X}\n")
            f.write(f"T_Y'={self.T_Y}\n")   
        return path

    def generate_basic_tile_config(self):

        return self.edit_tile_config(
            1,1,1,1,1,1,1,1
        )

        

conv_tiles = ConvTileConfig()

