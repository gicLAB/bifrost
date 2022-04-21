from typing import List
from .tile_tuner import create_conv_tile_tuning_space, create_fc_tile_tuning_space

class TuningParameters(object):
    def __init__(
        self
    ) -> None:
        self.tune_convolutions_tile:bool = False
        self.tune_fc_tile:bool = False

        self.fc_num: int = 20
        self.conv_num: int = 20

        self.conv_tile_knobs:List = []
        self.fc_tile_knobs:List = []

        self.tune_accumulation_buffer: bool = False
        self.tune_sparsity_ratio: bool = False
        self.sparsity_ratio_range = []
        self.tune_reduce_network_type:bool = False
        self.tune_ms_network_type:bool = False   
        self.tune_ms_size:bool = False
        self.ms_size_range = [8,16,32,64,128]
        
        self.tune_psums:bool = False

        self.tune_rn_bw:bool = False
        self.tune_dn_bw:bool = False
        self.rn_bw_range = [8,16,32,64,128]
        self.dn_bw_range = [8,16,32,64,128]

    def create_knobs(self, conv = False, dense = False)->List:
        """
        Based on set tuning parameters, create all the knobs

        Returns
        -------
        all_knobs: List[Tuple(str,List[Object])]
            A list of the tuning knobs

        """
        all_knobs = []
        if conv:
            all_knobs.extend(self.conv_tile_knobs)
        if dense:
            all_knobs.extend(self.fc_tile_knobs)
        if self.tune_accumulation_buffer:
            all_knobs.append(("accumulation_buffer",[True,False]))
        if self.tune_reduce_network_type:
            all_knobs.append(("reduce_network_type",["ASNETWORK","FENETWORK"]))
        if self.tune_ms_size:
            all_knobs.append(("ms_size",self.ms_size_range))
        if self.tune_rn_bw:
            all_knobs.append(("rn_bw",self.rn_bw_range))
        if self.tune_dn_bw:   
            all_knobs.append(("dn_bw",self.dn_bw_range))
        if self.tune_sparsity_ratio:
            all_knobs.append(("sparsity_ratio", self.sparsity_ratio_range))
        self.conv_tile_knobs = []
        self.fc_tile_knobs = []
        return all_knobs


    def conv_tile(self,
        R: int,
        S: int,
        C:int,
        K:int,
        G:int,
        X:int,
        Y:int,
        strides:int,
    ):
        self.conv_tile_knobs = create_conv_tile_tuning_space(R,S,C,K,G,X,Y,strides,self.conv_num)

    def fc_tile(self):
        self.fc_tile_knobs = create_fc_tile_tuning_space(self.fc_num)

tune_parameters = TuningParameters()
