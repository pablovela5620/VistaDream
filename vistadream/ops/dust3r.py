import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/Dust3r'
sys.path.append(reference)

from dust3r_command import Dust3r

class Dust3r_Tool():
    def __init__(self,cfg):
        self.ckpt=cfg.model.mde.dust3r.ckpt
        self.tool = Dust3r(self.ckpt,'cpu')
    
    def to(self,device):
        self.tool.device = device
        self.tool.model.to(device)
    
    def __call__(self,filelist):
        frames = self.tool(filelist)
        return frames