import numpy as np
from dataclasses import dataclass
import draccus
from pathlib import Path

@dataclass
class PreprocessConfig:
    root: str
    
@draccus.wrap()
def data_accumulator(cfg: PreprocessConfig)->None:
    p = Path(cfg.root)
    data = []
    for subdir in p.iterdir():
        if subdir.is_dir():
            frame_file = subdir / 'frames.npy'
            action_file  = subdir / 'action.npy'
            instruction_file = subdir / 'instruction.txt'
            
            frames = np.load(frame_file)
            actions = np.load(action_file)
            with open(instruction_file, 'r') as f:
                instruction = f.read()
                
            for i in range(len(actions)):
                data_pack = {}
                data_pack["dataset_name"] = "PIZZADATASET"
                data_pack['action'] = [actions[i]]
                data_pack["observation"] = {}
                data_pack["observation"]["image_primary"] = [frames[i]]
                data_pack["task"] = {}
                data_pack["task"]["language_instruction"] = instruction
                data.append(data_pack)
                
            print(data[0],'\n',data[-1])