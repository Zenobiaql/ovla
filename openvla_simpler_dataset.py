from pathlib import Path
import numpy as np
import os

import torch
from torch.utils.data import Dataset

from prismatic.vla.datasets import RLDSBatchTransform
from tqdm import tqdm

from dataclasses import dataclass

class PizzaDataset(Dataset):
    def __init__(self, data_root_dir, action_tokenizer, processtokenier, image_transform, prompt_builder_fn):
        self.data_dir = data_root_dir
        self.batchTransform = RLDSBatchTransform(
            action_tokenizer,
            processtokenier,
            image_transform, 
            prompt_builder_fn)
    
        p = Path(data_root_dir)
        self.data = []
        for subdir in tqdm(p.iterdir(), desc="Processing subdirectories"):
            if subdir.is_dir():
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir():
                        frame_file = subdir / 'frames.npy'
                        action_file  = subdir / 'action.npy'
                        instruction_file = subdir / 'instruction.txt'
                        
                        frames = np.load(frame_file)
                        actions = np.load(action_file)
                        with open(instruction_file, 'r') as f:
                            instruction = f.read()
                            
                        for i in tqdm(range(len(actions)), desc=f"Processing files in {subdir.name}"):
                            data_pack = {}
                            data_pack["dataset_name"] = "PIZZADATASET"
                            data_pack['action'] = [actions[i]]
                            data_pack["observation"] = {}
                            data_pack["observation"]["image_primary"] = [frames[i]]
                            data_pack["task"] = {}
                            data_pack["task"]["language_instruction"] = instruction
                            self.data.append(data_pack)
                    else:
                        pass
            else:
                pass
                      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.batchTransform(self.data[idx])