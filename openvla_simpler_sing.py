import os
import draccus

import torch
import torch.distributed as dist

from log import setup_logging

from dataclasses import dataclass

def ddp_setup

@dataclass
class finetune_config:
    
    vla_path: str
    

@draccus.wrap()
def finetune(cfg: finetune_config)->None:
    ddp_setup()
    
    finetune_logs = setup_logging()
    
    finetune_logs.setup_logger()
    
    