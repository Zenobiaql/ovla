import numpy as np
import os

frames = np.load("C:\\Users\\v-qilinzhang\\Desktop\\ovla\\dataset\\frames.npy")
actions = np.load("C:\\Users\\v-qilinzhang\\Desktop\\ovla\\dataset\\action.npy")
with open("C:\\Users\\v-qilinzhang\\Desktop\\ovla\\dataset\\instruction.txt") as f:
    instruction = f.read()

data = []
for i in range(len(actions)):
    data_pack = {}
    data_pack["dataset_name"] = "PIZZADATASET"
    data_pack['action'] = [actions[i]]
    data_pack["observation"] = {}
    data_pack["observation"]["image_primary"] = [frames[i]]
    data_pack["task"] = {}
    data_pack["task"]["language_instruction"] = instruction
    data.append(data_pack)
    
print(data[0])
