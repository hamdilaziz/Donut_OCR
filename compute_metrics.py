import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch 
from transformers import VisionEncoderDecoderModel, DonutProcessor
import wandb
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import json
import re
import editdistance


data_root = "/gpfsstore/rech/jqv/ubb84id/data/IAM"
sub_folder_name = "IAM_page_sem"
json_name = "formatted-IAM-DB-subwords-bart.json"
imgs_folder = "flat-300dpi"

# load the data
with open(os.path.join(data_root, sub_folder_name, json_name)) as f:
    content = json.load(f)
charset = content['charset']
ground_truth = content['ground_truth']
train_set, valid_set, test_set = ground_truth['train'], ground_truth['valid'], ground_truth['test']
train_names = list(train_set.keys())
valid_names = list(valid_set.keys())
test_names = list(test_set.keys())

# parameters    
config = {
  "path" : "/gpfsstore/rech/jqv/ubb84id/output_models",
  "model_names":["donut_all_lr1e-05_h1280_w960", "donut_all_lr5e-06_h1280_w960","donut_encoder_only_lr1e-06_h1280_w960",
                 "donut_all_lr1e-06_h1280_w960", "donut_encoder_only_lr1e-05_h1280_w960", "donut_encoder_only_lr5e-06_h1280_w960"],
  "mean":[0.485, 0.456, 0.406],
  "std":[0.229, 0.224, 0.225],
  "image_size":[960, 1280],
  "max_length":224,
  "batch_size":10,
  "device":'cuda' if torch.cuda.is_available() else 'cpu',
}

# dataset
class IAM_dataset(Dataset):
    def __init__(self, 
                 paths,
                 data_set, 
                 data_folder_path = "/gpfsstore/rech/jqv/ubb84id/data/IAM/1280_960",
                 device='cpu',
                 ext='.pt'):
        
        self.paths = paths
        self.data_set = data_set
        self.data_folder_path = data_folder_path
        self.device = device
        self.ext=ext
            
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,indices):
        x,y = [],[]
        for idx in indices:
            name = self.paths[idx].split('.')[0]
            x.append(torch.load(os.path.join(self.data_folder_path,
                                             self.data_set,
                                             'images',
                                             name+self.ext), map_location=self.device))
            y.append(torch.load(os.path.join(self.data_folder_path,
                                             self.data_set,
                                             'gt',
                                             name+self.ext), map_location=self.device))
                
        x = torch.vstack(x)
        y = torch.vstack(y)
        return x,y

train_dataset = IAM_dataset(train_names, 'train', device=config['device'])
train_indices = DataLoader(range(len(train_dataset)), batch_size=config['batch_size'], shuffle=True)

valid_dataset = IAM_dataset(valid_names, 'valid', device=config['device'])
valid_indices = DataLoader(range(len(valid_dataset)), batch_size=config['batch_size'], shuffle=True)

# Load the processor 
processor = DonutProcessor.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/processor")
# model = VisionEncoderDecoderModel.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/model")



def keep_all_but_tokens(str, tokens):
    """
    Remove all layout tokens from string
    """
    return re.sub('([' + tokens + '])', '', str)

def format_string_for_cer(str, layout_tokens):
    """
    Format string for CER computation: remove layout tokens and extra spaces
    """
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([\n])+', "\n", str)  # remove consecutive line breaks
    str = re.sub('([ ])+', " ", str).strip()  # remove consecutive spaces
    return str

def edit_cer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at character level
    """
    gt = format_string_for_cer(gt, layout_tokens)
    pred = format_string_for_cer(pred, layout_tokens)
    return editdistance.eval(gt, pred)

# Exemple d'utilisation
# non_character_tokens = ["ⓟ","Ⓟ"]
# metrics["edit_chars"] = [edit_cer_from_string(u, v, non_character_tokens) for u, v in zip(values["str_y"], values["str_x"])]
# metrics["nb_chars"] = [nb_chars_cer_from_string(gt, non_character_tokens) for gt in values["str_y"]]
