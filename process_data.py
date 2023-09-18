import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch 
from transformers import DonutProcessor
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import json 

data_folder_path = "/gpfsstore/rech/jqv/ubb84id/data/IAM"
sub_folder_name = "IAM_page_sem"
json_name = "formatted-IAM-DB-subwords-bart.json"
imgs_folder = "flat-300dpi"

# load the data
content = json.load(os.path.join(data_folder_path, sub_folder_name, json_name))
charset = content['charset']
ground_truth = content['ground_truth']
train_set, valid_set, test_set = ground_truth['train'], ground_truth['valid'], ground_truth['test']
train_names = list(train_set.keys())
valid_names = list(valid_set.keys())
test_names = list(test_set.keys())

# processing parameters
config = {
    "mean":[0.485, 0.456, 0.406],
    "std":[0.229, 0.224, 0.225],
    "image_size":[960, 1280],
    "max_length":224
}

# load donut processor
processor = DonutProcessor.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/processor")

mean = config['mean']
std = config['std']
image_size = config['image_size']
max_length = config['max_length'] # the maximum length found on the data 178

processor.image_processor.size = image_size
processor.image_processor.mean = mean
processor.image_processor.std = std
processor.image_processor.do_align_long_axis = False
processor.image_processor.do_resize = False
tokenizer = processor.tokenizer


# create output paths
ext = '.pt'
if os.path.exists(os.path.join(data_folder_path, '1280_960')) == False:
  os.mkdir(os.path.join(data_folder_path, '1280_960'))
  for d in ['train', 'valid','test']:
    os.mkdir(os.path.join(data_folder_path, '1280_960',d))
    for m in ['images','gt']:
      os.mkdir(os.path.join(data_folder_path, '1280_960',d,m))

# processing     
for names,dt_set,out_set in zip([train_names, valid_names, test_names], [train_set, valid_set, test_set], ['train','valid','test']):
  for name in tqdm(names):
    gt = dt_set[name]['pages'][0]["text"][1:-1]
    img = plt.imread(os.path.join(data_folder_path, sub_folder_name, imgs_folder, name))
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = cv.resize(img, tuple(image_size), cv.INTER_AREA)
    inputs = processor(
        img,
        text = gt,
        add_special_tokens=True,
        max_length=config['max_length'],
        padding="max_length",
        truncation=False,
        return_tensors = 'pt',
    )
    # save
    torch.save(inputs['pixel_values'], os.path.join(data_folder_path, '1280_960/images', out_set, name.split('.')[0]+ex))
    torch.save(inputs['labels'], os.path.join(data_folder_path, '1280_960/gt', out_set, name.split('.')[0]+ex))
