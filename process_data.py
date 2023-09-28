# import librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm.notebook import tqdm
import cv2 as cv
import json 
# deep learning lb
import torch 
import torchvision.transforms.functional as fn
from transformers import DonutProcessor
from torch.utils.data import Dataset, DataLoader

# original data variables
data_folder_path = "/gpfsstore/rech/jqv/ubb84id/data/IAM"
sub_folder_name = "IAM_page_sem"
json_name = "formatted-IAM-DB-subwords-bart.json"
imgs_folder = "flat-300dpi"

# load the data
with open(os.path.join(data_folder_path, sub_folder_name, json_name)) as f:
    content = json.load(f)
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
    "image_size":[1920, 2560],
    "max_length":224
}

# load donut processor
processor = DonutProcessor.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/processor")
processor.image_processor.size = config['image_size']
processor.image_processor.mean = config['mean']
processor.image_processor.std = config['std']
processor.image_processor.do_align_long_axis = False
processor.image_processor.do_resize = False
tokenizer = processor.tokenizer

# create output paths
ext = '.pt'
data_folder_name = "{}_{}".format(config['image_size'][1], config['image_size'][0])
if os.path.exists(os.path.join(data_folder_path, data_folder_name)) == False:
  os.mkdir(os.path.join(data_folder_path, data_folder_name))
  for d in ['train', 'valid','test']:
    os.mkdir(os.path.join(data_folder_path, data_folder_name,d))
    for m in ['images','gt']:
      os.mkdir(os.path.join(data_folder_path, data_folder_name,d,m))

# processing     
for names,dt_set,out_set in zip([train_names, valid_names, test_names], [train_set, valid_set, test_set], ['train','valid','test']):
    for name in names:
        gt = dt_set[name]['pages'][0]["text"][1:-1]
        img = plt.imread(os.path.join(data_folder_path, sub_folder_name, imgs_folder, name))
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = cv.resize(img, tuple(config['image_size']), cv.INTER_AREA)
        tensor_image = fn.to_tensor(img)
        normalized = fn.normalize(tensor_image, mean=config['mean'], std=config['std'])
        normalized = normalized.unsqueeze(0)
        labels = tokenizer(gt,
                           add_special_tokens=True,
                           max_length=config['max_length'],
                           padding="max_length",
                           truncation=False,
                           return_tensors = 'pt')
        torch.save(normalized, os.path.join(data_folder_path, data_folder_name, 'train','images', name.split('.')[0]+ext))
        torch.save(labels['input_ids'], os.path.join(data_folder_path, data_folder_name, out_set,'gt', name.split('.')[0]+ext))
    
# for names,dt_set,out_set in zip([train_names, valid_names, test_names], [train_set, valid_set, test_set], ['train','valid','test']):
#   for name in tqdm(names):
#     gt = dt_set[name]['pages'][0]["text"][1:-1]
#     img = plt.imread(os.path.join(data_folder_path, sub_folder_name, imgs_folder, name))
#     img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
#     img = cv.resize(img, tuple(config['image_size']), cv.INTER_AREA)
#     tensor_image = fn.to_tensor(img)
#     normalize = fn.normalize(tensor_image, mean=config['mean'], std=config['std'])
#     normalize = normalize.unsqueeze(0)
#     labels = tokenizer(gt,
#                        add_special_tokens=True,
#                        max_length=config['max_length'],
#                        padding="max_length",
#                        truncation=False,
#                        return_tensors = 'pt')
#     # save
#     torch.save(normalize, os.path.join(data_folder_path, data_folder_name, out_set,'images', name.split('.')[0]+ext))
#     torch.save(labels, os.path.join(data_folder_path, data_folder_name, out_set,'gt', name.split('.')[0]+ext))
