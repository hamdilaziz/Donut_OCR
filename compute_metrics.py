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

#####################" cer thomas
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

####################################
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
  "batch_size":1,
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


test_dataset = IAM_dataset(test_names, 'test', device=config['device'])
test_indices = DataLoader(range(len(test_dataset)), batch_size=config['batch_size'], shuffle=True)

# Load the processor 
processor = DonutProcessor.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/processor")
tokenizer = processor.tokenizer

# evaluation
sepcial_tokens = tokenizer.special_tokens_map.values()
cer = {}
f = open('result.txt', mode='w')
for model_name in config['model_names']:
    model = VisionEncoderDecoderModel.from_pretrained(os.path.join(config["path"], model_name))
    cer_list = []
    model.eval()
    model.to(config['device'])
    with torch.no_grad():
        for batch in tqdm(test_indices):
            x_test,y_test = test_dataset[batch]
            output = model(**{'pixel_values':x_test, 'labels':y_test})
            # test_loss = output.loss.mean().item()
            logits = output.logits
            preds = logits.argmax(-1).detach().cpu()
    
            # for i in range(config['batch_size']):
            # img = np.moveaxis(x_test[i].detach().cpu().numpy(), 0,2)
            i = 0
            tokens = tokenizer.convert_ids_to_tokens(y_test[i].detach().cpu())
            text = tokenizer.convert_tokens_to_string([t for t in tokens if t not in sepcial_tokens])
            
            pred_tokens = tokenizer.convert_ids_to_tokens(preds[i])
            pred_text = tokenizer.convert_tokens_to_string([t for t in pred_tokens if t not in sepcial_tokens])
            cer_list.append(edit_cer_from_string(text, pred_text)/len(text))
                
    f.write("Model : {}, CER : {}\n".format(model_name, np.mean(cer_list)))
                                                      

    
f.close()
