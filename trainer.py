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
import logging
from torch.optim import AdamW
import json

from metrics import edit_cer_from_string, edit_wer_from_string

logger = logging.getLogger(__name__)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)


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
  "part":"encoder",
  "mean":[0.485, 0.456, 0.406],
  "std":[0.229, 0.224, 0.225],
  "image_size":[1920, 2560],
  "max_length":224,
  "batch_size":3,
  "learning_rate":5e-6,
  "device":'cuda' if torch.cuda.is_available() else 'cpu',
  "epochs":30
}

# dataset
class IAM_dataset(Dataset):
    def __init__(self, 
                 paths,
                 data_set, 
                 data_folder_path = "/gpfsstore/rech/jqv/ubb84id/data/IAM/2560_1920",
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

# Load the model 
processor = DonutProcessor.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/processor")
model = VisionEncoderDecoderModel.from_pretrained("/gpfsstore/rech/jqv/ubb84id/output_models/model")

processor.image_processor.size = config['image_size']
processor.image_processor.mean = config['mean']
processor.image_processor.std = config['std']
processor.image_processor.do_align_long_axis = False
processor.image_processor.do_resize = False
tokenizer = processor.tokenizer
sepcial_tokens = tokenizer.special_tokens_map.values()

# Adjust our image size and output sequence lengths
model.config.encoder.image_size = config['image_size'][::-1] # (height, width)
model.config.decoder.max_length = config['max_length']

# Add task token for decoder to start
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(['<s>'])[0]


# wandb
KEY = "001b0cc1cdeb60216984dcc127298026eb6a9e8e"
PROJECT_NAME = "Donut_OCR"
os.environ["WANDB_MODE"]="offline"
wandb.login(key = KEY)
run = wandb.init(project="Donut",
                 entity="lazizhamdi",
                 config=config)

# training forloop
model.to(config['device'])
# train only the decoder
for p in model.decoder.parameters():
    p.requires_grad = False
  
opt = AdamW(model.parameters(), lr=config['learning_rate'])
model.train()
step = 0
best_valid_loss = float('inf')
for epoch in tqdm(range(config['epochs'])):
    for batch in tqdm(train_indices):
        # training
        x_train,y_train = train_dataset[batch]
        output = model(**{'pixel_values':x_train, 'labels':y_train})
        train_loss = output.loss
        train_loss.backward()
        opt.step()
        # log & eval
        if step % 8 == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(valid_indices))
                x_valid,y_valid = valid_dataset[batch]
                output = model(**{'pixel_values':x_valid, 'labels':y_valid})
                valid_loss = output.loss.mean().item()
                # get pred text
                preds = output.logits.argmax(-1)[0].detach().cpu()
                tokens = tokenizer.convert_ids_to_tokens(preds)
                pred_text = tokenizer.convert_tokens_to_string([t for t in tokens if t not in sepcial_tokens])
                # gt text
                y_valid = y_valid[0].detach().cpu()
                tokens = tokenizer.convert_ids_to_tokens(y_valid)
                text = tokenizer.convert_tokens_to_string([t for t in tokens if t not in sepcial_tokens])
                cer = edit_cer_from_string(text, pred_text)/len(text)
                wer = edit_wer_from_string(text, pred_text)/len(text.split())
                run.log({"train_loss":train_loss.mean().item(), "valid_loss":valid_loss, "cer":cer, "wer":wer})
                print("img size {}, Train loss {}, valid loss {}, cer {}, wer {}".format(x_valid.shape, train_loss.mean().item(), valid_loss, cer, wer))
        step += 1
    #### compte loss over valid if valid loss is better save checkpoints
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        batch = next(iter(valid_indices))
        x_valid,y_valid = valid_dataset[batch]
        output = model(**{'pixel_values':x_valid, 'labels':y_valid})
        valid_loss += output.loss.mean().item()
    valid_loss = valid_loss/len(valid_indices)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        output_folder_name = "encoder_lr{}_h{}_w{}".format(config['learning_rate'], config['image_size'][1], config['image_size'][0])
        model.save_pretrained("/gpfsstore/rech/jqv/ubb84id/output_models/"+output_folder_name)
        with open("/gpfsstore/rech/jqv/ubb84id/output_models/"+output_folder_name+"/info.txt", "w") as f:
            f.write("checkpoints created at epoch: {} with train loss : {} and valid loss : {}".format(epoch, train_loss, best_valid_loss))
            print("checkpoints created at epoch: {} with train loss : {} and valid loss : {}".format(epoch, train_loss, best_valid_loss))
    model.train()
    
run.finish()
