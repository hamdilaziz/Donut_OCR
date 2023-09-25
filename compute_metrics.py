import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
from transformers import VisionEncoderDecoderModel, DonutProcessor
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import json
from metrics import edit_cer_from_string, edit_wer_from_string

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
  "model_names":["donut_decoder_cross_lr1e-06_h2560_w1920", "donut_decoder_after_encoder_lr1e-06_h2560_w1920","donut_encoder_after_decoder_lr1e-06_h2560_w1920",
                 "donut_encoder_only_lr1e-06_h2560_w1920"],
  "mean":[0.485, 0.456, 0.406],
  "std":[0.229, 0.224, 0.225],
  "image_size":[1920, 2560],
  "max_length":224,
  "batch_size":1,
  "device":'cuda' if torch.cuda.is_available() else 'cpu',
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


test_dataset = IAM_dataset(test_names, 'test', device=config['device'])
test_indices = DataLoader(range(len(test_dataset)), batch_size=config['batch_size'], shuffle=True)

# Load the processor 
processor = DonutProcessor.from_pretrained("/gpfsstore/rech/jqv/ubb84id/huggingface_models/donut/processor")
tokenizer = processor.tokenizer

# evaluation
sepcial_tokens = tokenizer.special_tokens_map.values()
cer = {}
f = open('metrics_result/metrics.txt', mode='w')
for model_name in config['model_names']:
    model = VisionEncoderDecoderModel.from_pretrained(os.path.join(config["path"], model_name))
    cer_list = []
    wer_list = []
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
    
            tokens = tokenizer.convert_ids_to_tokens(y_test[0].detach().cpu())
            text = tokenizer.convert_tokens_to_string([t for t in tokens if t not in sepcial_tokens])
            
            pred_tokens = tokenizer.convert_ids_to_tokens(preds[0])
            pred_text = tokenizer.convert_tokens_to_string([t for t in pred_tokens if t not in sepcial_tokens])
            cer = edit_cer_from_string(text, pred_text)/len(text)
            wer = edit_wer_from_string(text, pred_text)/len(text)
            
            cer_list.append(cer)
            wer_list.append(wer)
                
    f.write("Model : {}, CER : {}, WER : {}\n".format(model_name, np.mean(cer_list), np.mean(wer_list)))
    print("Model : {}, CER : {}, WER : {}".format(model_name, np.mean(cer_list), np.mean(wer_list)))
                                                      
f.close()
