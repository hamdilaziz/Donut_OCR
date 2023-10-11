import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
from transformers import VisionEncoderDecoderModel, DonutProcessor
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import pickle
from metrics import edit_cer_from_string, edit_wer_from_string

data_folder_path = "/gpfsstore/rech/jqv/ubb84id/data/RIMES/RIMES_page_sem"
pkl_name = "/gpfsstore/rech/jqv/ubb84id/data/RIMES/RIMES_page_sem/labels-subwords.pkl"

# load the data
with open(os.path.join(data_folder_path, pkl_name), mode='rb') as f:
    content = pickle.load(f)
charset = content['charset']
ground_truth = content['ground_truth']
train_set, valid_set, test_set = ground_truth['train'], ground_truth['valid'], ground_truth['test']
train_names = list(train_set.keys())
valid_names = list(valid_set.keys())
test_names = list(test_set.keys())

models_path = "/gpfsstore/rech/jqv/ubb84id/output_models/RIMES"
# parameters    
config = {
  "model_names": ["decoder_lr1e-06_h2560_w1920",
                  "encoder_lr1e-06_h2560_w1920"
                 ],#os.listdir(models_path),
  "mean":[0.485, 0.456, 0.406],
  "std":[0.229, 0.224, 0.225],
  "image_size":[1920, 2560],
  "max_length":860,
  "batch_size":1,
  "device":'cuda' if torch.cuda.is_available() else 'cpu',
}

# dataset
class IAM_dataset(Dataset):
    def __init__(self, 
                 paths,
                 data_set, 
                 data_folder_path = "/gpfsstore/rech/jqv/ubb84id/data/RIMES/RIMES_page_sem/2560_1920",
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

sepcial_tokens = tokenizer.special_tokens_map.values()
f = open('metrics_result/metrics.txt', mode='w')
for model_name in config['model_names']:
    model = VisionEncoderDecoderModel.from_pretrained(os.path.join(models_path, model_name))
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
    
            tokens = tokenizer.convert_ids_to_tokens(y_test[0].detach().cpu())
            text = tokenizer.convert_tokens_to_string([t for t in tokens if t not in sepcial_tokens])
            
            pred_tokens = tokenizer.convert_ids_to_tokens(preds[0])
            pred_text = tokenizer.convert_tokens_to_string([t for t in pred_tokens if t not in sepcial_tokens])
            cer = edit_cer_from_string(text, pred_text)/len(text)
            wer = edit_wer_from_string(text, pred_text)/len(text.split())
            
            cer_list.append(cer)
            wer_list.append(wer)
                
    f.write("Model : {}, CER : {}, WER : {} \n".format(model_name, np.mean(cer_list), np.mean(wer_list)))
                                                      
f.close()
