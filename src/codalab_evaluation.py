import os
import datetime
import pandas as pd
import numpy as np

from dataloader import ChestImages, image_transforms, train_model, device
from config import paths, P

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils.data import DataLoader, Dataset
# from utils import load_and_resize_img
import argparse
from PIL import Image
from dataloader import image_transforms

class EvaluationData(Dataset):
    def __init__(self, file_path, transform=None, clip_limit=None):
        self.df = pd.read_csv(file_path)
        self.transform = transform
        self.clip_limit = clip_limit
        
    @property
    def classes(self):
        return ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    
    def __len__(self):
        return self.df.index.size
    
    def __getitem__(self, index):
        image_path = self.df.loc[index].Path
        image_path_short = "/".join(image_path.split("/")[1:])
        print(image_path_short)
        # image = Image.fromarray(load_and_resize_img(image_path_short))
        image = Image.open(image_path_short)

        if self.clip_limit:
            image_eq = exposure.equalize_adapthist(np.array(image), clip_limit=self.clip_limit)
            image = Image.fromarray(np.uint8(plt.cm.Greys_r(image_eq)*255))

        if self.transform:
            image = self.transform(image)
        
        return {"image": image, "image_path": "/".join(image_path.split("/")[:-1])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', action="store", type=str)
    parser.add_argument('output_file', action="store", type=str)

    args = parser.parse_args()

    dataset = EvaluationData(args.input_file, transform=image_transforms, clip_limit=P.clip_limit)
    dataloader = DataLoader(dataset, 
            batch_size=P.batch_size, 
            shuffle=P.shuffle, 
            num_workers=P.num_workers)

    torch.hub.set_dir(".")
    model = P.model(5)
    checkpoint = torch.load("model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    all_probs = []
    all_paths = []
    
    for i, data in enumerate(dataloader):
        inputs, image_paths = data["image"], data["image_path"]
        inputs = inputs.to(device)
        all_paths += [image_paths]    
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs += [probs]
                
    all_probs = torch.cat(all_probs, dim=0).detach().numpy()

    df1 = pd.DataFrame(np.array(all_paths).reshape(-1,1), columns=["Study"])
    df2 = pd.DataFrame(all_probs, columns=dataset.classes)
    df = pd.concat((df1, df2), axis=1)\
        .groupby("Study")\
        .max()\
        .reset_index()
    
    df.to_csv(args.output_file, index=False)