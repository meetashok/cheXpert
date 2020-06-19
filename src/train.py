import os
import datetime
import pandas as pd
import numpy as np

from dataloader import ChestImages, train_model, device
from config import paths, P

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils import tensorboard
from torch.utils.data import DataLoader

if __name__ == "__main__":
    print(device)
    torch.manual_seed(0)
    np.random.seed(0)
    
    suffix = datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_data = ChestImages(paths[os.name][P.dataset]["train_location"], 
        paths[os.name][P.dataset]["dirpath"], 
        P,
        frac=P.frac,
        classes_type=P.classes_type)

    valid_data = ChestImages(paths[os.name][P.dataset]["valid_location"], 
        paths[os.name][P.dataset]["dirpath"], 
        P,
        frac=1.0,
        classes_type=P.classes_type)

    dataloaders = {
        "train": DataLoader(train_data, 
            batch_size=P.batch_size, 
            shuffle=P.shuffle, 
            num_workers=P.num_workers),

        "valid": DataLoader(valid_data, 
            batch_size=P.batch_size, 
            shuffle=False, 
            num_workers=P.num_workers)
    }

    # classes_type = dataloaders["train"].dataset.classes_type
    classes = dataloaders["train"].dataset.classes[P.classes_type]
    nclasses = len(classes)

    model = P.model(nclasses)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=P.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=P.step_size, gamma=P.gamma)

    writer = tensorboard.SummaryWriter(log_dir="runs/{}_{}".format(model._get_name(), suffix))
    for k,v in P.__dict__.items():
        if not k.startswith("__"):
            writer.add_text("modelparams", "{}:{}".format(k, v))
    
    # If continued training
    retrain_model = P.retrain_model
    modelpath = os.path.join(paths[os.name][P.dataset]["dirpath"], "models", retrain_model) if retrain_model else None
    
    # Model outputfil
    outdir = os.path.join(paths[os.name][P.dataset]["dirpath"], "models", model._get_name() + "_" +suffix)
    os.mkdir(outdir)

    model = train_model(model, 
        dataloaders, 
        outdir, 
        criterion, 
        optimizer, 
        scheduler, 
        writer, 
        P,
        modelpath=modelpath, 
        num_epochs=P.num_epochs,
        maxmodels=P.maxmodels)
