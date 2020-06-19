import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils, datasets, models

import time
import copy
import os
from collections import namedtuple

from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from skimage import exposure
import matplotlib.pyplot as plt

from config import P

age_min, age_max, age_mean = 18., 90., 60.43
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_metrics(outputs, targets):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        # precision[i], recall[i], _ = precision_recall_curve(targets[:,i], outputs[:,i])
        # fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall}
    return metrics

class ChestImages(Dataset):
    def __init__(self, file_path, root_dir, P=None, frac=None, classes_type=None):
        assert P.upolicy in ["one", "zero", "ignore", "multi-class"], "Incorrect choice for upolicy"
        assert P.classes_type in ["all", "subset"], "Incorrect choice for classes_type"
        df = pd.read_csv(os.path.join(root_dir, file_path))\
            .sample(frac=frac)
        df.index = np.arange(0, len(df.index))
        self.P = P
        self.df = df
        self.root_dir = root_dir
        self.filepath = file_path
        self.transform = transforms.Compose([
            transforms.Resize(P.resize),
            transforms.RandomCrop(P.crop),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes_type = classes_type

    @property
    def classes(self):
        return {
            "all": ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
               'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
               'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
               'Pleural Other', 'Fracture', 'Support Devices'],
            "subset": ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        }
    
    @property
    def attributes(self):
        return ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']

    def _targets(self, d):
        targets = [0 if np.isnan(d.get(_class)) else d.get(_class) for _class in self.classes[self.classes_type]]
        targets = np.array(targets, np.float32)
        if self.P.upolicy == "one":
            return np.where(targets == -1, 1.0, targets)
            # targets = [target if target in [0,1] else 1 for target in targets]
        elif self.P.upolicy == "zero":
            return np.where(targets == -1, 0.0, targets)
        else:
            return targets
        
            # targets = [target if target in [0,1] else 0 for target in targets]
        # return np.array(targets, np.float32)
    
    def __len__(self):
        return self.df.index.size
    
    def __getitem__(self, index):
        image_dict = self.df.loc[index].to_dict()
        image = Image.open(os.path.join(self.root_dir, image_dict.get("Path")))
        if self.P.clip_limit:
            image_eq = exposure.equalize_adapthist(np.array(image), clip_limit=self.P.clip_limit)
            image = Image.fromarray(np.uint8(plt.cm.Greys_r(image_eq)*255))
        targets = self._targets(image_dict)

        age_raw = image_dict.get("Age")
        if age_raw == 0:
            age = torch.tensor(0.)
        else:
            age = (age_raw - age_mean) / (age_max - age_min)
            age = torch.tensor(age)

        if image_dict.get("Sex") == "F":
            sex = torch.tensor(1.)
        else:
            sex = torch.tensor(0.)
        
        if image_dict.get("Frontal/Lateral") == "Frontal":
            frontal_lateral = torch.tensor(1.)
        else:
            frontal_lateral = torch.tensor(0.)
        
        if image_dict.get("AP/PA") in ["AP", "LL", "RL"]:
            ap = torch.tensor(1.)
        else:
            ap = torch.tensor(0.)
            
        if image_dict.get("AP/PA") == "PA":
            pa = torch.tensor(1.)
        else:
            pa = torch.tensor(0.)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "sex": sex,
            "frontal_lateral": frontal_lateral,
            "ap": ap,
            "pa": pa,
            "age": age,
            "targets": torch.tensor(targets)
        }

def evaluate_singlemodel(model, criterion, dataloader):
    eval_loss = 0.0
    model.eval()
    all_probs = []
    all_labels = []
    for i, data in enumerate(dataloader):
        inputs, labels = data["image"], data["targets"]
        
        labels = labels.to(device)
        inputs = inputs.to(device)

        
        all_labels += [labels.cpu()]
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs += [probs.cpu()]
            loss = criterion(outputs, labels)
        
        eval_loss += loss.item() * inputs.size(0)
    
    eval_loss = eval_loss / dataloader.dataset.__len__()

    return all_probs, all_labels, eval_loss

def evaluate_multiplemodels(model, criterion, dataloader, outdir):
    eval_loss = 0.0

    all_probs = []
    all_losess = []
    for file in os.listdir(outdir):
        if file[-3:] == ".pt":
            checkpoint = torch.load(os.path.join(path, file))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        _probs, labels, _eval_loss = evaluate_singlemodel(model, criterion, dataloader)
    
        all_probs += [_probs]
        all_labels += [_labels]
        all_losess += [_eval_loss]
    
    all_probs = torch.stack(all_probs, dim=2).mean(2)
    all_losess = torch.stack(all_losess, dim=2).mean(2)

    all_metrics = compute_metrics(all_probs, labels)
    return all_metrics, all_losess

def train_model(model, dataloaders, outdir, criterion, optimizer, scheduler, writer, P, modelpath=None, num_epochs=None, maxmodels=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e6
    modelinfo = namedtuple("Model", ("time", "path", "auc"))
    modelsinfo = [modelinfo(i, "output", 0.0) for i in range(maxmodels)]

    if modelpath:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint["model_state_dict"])

    class_names = dataloaders["train"].dataset.classes[P.classes_type]
    n_classes = len(class_names)
    global_step = 0
    # model.train()  

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        running_loss = 0.0
        # running_corrects = torch.zeros(n_classes).to(device)
        all_probs, all_labels = [], []

        for i, data in enumerate(dataloaders["train"]):
            model.train()
            
            inputs, labels = data["image"], data["targets"]
            inputs = inputs.to(device)
            labels = labels.to(device)
            global_step += labels.size(0)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                if P.upolicy == "ignore":
                    outputs_f = outputs.flatten()
                    labels_f = labels.flatten()

                    mask = labels_f != -1
                    outputs_f, labels_f = outputs_f[mask], labels_f[mask]
                    loss = criterion(outputs_f, labels_f)
                else:
                    loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)

                all_probs += [probs]
                all_labels += [labels]
                
                loss.backward()
                optimizer.step()

                # writer.add_scalar("loss/train", loss, global_step=global_step)

            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels, 0)
            if (i % P.printevery == 0) or (i==len(dataloaders["train"])-1):
                time_since = time.time() - since 
                print("Phase: Train, Epoch: {:2}, Iteration: {:2}/{}, Progress: {:.0f}%, Loss: {:.4f}, Time: {:.0f}m {:.0f}s".format( 
                    epoch+1, i, len(dataloaders["train"]), 100. * (i) / len(dataloaders["train"]), loss.item(), time_since // 60, time_since % 60))
            #if i % P.evaluateevery == 0:
                all_probs_valid, all_labels_valid, eval_loss = evaluate_singlemodel(model, criterion, dataloaders["valid"])
                all_probs_valid = torch.cat(all_probs_valid, dim=0).detach().numpy()
                all_labels_valid = torch.cat(all_labels_valid, dim=0).detach().numpy()
                eval_metrics = compute_metrics(all_probs_valid, all_labels_valid)
                eval_auc = roc_auc_score(all_labels_valid.ravel(), all_probs_valid.ravel())

                for j in range(n_classes):
                    writer.add_scalar("AUC_valid/{}".format(class_names[j]), eval_metrics["aucs"][j], global_step=global_step)
                    print("AUC for {:30} = {:.3f}".format(class_names[j], eval_metrics["aucs"][j]))
                writer.add_scalar("AUC_valid/overall", eval_auc, global_step=global_step)
                writer.add_scalars("loss", {"train": loss.item(), "valid": eval_loss}, global_step=global_step)
            
                if eval_auc > modelsinfo[-1].auc:
                    if modelsinfo[-1].path != "output": # delete the worst existing model
                        os.remove(os.path.join(outdir, modelsinfo[-1].path))
                    
                    timenow = int(round(time.time(), 0))
                    outfile = "epoch{}_itr{}.pt".format(epoch+1, i)
                    modelsinfo[-1] = modelinfo(timenow, outfile, eval_auc)
                    modelsinfo = sorted(modelsinfo, key=lambda model: model.auc, reverse=True)

                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "valid_loss": eval_loss,
                        "epoch": epoch,
                        "global_step": global_step,
                        "params": P,
                        "classes": class_names,
                        "labels": all_labels_valid,
                        "probs": all_probs_valid},
                        # "models": modelsinfo}, 
                        os.path.join(outdir, outfile))

        scheduler.step()

        epoch_loss = running_loss / len(dataloaders["train"])
        print('Train Loss: {:.4f}'.format(epoch_loss))

        print()

    best_auc = modelsinfo[0].auc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_auc))

    # model.load_state_dict(best_model_wts)
    return modelsinfo