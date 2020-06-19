from models import DenseNet121, GoogleNet, ResNet152, ResNet18
import os

paths = {
    "posix": {
        "small": {
            "dirpath": "/Users/ashok/Downloads/Chexpert/",
            "train_location": "CheXpert-v1.0-small/valid.csv",
            "valid_location": "CheXpert-v1.0-small/valid.csv",
        },
        "full": {
            "dirpath": None,
            "train_location": None,
            "valid_location": None,
        },
    },
    "nt": {
        "small": {
            "dirpath": "C:/Users/Ashok/Documents/MS/",
            "train_location": "CheXpert-v1.0-small/train.csv",
            "valid_location": "CheXpert-v1.0-small/valid.csv",
        },
        "full": {
            "dirpath": "E:/",
            "train_location": "CheXpert-v1.0/train.csv",
            "valid_location": "CheXpert-v1.0/valid.csv",
        },
    }
}

class P:
    model = ResNet18
    upolicy = "one"
    classes_type = "subset"
    resize = 320
    crop = 320
    clip_limit = None
    frac = 1
    batch_size = 64
    num_epochs = 6
    shuffle = True
    num_workers = 1
    lr = 0.00001
    step_size = 2
    gamma = 0.1
    dataset = "small" if os.name == "posix" else "small"
    printevery = 100
    evaluateevery = 5000 // batch_size
    maxmodels = 10
    retrain_model = None
    all_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
               'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
               'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
               'Pleural Other', 'Fracture', 'Support Devices']
    training_classes = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
