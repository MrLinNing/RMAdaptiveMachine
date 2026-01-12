# Scikit-learn
from sklearn.datasets import fetch_olivetti_faces
# Python
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np
from utils import train
import argparse
import os
from torchvision import transforms
from model_mixer import MLP_Mixer
from sklearn.model_selection import train_test_split

from torchinfo import summary


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    
    parser = argparse.ArgumentParser(description='PyTorch Olivetti Faces unlearning')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    
    parser.add_argument('--model_name', type=str, default="learn", 
                        help='model name (default: mixer)')

    ## unlearn face id
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='validation set size ratio (default: 0.2)')

    ## train configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')


    args = parser.parse_args()
    return args

options = parse_args()

print(options)


os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


set_seed(options.seed)

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.2)),
    transforms.ToTensor()
])

X, y = fetch_olivetti_faces(shuffle=True, return_X_y=True)
X, y = X / 1.0, y


selected_classes = list(range(5))
selected_mask = np.isin(y, selected_classes)
X_selected = X[selected_mask]
y_selected = y[selected_mask]


X = torch.tensor(X_selected.reshape(-1, 1, 64, 64), dtype=torch.float32)
y = torch.tensor(y_selected, dtype=torch.long)


augmented_X, augmented_y = [], []

for idx in range(len(X)):
    img = X[idx].numpy().transpose((1, 2, 0))
    label = y[idx].item()
    augmented_X.append(X[idx].numpy())
    augmented_y.append(label)
    for _ in range(29):  
        augmented_img = data_transforms(img).numpy()
        augmented_X.append(augmented_img)
        augmented_y.append(label)


augmented_X = torch.tensor(augmented_X, dtype=torch.float32).reshape(-1, 1, 64, 64)
augmented_y = torch.tensor(augmented_y, dtype=torch.long)


X_train, X_val, y_train, y_val = train_test_split(
    augmented_X, 
    augmented_y, 
    test_size=options.test_size,
    random_state=options.seed,
    stratify=augmented_y.numpy()
)

print(f"\nDataset split:")
print(f"Training set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")

X_train_resized = F.interpolate(X_train, size=(16, 16), mode='bilinear', align_corners=False).to(device)
X_val_resized = F.interpolate(X_val, size=(16, 16), mode='bilinear', align_corners=False).to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)

model = MLP_Mixer(
        n_layers    = 2,     
        n_channel   = 20,    
        n_hidden    = 32,     
        n_output    = 6,          
        image_size  = 16,        
        patch_size  = 4,   
        n_image_channel=1   
        ).to(device)

summary(model, 
        input_size=(1, 
                   1, 
                   16, 
                   16),
        col_names=("input_size", "output_size", "num_params"),
        verbose=1)


optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
loss_function = nn.CrossEntropyLoss()

train_loss, val_loss, train_acc, val_acc = train(
    training_features=X_train_resized,
    training_labels=y_train,
    test_features=X_val_resized,
    test_labels=y_val,
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    epochs=options.epochs,
    args = options,
    device=device)


