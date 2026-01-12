# Scikit-learn imports
from sklearn.datasets import fetch_olivetti_faces
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
    """Set random seeds for reproducibility across libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Olivetti Faces baseline training')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--model_name', type=str, default="mixer_continual_baseline",
                        help='model name / identifier')
    
    ## unlearn face id
    parser.add_argument('--unlearn_class', type=int, default=2,
                        help='class id to remove from training set (simulate unlearning)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='validation set size ratio (default: 0.2)')

    ## train configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU id')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

set_seed(options.seed)


# Define data augmentation pipeline
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.2)),
    transforms.ToTensor()
])


# Load Olivetti faces dataset
X, y = fetch_olivetti_faces(shuffle=True, return_X_y=True)
X, y = X / 1.0, y


# Select only the first 6 classes (0–5)
selected_classes = list(range(6))
selected_mask = np.isin(y, selected_classes)
X_selected = X[selected_mask]
y_selected = y[selected_mask]


# Convert to PyTorch tensors
X = torch.tensor(X_selected.reshape(-1, 1, 64, 64), dtype=torch.float32)
y = torch.tensor(y_selected, dtype=torch.long)


# Perform data augmentation (30x per sample: 1 original + 29 augmented)
augmented_X, augmented_y = [], []

for idx in range(len(X)):
    img = X[idx].numpy().transpose((1, 2, 0))
    label = y[idx].item()
    augmented_X.append(X[idx].numpy())
    augmented_y.append(label)
    for _ in range(29):  # 29 augmentations + 1 original = 30 images per sample
        augmented_img = data_transforms(img).numpy()
        augmented_X.append(augmented_img)
        augmented_y.append(label)


# Convert augmented data back to tensors
augmented_X = torch.tensor(augmented_X, dtype=torch.float32).reshape(-1, 1, 64, 64)
augmented_y = torch.tensor(augmented_y, dtype=torch.long)


# ===== Key change: use single train/val split instead of cross-validation =====
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


# ===== Remove target class from training set after split =====
remove_class = options.unlearn_class  # Class to exclude from training (simulate unlearning)

# Create mask to keep samples that do NOT belong to the removed class
train_mask = (y_train != remove_class)

# Apply mask to filter training data
X_train = X_train[train_mask]
y_train = y_train[train_mask]


# Data preprocessing: resize images to 16×16
X_train_resized = F.interpolate(X_train, size=(16, 16), mode='bilinear', align_corners=False).to(device)
X_val_resized   = F.interpolate(X_val,   size=(16, 16), mode='bilinear', align_corners=False).to(device)
y_train = y_train.to(device)
y_val   = y_val.to(device)


# Define MLP-Mixer model
model = MLP_Mixer(
    n_layers        = 2,
    n_channel       = 20,
    n_hidden        = 32,
    n_output        = 6,
    image_size      = 16,
    patch_size      = 4,
    n_image_channel = 1   
).to(device)


# Load pretrained weights (from scratch unlearning checkpoint)
best_model_state = torch.load("./checkpoint/model_unlearn_finetune.pth")
model.load_state_dict(best_model_state)


# Print model summary
summary(model, 
        input_size=(1, 1, 16, 16),
        col_names=("input_size", "output_size", "num_params"),
        verbose=1)


# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
loss_function = nn.CrossEntropyLoss()


# Start training
train_loss, val_loss, train_acc, val_acc = train(
    training_features = X_train_resized,
    training_labels   = y_train,
    test_features     = X_val_resized,
    test_labels       = y_val,
    model             = model,
    optimizer         = optimizer,
    loss_function     = loss_function,
    epochs            = options.epochs,
    args              = options,
    device            = device
)