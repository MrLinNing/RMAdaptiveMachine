# Scikit-learn
import os
import torch
from torch import nn
import torch.nn.functional as F  
import numpy as np
from model_mixer import MLP_Mixer, MLP_Mixer
from utils import train_with_gradient_ascent
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
# from lora_utils import print_params, get_updated_model
from lora_utils_multi import get_updated_model, print_params, LoRALinear_Attention

from sklearn.model_selection import train_test_split
from model_mixer import TokenMixingMLP, ChannelMixingMLP, OutputMLP

import argparse


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Olivetti Faces unlearning')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    ## unlearn face id
    parser.add_argument('--unlearn', type=str, default="5",
                        help='unlearning person id')
    parser.add_argument('--unweight', type=float, default=-0.5,
                        help='weight for unlearning class')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='validation set size ratio (default: 0.2)')
    
    parser.add_argument('--model_name', type=str, default="lora_unlearn",
                        help='model name / identifier')
    
    parser.add_argument('--unlearn_class', type=int, default=3, 
                        help='Unlearn class id, default is 3')

    ## train configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
set_seed(options.seed)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Unlearn ID set (example)
unlearn_id_set = [options.unlearn_class]  # Replace with actual unlearn ids if needed
print(f"unlearning people id set is {unlearn_id_set}")


# Define data augmentation transforms
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.2)),
    transforms.ToTensor()
])


# Load dataset
X, y = fetch_olivetti_faces(shuffle=True, return_X_y=True)
X, y = X / 1.0, y

# Select only the first 5 classes (0–4) for initial filtering
selected_classes = list(range(5))
selected_mask = np.isin(y, selected_classes)
X_selected = X[selected_mask]
y_selected = y[selected_mask]

# Convert to tensors
X = torch.tensor(X_selected.reshape(-1, 1, 64, 64), dtype=torch.float32)
y = torch.tensor(y_selected, dtype=torch.long)


# Data augmentation
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

# Convert back to tensors
augmented_X = torch.tensor(augmented_X, dtype=torch.float32).reshape(-1, 1, 64, 64)
augmented_y = torch.tensor(augmented_y, dtype=torch.long)


# ===== Key change: replace cross-validation with single split =====
# Create a single train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    augmented_X, 
    augmented_y, 
    test_size=options.test_size,
    random_state=options.seed,
    stratify=augmented_y.numpy()
)


# Filter out the sixth class (class 5) from the training set
train_classes = list(range(5))  # Classes 0–4
train_mask = torch.isin(y_train, torch.tensor(train_classes, dtype=torch.long, device=y_train.device))
X_train = X_train[train_mask]
y_train = y_train[train_mask]

# Debug: Check shapes and class distribution
print(f"\nDataset split:")
print(f"Training set size: {len(X_train)} samples (classes 0–4)")
print(f"Validation set size: {len(X_val)} samples (classes 0–5)")
print("Training set class distribution:")
for i in range(6):
    print(f"Class {i}: {(y_train == i).sum().item()} samples")
print("Validation set class distribution:")
for i in range(6):
    print(f"Class {i}: {(y_val == i).sum().item()} samples")


# Data preprocessing (resize to 16×16)
X_train_resized = F.interpolate(X_train, size=(16, 16), mode='bilinear', align_corners=False).to(device)
X_val_resized   = F.interpolate(X_val,   size=(16, 16), mode='bilinear', align_corners=False).to(device)
y_train = y_train.to(device)
y_val   = y_val.to(device)


# Model definition
model = MLP_Mixer(
    n_layers       = 2,          # 2         2           6
    n_channel      = 20,         # 20        20          128
    n_hidden       = 32,         # 64        64          128
    n_output       = 6,          # 10        10          100
    image_size     = 16,         # 28        32          32
    patch_size     = 4,          # 2         4           4
    n_image_channel= 1           # 1 
).to(device)

# Load pretrained weights
best_model_state = torch.load("./checkpoint/best_learn_model.pth")
model.load_state_dict(best_model_state)


print('train unlearning model')


# Prepare model for unlearning
print("\nOriginal model parameters:")
print_params(model)


token_only = (ChannelMixingMLP, OutputMLP)

# Create LoRA-adapted model
lora_model = get_updated_model(
    model,
    target_modules = token_only,
    rank           = 6,
    alpha          = 4.0,
    mode           = "add",
    lora_name      = "lora1",
    device         = device
)
print("\nUpdated model parameters:")
print_params(lora_model)


# Activate specific LoRA and set trainable parameters
for module in lora_model.modules():
    if isinstance(module, LoRALinear_Attention):
        module.set_active_lora("lora1")
        module.set_trainable_lora(["lora1"])


# Print gradient status of each parameter
print("\nParameter gradient status:")
for name, param in lora_model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")


# Training configuration
optimizer = torch.optim.Adam(lora_model.parameters(), lr=options.lr)
loss_function = nn.CrossEntropyLoss()


# Start unlearning training
print(f"\nStarting unlearning training for {options.epochs} epochs...")
train_loss, test_loss, train_acc, overall_test_acc, target_test_acc, other_test_acc = train_with_gradient_ascent(
    training_features = X_train_resized,
    training_labels   = y_train,
    test_features     = X_val_resized,
    test_labels       = y_val,
    model             = lora_model,
    optimizer         = optimizer,
    loss_function     = loss_function,
    epochs            = options.epochs,
    target_classes    = unlearn_id_set,
    class_weight_values = options.unweight,
    device            = device,
    args              = options,
)


# Print final performance summary
print(f"\n{'='*50}")
print(f"Final Training Accuracy:   {train_acc[-1]:.4f}")
print(f"Final Validation Accuracy: {overall_test_acc[-1]:.4f}")
print(f"Unlearned classes:         {unlearn_id_set}")
print(f"Unlearned accuracy:        {target_test_acc[-1]:.4f}")
print(f"Remaining accuracy:        {other_test_acc[-1]:.4f}")
print('='*50)