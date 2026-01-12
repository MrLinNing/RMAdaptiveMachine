# Scikit-learn
import os
import torch
from torch import nn
import torch.nn.functional as F  
import numpy as np
from model_mixer import MLP_Mixer, MLP_Mixer
from utils import train_from_scratch
from sklearn.datasets import fetch_olivetti_faces
from torchvision import transforms
from lora_utils_multi import print_params
from sklearn.model_selection import train_test_split

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
    parser = argparse.ArgumentParser(description='PyTorch Olivetti Faces unlearning (from scratch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    ## unlearn face id
    parser.add_argument('--unlearn', type=str, default="5",
                        help='unlearning person id')
    parser.add_argument('--unweight', type=float, default=-0.5,
                        help='weight for unlearning class')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='validation set size ratio (default: 0.2)')
    
    parser.add_argument('--model_name', type=str, default="unlearn_finetune",
                        help='model name / identifier')
    
    parser.add_argument('--unlearn_class', type=int, default=3, 
                        help='Unlearn class id, default is 3')

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
set_seed(options.seed)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Unlearn ID set
unlearn_id_set = [options.unlearn_class]  # Replace with actual unlearn ids if needed
print(f"Unlearning person ID set: {unlearn_id_set}")


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

# Select only the first 5 classes (0–4)
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


# === Key modification: create full array then remove samples from unlearn_id_set ===
# Convert augmented_y list to PyTorch tensor
augmented_y_tensor = torch.tensor(augmented_y, dtype=torch.long)

# Now we can safely use .numpy()
y_all = augmented_y_tensor.numpy()
augmented_X_array = np.array(augmented_X)

# Create full train/val split (including all classes initially)
X_train_full, X_val, y_train_full, y_val = train_test_split(
    augmented_X_array, 
    augmented_y_tensor.numpy(),           # NumPy array needed for stratification
    test_size=options.test_size,
    random_state=options.seed,
    stratify=augmented_y_tensor.numpy()   # stratification requires array
)

print(f"\nOriginal dataset size: {len(augmented_X_array)}")
print(f"Initial split results:")
print(f"  Training set size: {len(X_train_full)}")
print(f"  Validation set size: {len(X_val)}")


# Create mask: keep only samples NOT in unlearn_id_set for training
train_keep_mask = ~np.isin(y_train_full, unlearn_id_set)

# Apply mask to filter training data
X_train = X_train_full[train_keep_mask]
y_train = y_train_full[train_keep_mask]

print(f"\nFiltered training set size: {len(X_train)} (removed {len(X_train_full) - len(X_train)} samples)")
print(f"Validation set size remains: {len(X_val)} (contains all classes)")


# Convert back to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 64, 64)
X_val   = torch.tensor(X_val,   dtype=torch.float32).reshape(-1, 1, 64, 64)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)


# === Statistics ===
# Removed samples statistics
train_unlearn_mask = np.isin(y_train_full, unlearn_id_set)
train_unlearn_y = y_train_full[train_unlearn_mask]

print(f"\nRemoved samples from training set statistics:")
print(f"Total removed samples: {len(train_unlearn_y)}")
print(f"Removed class distribution:")
for unlearn_id in unlearn_id_set:
    count = np.sum(train_unlearn_y == unlearn_id)
    print(f"  Class {unlearn_id}: {count} samples")


# Validation set distribution
val_y_np = y_val.numpy()
val_unique, val_counts = np.unique(val_y_np, return_counts=True)
print(f"\nValidation set class distribution (includes all classes):")
for cls, count in zip(val_unique, val_counts):
    print(f"  Class {cls}: {count} samples")


# Training set distribution after removal
y_train_np = y_train.numpy() 
train_unique, train_counts = np.unique(y_train_np, return_counts=True)
print(f"\nTraining set class distribution after removal:")
for cls, count in zip(train_unique, train_counts):
    print(f"  Class {cls}: {count} samples")


# Final sizes
print(f"\nFinal dataset sizes:")
print(f"Training set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")


# Data preprocessing (resize to 16×16)
X_train_resized = F.interpolate(X_train, size=(16, 16), mode='bilinear', align_corners=False).to(device)
X_val_resized   = F.interpolate(X_val,   size=(16, 16), mode='bilinear', align_corners=False).to(device)
y_train = y_train.to(device)
y_val   = y_val.to(device)


# Define model
model = MLP_Mixer(
    n_layers       = 2,
    n_channel      = 20,
    n_hidden       = 32,
    n_output       = 6,
    image_size     = 16,
    patch_size     = 4,
    n_image_channel= 1 
).to(device)

# Optionally load pretrained weights (commented out)
best_model_state = torch.load("./checkpoint/best_learn_model.pth")
model.load_state_dict(best_model_state)


print('Training unlearning model from scratch')


# Prepare model for unlearning
print("\nOriginal model parameters:")
print_params(model)


# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
loss_function = nn.CrossEntropyLoss()


# Start unlearning training from scratch
print(f"\nStarting unlearning training for {options.epochs} epochs...")
train_loss, test_loss, train_acc, overall_test_acc, target_test_acc, other_test_acc = train_from_scratch(
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


# Print final performance summary
print(f"\n{'='*50}")
print(f"Final Training Accuracy:   {train_acc[-1]:.4f}")
print(f"Final Validation Accuracy: {overall_test_acc[-1]:.4f}")
print(f"Unlearned classes:         {unlearn_id_set}")
print(f"Unlearned accuracy:        {target_test_acc[-1]:.4f}")
print(f"Remaining accuracy:        {other_test_acc[-1]:.4f}")
print('='*50)