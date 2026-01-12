
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datasets_SSC_speakers import SSC_dataloaders
from models import SNN_LSM_Model 

import random
import numpy as np
import argparse
import os

from utils import freeze_all_param, print_trainable_params, labels_to_one_hot
from lora_patch_multi import patch_model_with_lora, print_model_lora_status

import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def arg_parse():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epochs', type=int, default=60,
                        help='num of epochs')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--Tw', type=int, default=10, 
                        help='Time window value of LIF')

    # model info
    parser.add_argument('--mlp', type=str, default="140,512,5",
                        help="mlp node, first node is 16 times larger than cnn last node")
    
    # dataset info
    parser.add_argument('--datasets_path', type=str, default='Datasets/SSC',
                        help="dataset path")
    parser.add_argument('--dataset', type=str, default='SSC',
                        help="dataset type")
    parser.add_argument('--n_bins', type=int, default=5, 
                        help='number of bins for encoding')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--num_selected_speakers', type=int, default=5, 
                        help='number of speakers selected for training')
    parser.add_argument('--unlearned_class_idx', type=int, default=1,
                        help='index of the class to be unlearned')

    # snn information
    parser.add_argument('--vth', type=float, default=0.3,
                        help='lif neuron vth')
    parser.add_argument('--decay', type=float, default=0.5,
                        help='lif neuron decay')
    parser.add_argument('--const', type=float, default=0.5,
                        help='lif neuron constant')


    ## setup
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=42, help='seed id')
    ## checkname
    parser.add_argument('--name',type=str, default='snn_lsm_512_randlabel_5cls_10tw')

    # add rram noise
    parser.add_argument('--noise_std', type=float, default=1.0, help='Standard deviation of RRAM noise')
    parser.add_argument('--pretrained_model', type=str, default='./checkpoints/snn_lsm_512_5cls_10tw_best_model.pth', help='Path to the pretrained model')

    # lora
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA')

    
    return parser.parse_args()

args = arg_parse()

print(args)

setup_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

cfg_fc = [int(num) for num in args.mlp.split(",")]

num_selected_speakers = args.num_selected_speakers
unlearned_class_idx = args.unlearned_class_idx

os.makedirs('checkpoints_restore', exist_ok=True) 
log_file = open(f'checkpoints_restore/{args.name}_log.txt', 'w')

with open(f'checkpoints_restore/{args.name}_config.txt', 'w') as config_file:
    for key, value in vars(args).items():
        config_file.write(f"{key}: {value}\n")


# n_bins = 5
# batch_size = args.batch
n_epochs = args.epochs
learning_rate = args.lr

# train_loader, valid_loader = SHD_dataloaders(n_bins=n_bins, batch=batch_size)

train_loader, test_loader = SSC_dataloaders(T = args.Tw, 
                                            num_selected_speakers = args.num_selected_speakers, 
                                            root_path = args.datasets_path, 
                                            dataset = args.dataset, 
                                            seed = args.seed, 
                                            batch_size = args.batch_size, 
                                            n_bins=args.n_bins
                                            )

model = SNN_LSM_Model(cfg=cfg_fc,time_window=args.Tw,thresh=args.vth,decay_lsm=args.decay, const=args.const)
model.to(device)

networkdata = torch.load(args.pretrained_model)
model.load_state_dict(networkdata)
print("Pretrained model loaded.")


def evalute_model(model, test_loader, num_selected_speakers, unlearned_class_idx):
    test_loader.reset()
    model.eval()
    test_correct = 0
    test_total = 0
    cls_total_dict_test = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    cls_correct_dict_test = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, label_speakers = data[0], data[2]
            inputs, label_speakers = inputs.to(device), label_speakers.to(device)
            
            # Convert labels to one-hot encoding
            one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
            
            outputs = model(inputs)

            # functional.reset_net(model) 

            _, predicted = torch.max(outputs, 1)
            test_total += label_speakers.size(0)
            test_correct += predicted.eq(label_speakers).sum().item()

            # acc for each class
            for cls_idx in range(num_selected_speakers):
                cls_mask = (label_speakers == cls_idx)
                cls_total = cls_mask.sum().item()
                if cls_total == 0:
                    continue
                cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
                cls_total_dict_test[cls_idx] += cls_total
                cls_correct_dict_test[cls_idx] += cls_correct
    
    cls_accs_dict_test = {}
    for cls_idx in range(num_selected_speakers):
        if cls_total_dict_test[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_test[cls_idx] / cls_total_dict_test[cls_idx]
        cls_accs_dict_test[cls_idx] = cls_acc
    for cls_idx, cls_acc in cls_accs_dict_test.items():
        print(f"Testing Class {cls_idx} accuracy: {cls_acc:.4f}")
        with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Testing Class {cls_idx} accuracy: {cls_acc:.4f}\n")

    unlearned_cls_acc_test = cls_accs_dict_test.get(unlearned_class_idx, None)
    other_cls_total_test = sum([total for idx, total in cls_total_dict_test.items() if idx != unlearned_class_idx])
    other_cls_correct_test = sum([correct for idx, correct in cls_correct_dict_test.items() if idx != unlearned_class_idx])
    other_cls_acc_test = other_cls_correct_test / other_cls_total_test

    test_acc = test_correct / test_total
    return test_acc, unlearned_cls_acc_test, other_cls_acc_test, cls_accs_dict_test

print("Evaluating original model before adding noise...")
with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"\n-----------------Evaluating original model before adding noise...-----------------------\n")
orig_test_acc, orig_unlearned_cls_acc, orig_other_cls_acc, orig_cls_accs_dict = evalute_model(model, test_loader, num_selected_speakers, unlearned_class_idx)


# 1. Loading model parameters into dictionary
weights = {name: param.detach().cpu() for name, param in model.named_parameters() if 'weight' in name}

# 2. Defining conductance parameters
conductance_min = 20  # Minimum conductance value
conductance_max = 80  # Maximum conductance value
conductance_zero = 50  # Conductance value corresponding to zero weight

# 3. Initialize recording variables
layer_record = []
conductance = {}

# 4. Iterate over the weight dictionary for conversion
for k, v in weights.items():
    # Skip bias terms (only process weights)
    if 'bias' in k:  
        print(f"Skipping bias layer: {k}")
        continue

    v = v.view(-1)  # Flatten weight tensor
    max_val = v.max()
    min_val = v.min()

    # Calculate positive and negative maximum values
    pos_max = max(0, max_val)
    neg_max = abs(min(0, min_val))

    # Scaling logic
    if pos_max == 0 and neg_max == 0:
        scale = 1.0
        conductance[k] = torch.ones_like(v) * conductance_zero
    elif pos_max > neg_max:
        scale = (conductance_max - conductance_zero) / pos_max
        conductance[k] = conductance_zero + scale * v
    else:
        scale = (conductance_zero - conductance_min) / neg_max
        conductance[k] = conductance_zero + scale * v

    conductance[k] = conductance[k].numpy()

    # Record metadata
    layer_record.append((
        k, 
        max_val.item(), 
        min_val.item(), 
        conductance[k].max(), 
        conductance[k].min(), 
        scale.item()
    ))

# 5. Saving CSV records
layer_df = pd.DataFrame(layer_record, 
                        columns=['Layer', 'Max', 'Min', 'Conductance Max', 'Conductance Min', 'Scale'])


# 6. Constructing scale factor dictionary (for inverse transformation)
scale_dict = {record[0]: record[5] for record in layer_record}  # {layer_name: scale_factor}

noise_std = args.noise_std  # Define noise standard deviation

original_state_dict = model.state_dict().copy()

noisy_weights = {}
for layer_name, cond_array in conductance.items():
    cond_tensor = torch.tensor(cond_array, device=device, dtype=torch.float32)
    noise = torch.normal(0, noise_std, cond_tensor.shape, device=device, dtype=torch.float32)
    noisy_conductance = torch.clamp(cond_tensor + noise, 20, 80)
    
    scale = scale_dict[layer_name]
    restored_weight = (noisy_conductance - conductance_zero) / scale
    original_shape = model.state_dict()[layer_name].shape
    noisy_weights[layer_name] = restored_weight.view(original_shape)

# Update model weights with noisy weights
with torch.no_grad():
    for name, param in model.named_parameters():
        if name in noisy_weights:
            param.copy_(noisy_weights[name])

# ========== Model evaluation ==========
model.eval()

# Overall accuracy calculation
print("Evaluating model after adding RRAM noise...")
with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"\n-----------------Evaluating model after adding RRAM noise...-----------------------\n")
test_acc, unlearned_cls_acc, other_cls_acc, cls_accs_dict = evalute_model(model, test_loader, num_selected_speakers, unlearned_class_idx)
    

# Apply LoRA patches to the model for restoration
freeze_all_param(model)

def custom_filter(name, layer):
    if isinstance(layer, nn.Conv2d):

        return layer.out_channels > 64
    elif isinstance(layer, nn.Linear):

        return layer.in_features * layer.out_features >= 10000
    else:
        return False

rank = args.lora_rank
lora_alpha = args.lora_alpha
patched_count, total_count, report = patch_model_with_lora(
    model, 
    rank=rank, 
    lora_alpha=lora_alpha,
    target_layers=custom_filter,
    verbose=True
)

print_model_lora_status(model)
print_trainable_params(model)

# Calculate total number of model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params} ")

# If you want to see the number of trainable parameters (non-frozen parameters)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} ")


# Change the criterion to nn.MSELoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
    log_file.write(f"\n-----------------Starting restoration training...-----------------------\n")

# restoration training loop
best_test_acc = 0
for epoch in range(n_epochs):

    train_loader.reset()
    # valid_loader.reset()
    test_loader.reset()

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    cls_total_dict_train = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    cls_correct_dict_train = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    unlearned_cls_acc = 0.0

    loss_lower_bound = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels, label_speakers = data[0], data[1], data[2]
        inputs, label_speakers = inputs.to(device), label_speakers.to(device)

        one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, one_hot_labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += label_speakers.size(0)
        correct += predicted.eq(label_speakers).sum().item()

        for cls_idx in range(num_selected_speakers):
            cls_mask = (label_speakers == cls_idx)
            cls_total = cls_mask.sum().item()
            if cls_total == 0:
                continue
            cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
            cls_total_dict_train[cls_idx] += cls_total
            cls_correct_dict_train[cls_idx] += cls_correct
    
    cls_accs_dict_train = {}
    for cls_idx in range(num_selected_speakers):
        if cls_total_dict_train[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_train[cls_idx] / cls_total_dict_train[cls_idx]
        cls_accs_dict_train[cls_idx] = cls_acc
    # print acc for each class
    for cls_idx, cls_acc in cls_accs_dict_train.items():
        print(f"Training Class {cls_idx} accuracy during training: {cls_acc:.4f}")
        with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Training Class {cls_idx} accuracy during training: {cls_acc:.4f}\n")

    # print unlearned class accuracy and other classes accuracy
    unlearned_cls_acc_train = cls_accs_dict_train.get(unlearned_class_idx, None)
    other_cls_total_train = sum([total for idx, total in cls_total_dict_train.items() if idx != unlearned_class_idx])
    other_cls_correct_train = sum([correct for idx, correct in cls_correct_dict_train.items() if idx != unlearned_class_idx])
    other_cls_acc_train = other_cls_correct_train / other_cls_total_train

    train_loss = total_loss / (i + 1)
    train_acc = correct / total

    print(f"Epoch {epoch + 1}/{n_epochs}: Train Acc: {train_acc:.4f}")
    with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Train Acc: {train_acc:.4f}\n")

    


    # Testing loop
    model.eval()
    test_correct = 0
    test_total = 0
    cls_total_dict_test = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}
    cls_correct_dict_test = {cls_idx: 0 for cls_idx in range(num_selected_speakers)}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, label_speakers = data[0], data[2]
            inputs, label_speakers = inputs.to(device), label_speakers.to(device)
            
            # Convert labels to one-hot encoding
            one_hot_labels = labels_to_one_hot(label_speakers, num_classes=cfg_fc[-1])
            
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            test_total += label_speakers.size(0)
            test_correct += predicted.eq(label_speakers).sum().item()

            # acc for each class
            for cls_idx in range(num_selected_speakers):
                cls_mask = (label_speakers == cls_idx)
                cls_total = cls_mask.sum().item()
                if cls_total == 0:
                    continue
                cls_correct = (predicted[cls_mask] == label_speakers[cls_mask]).sum().item()
                cls_total_dict_test[cls_idx] += cls_total
                cls_correct_dict_test[cls_idx] += cls_correct
    
    cls_accs_dict_test = {}
    for cls_idx in range(num_selected_speakers):
        if cls_total_dict_test[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_test[cls_idx] / cls_total_dict_test[cls_idx]
        cls_accs_dict_test[cls_idx] = cls_acc
    # print acc for each class
    for cls_idx, cls_acc in cls_accs_dict_test.items():
        print(f"Testing Class {cls_idx} accuracy after restoration: {cls_acc:.4f}")
        with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
            log_file.write(f"Testing Class {cls_idx} accuracy after restoration: {cls_acc:.4f}\n")
            
    # print unlearned class accuracy and other classes accuracy
    unlearned_cls_acc_test = cls_accs_dict_test.get(unlearned_class_idx, None)
    print(f"Testing Unlearned class (index {unlearned_class_idx}) accuracy after restoration: {unlearned_cls_acc_test:.4f}")
    with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Unlearned class (index {unlearned_class_idx}) accuracy after restoration: {unlearned_cls_acc_test:.4f}\n")
    other_cls_total_test = sum([total for idx, total in cls_total_dict_test.items() if idx != unlearned_class_idx])
    other_cls_correct_test = sum([correct for idx, correct in cls_correct_dict_test.items() if idx != unlearned_class_idx])
    other_cls_acc_test = other_cls_correct_test / other_cls_total_test
    print(f"Testing Other classes accuracy after restoration: {other_cls_acc_test:.4f}")
    with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch + 1}/{n_epochs}: Other classes accuracy after restoration: {other_cls_acc_test:.4f}\n")
            

    test_acc = test_correct / test_total

    if test_acc > best_test_acc:
        best_test_acc = test_acc

        torch.save(model.state_dict(), f'checkpoints_restore/{args.name}_best_model.pth')
        print(f"New best model saved! Accuracy: {test_acc:.4f}")
    
    
    print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f}")
    with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
        write_line = f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f}\n"
        log_file.write(write_line)

    print('--------------------------------------------\n')
    with open(f'checkpoints_restore/{args.name}_log.txt', 'a') as log_file:
        write_line = f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}\n--------------------------------------------\n"
        log_file.write(write_line)
    
