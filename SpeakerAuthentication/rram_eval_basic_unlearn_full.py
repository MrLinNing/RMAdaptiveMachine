
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from datasets_SSC_speakers import SSC_dataloaders
from models import SNN_LSM_Model

import random
import numpy as np
import argparse
import os

from utils import freeze_all_param, print_trainable_params, labels_to_one_hot

import pandas as pd
import copy
import time
import csv


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
    parser.add_argument('--unlearned_class_idx', type=int, default=1, help='Index of the class to unlearn')

    # snn information
    parser.add_argument('--vth', type=float, default=0.3,
                        help='lif neuron vth')
    parser.add_argument('--decay', type=float, default=0.5,
                        help='lif neuron decay')
    parser.add_argument('--const', type=float, default=0.5,
                        help='lif neuron constant')


    ## setup
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='seed id')
    ## checkname
    parser.add_argument('--name',type=str, default='snn_lsm_512_full_randlabel_5cls_10twat50')


    parser.add_argument('--pretrained_model', type=str, default='./checkpoints_randlabel_unlearn_full/snn_lsm_512_full_randlabel_5cls_10twat51.pth',
                         help='Path to the pretrained model')


    
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

assert cfg_fc[-1] == num_selected_speakers

n_epochs = args.epochs

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

# Apply LoRA patches to the model for unlearning
freeze_all_param(model)
print_trainable_params(model)

# tsne

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

    unlearned_cls_acc_test = cls_accs_dict_test.get(unlearned_class_idx, None)
    other_cls_total_test = sum([total for idx, total in cls_total_dict_test.items() if idx != unlearned_class_idx])
    other_cls_correct_test = sum([correct for idx, correct in cls_correct_dict_test.items() if idx != unlearned_class_idx])
    other_cls_acc_test = other_cls_correct_test / other_cls_total_test


    test_acc = test_correct / test_total
    return test_acc, unlearned_cls_acc_test, other_cls_acc_test, cls_accs_dict_test

orig_test_acc, orig_unlearned_cls_acc, orig_other_cls_acc, orig_cls_accs_dict = evalute_model(model, test_loader, num_selected_speakers, unlearned_class_idx)


# 1. Loading model parameters into dictionary
weights = {name: param.detach().cpu() for name, param in model.named_parameters() if 'weight' in name}

# 2. Defining conductance parameters
conductance_min = 20  # Minimum conductance value
conductance_max = 80  # Maximum conductance value
conductance_zero = 50  # Conductance value corresponding to zero weight

# 3. Initializing recording variables
layer_record = []
conductance = {}

# 4. Iterating over the weight dictionary for conversion
for k, v in weights.items():
    # Skip bias terms (only process weights)
    if 'bias' in k:  
        print(f"Skipping bias layer: {k}")
        continue

    v = v.view(-1)  # Flatten weight tensor
    max_val = v.max()
    min_val = v.min()

    # Calculate positive and negative maxima
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

    conductance[k] = conductance[k].numpy()  # Convert to NumPy array

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


# 6. Building scale factor dictionary (for inverse transformation)
scale_dict = {record[0]: record[5] for record in layer_record}  # {Layer name: Scale factor}

def evaluate_model_RRAM(model, test_loader, num_selected_speakers, unlearned_class_idx,
                        conductance=None, scale_dict=None, noise_std=0.0):
    device = next(model.parameters()).device
    
    # ========== Noise weight processing ==========
    if conductance and scale_dict and noise_std > 0:
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
    test_acc, unlearned_cls_acc, other_cls_acc, cls_accs_dict = evalute_model(model, test_loader, num_selected_speakers, unlearned_class_idx)
    
    # ========== Restoring original weights ==========
    if conductance and scale_dict and noise_std > 0:
        model.load_state_dict(original_state_dict)
    
    # Return overall accuracy, per-class accuracy dictionary, unlearned class accuracy, and remaining class accuracy
    return test_acc, unlearned_cls_acc, other_cls_acc, cls_accs_dict
# Getting baseline values for the original model
model_copy = copy.deepcopy(model).to(device)
orig_overall_acc, orig_unlearn_acc, orig_remain_acc, orig_class_acc = evaluate_model_RRAM(
    model_copy, test_loader, num_selected_speakers, unlearned_class_idx,
    conductance=conductance,
    scale_dict=scale_dict,
    noise_std=0.0
)
del model_copy
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"Original overall accuracy: {orig_overall_acc:.4f}")
print(f"Original per-class accuracy:")
for cls_idx, cls_acc in orig_class_acc.items():
    print(f"  Class {cls_idx}: {cls_acc:.4f}")
print(f"Original unlearned class error rate: {orig_unlearn_acc:.4f}")
print(f"Original remaining class accuracy: {orig_remain_acc:.4f}")

# Creating results list
results = []

# Creating CSV file and writing header
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_filename = f"noise_sensitivity_results_ul.csv"

# Building dynamic header
fieldnames = ['std', 'run', 'noisy_accuracy', 'unlearn_accuracy', 'remain_accuracy']
for cls_idx, cls_acc in orig_class_acc.items():
    fieldnames.append(f'class_{cls_idx}_accuracy')
    fieldnames.append(f'delta_class_{cls_idx}')
fieldnames.extend(['delta_noisy', 'delta_unlearn', 'delta_remain'])

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Setting noise level range
noise_levels = list(range(1, 11))

# Initializing progress bar
pbar = tqdm(total=len(noise_levels) * 100, desc="Noise Sensitivity Analysis")
# Iterating over all noise levels
for std in noise_levels:
    # Results list for the current noise level
    std_noisy_results = []
    std_unlearn_results = []
    std_remain_results = []
    std_class_results = [[] for _ in range(num_selected_speakers)]  # Results list for each class
    
    # Running 100 times for each noise level
    for run in range(100):
        # Creating model copy
        model_copy = copy.deepcopy(model).to(device)
        
        # Evaluating model with noise
        noisy_overall_acc, unlearn_acc, remain_acc, class_acc = evaluate_model_RRAM(
            model_copy, 
            test_loader, 
            num_selected_speakers, 
            unlearned_class_idx,
            conductance=conductance,
            scale_dict=scale_dict,
            noise_std=std
        )
        
        # Calculating accuracy changes
        delta_noisy = noisy_overall_acc - orig_overall_acc
        delta_unlearn = unlearn_acc - orig_unlearn_acc
        delta_remain = remain_acc - orig_remain_acc
        
        # Calculating changes for each class
        class_deltas = [class_acc[i] - orig_class_acc[i] for i in range(num_selected_speakers)]
        
        # Recording results
        result_entry = {
            'std': std,
            'run': run + 1,
            'noisy_accuracy': noisy_overall_acc,
            'unlearn_accuracy': unlearn_acc,
            'remain_accuracy': remain_acc,
            'delta_noisy': delta_noisy,
            'delta_unlearn': delta_unlearn,
            'delta_remain': delta_remain
        }
        
        # Adding accuracy and changes for each class
        for i in range(num_selected_speakers):
            result_entry[f'class_{i}_accuracy'] = class_acc[i]
            result_entry[f'delta_class_{i}'] = class_deltas[i]
        
        results.append(result_entry)
        
        # Saving current run results
        std_noisy_results.append(noisy_overall_acc)
        std_unlearn_results.append(unlearn_acc)
        std_remain_results.append(remain_acc)
        for i in range(num_selected_speakers):
            std_class_results[i].append(class_acc[i])
        
        # Saving to CSV file (real-time writing)
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result_entry)
        
        # Cleaning up resources
        del model_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Updating progress bar
        pbar.update(1)
    
    # Calculating statistics for the current noise level
    def calc_stats(values):
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        std_dev = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
        return mean_val, min_val, max_val, std_dev
    
    mean_noisy, min_noisy, max_noisy, std_noisy = calc_stats(std_noisy_results)
    mean_unlearn, min_unlearn, max_unlearn, std_unlearn = calc_stats(std_unlearn_results)
    mean_remain, min_remain, max_remain, std_remain = calc_stats(std_remain_results)
    
    # Calculating statistics for each class
    class_stats = []
    for i in range(num_selected_speakers):
        mean_val, min_val, max_val, std_val = calc_stats(std_class_results[i])
        class_stats.append((i, mean_val, min_val, max_val, std_val))
    
    # Printing summary information for the current noise level
    print(f"\nNoise level {std} completed:")
    print(f"  Overall accuracy: Mean={mean_noisy:.4f}±{std_noisy:.4f}, Range=[{min_noisy:.4f}, {max_noisy:.4f}]")
    print(f"  Unlearned class error rate: Mean={mean_unlearn:.4f}±{std_unlearn:.4f}, Range=[{min_unlearn:.4f}, {max_unlearn:.4f}]")
    print(f"  Remaining class accuracy: Mean={mean_remain:.4f}±{std_remain:.4f}, Range=[{min_remain:.4f}, {max_remain:.4f}]")
    
    # Printing statistics for each class
    print("\n  Accuracy statistics for each class:")
    for i, mean_val, min_val, max_val, std_val in class_stats:
        print(f"    Class {i}: Mean={mean_val:.4f}±{std_val:.4f}, Range=[{min_val:.4f}, {max_val:.4f}]")
# Closing progress bar
pbar.close()

# Creating DataFrame for further analysis
results_df = pd.DataFrame(results)

# Calculating statistical summaries for each noise level
def create_summary(df, prefix):
    return df.groupby('std').agg({
        f'{prefix}_accuracy': ['mean', 'min', 'max', 'std'],
        f'delta_{prefix}': ['mean', 'min', 'max', 'std']
    })

# Creating summaries for each metric
noisy_summary = create_summary(results_df, 'noisy')
unlearn_summary = create_summary(results_df, 'unlearn')
remain_summary = create_summary(results_df, 'remain')

# Creating summaries for each class
class_summaries = {}
for i in range(num_selected_speakers):
    class_df = results_df.rename(columns={
        f'class_{i}_accuracy': f'class_{i}_accuracy',
        f'delta_class_{i}': f'delta_class_{i}'
    })
    class_summaries[f'class_{i}'] = create_summary(class_df, f'class_{i}')

# Merging all summaries
summary_dfs = [noisy_summary, unlearn_summary, remain_summary]
summary_dfs.extend(class_summaries.values())
summary_df = pd.concat(summary_dfs, axis=1).reset_index()

# Saving summary results
summary_filename = f"./noise_sensitivity_summary_ul.csv"
summary_df.to_csv(summary_filename, index=False)

print(f"\nAll noise level tests completed! Detailed results saved to: {csv_filename}")
print(f"Summary statistics saved to: {summary_filename}")

# Printing final summary information
print("\nNoise sensitivity analysis summary:")
print(summary_df)

# Analyzing the impact of noise on each class
print("\nImpact of noise on each class:")
for i in range(num_selected_speakers):
    class_impact = results_df.groupby('std')[f'delta_class_{i}'].mean()
    robust_threshold = class_impact[class_impact > -0.05].index.max() if i != 2 else class_impact[class_impact < 0.05].index.max()
    impact_type = "Unlearned class" if i == 2 else f"Class {i}"
    
    print(f"  {impact_type}:")
    print(f"    Average accuracy change: {class_impact.mean():.4f}")
    print(f"    Maximum accuracy drop: {class_impact.min():.4f}")
    print(f"    Robustness threshold: std ≤ {robust_threshold}")
    print("    " + "-"*40)

