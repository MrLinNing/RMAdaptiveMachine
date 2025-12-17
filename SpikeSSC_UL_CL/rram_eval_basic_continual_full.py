
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

from utils import freeze_all_param, print_trainable_params, labels_to_one_hot, filter_samples, replace_cls_label, random_remove_samples

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
    parser.add_argument('--mlp', type=str, default="140,512,6",
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
    parser.add_argument('--num_selected_speakers', type=int, default=7, 
                        help='number of speakers selected for training')
    parser.add_argument('--num_selected_speakers_before', type=int, default=5, 
                        help='number of speakers before continue learning')
    parser.add_argument('--previous_unlearn_idx', type=int, default=1, help='Index of the class to unlearn')
    parser.add_argument('--continue_learn_idx', type=int, default=6, help='Index of the class to continue learning')

    # snn information
    parser.add_argument('--vth', type=float, default=0.3,
                        help='lif neuron vth')
    parser.add_argument('--decay', type=float, default=0.5,
                        help='lif neuron decay')
    parser.add_argument('--const', type=float, default=0.5,
                        help='lif neuron constant')


    ## setup
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='seed id')
    ## checkname
    parser.add_argument('--name',type=str, default='snn_lsm_512_full_continue_5cls_10twat38')


    parser.add_argument('--pretrained_model', type=str, default='./checkpoints_continue_full/snn_lsm_512_full_continue_5cls_10twat38.pth',
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
unlearned_class_idx = args.previous_unlearn_idx

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

def evalute_model(model, test_loader, args):
    test_loader.reset()
    model.eval()
    test_correct = 0
    test_total = 0
    available_idx_test_continue = [i for i in range(args.num_selected_speakers_before)] + [args.continue_learn_idx]
    continue_cls_node_idx = len(available_idx_test_continue) - 1
    cls_total_dict_test = {cls_idx: 0 for cls_idx in available_idx_test_continue}
    cls_correct_dict_test = {cls_idx: 0 for cls_idx in available_idx_test_continue}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, label_speakers = data[0], data[2]
            inputs, label_speakers = inputs.to(device), label_speakers.to(device)

            # filter inputs and labels to only keep available classes
            filtered_inputs, filtered_label_speakers = filter_samples(inputs, label_speakers, available_idx_test_continue)
            # replace continue learning class index to node index in continue learning model
            modified_label_speakers, mask_continue = replace_cls_label(filtered_label_speakers, old_idx = args.continue_learn_idx, new_idx = continue_cls_node_idx)

            # Convert labels to one-hot encoding
            one_hot_labels = labels_to_one_hot(modified_label_speakers, continue_cls_node_idx + 1)
            
            outputs = model(filtered_inputs)

            _, predicted = torch.max(outputs, 1)
            test_total += modified_label_speakers.size(0)
            test_correct += predicted.eq(modified_label_speakers).sum().item()

            # acc for each class
            for cls_idx in available_idx_test_continue:
                if cls_idx == args.continue_learn_idx:
                    cls_idx_in_model = continue_cls_node_idx
                else:
                    cls_idx_in_model = cls_idx
                cls_mask = (modified_label_speakers == cls_idx_in_model)
                cls_total = cls_mask.sum().item()
                if cls_total == 0:
                    continue
                cls_correct = (predicted[cls_mask] == modified_label_speakers[cls_mask]).sum().item()
                cls_total_dict_test[cls_idx] += cls_total
                cls_correct_dict_test[cls_idx] += cls_correct
    
    cls_accs_dict_test = {}
    for cls_idx in available_idx_test_continue:
        if cls_total_dict_test[cls_idx] == 0:
            continue
        cls_acc = cls_correct_dict_test[cls_idx] / cls_total_dict_test[cls_idx]
        cls_accs_dict_test[cls_idx] = cls_acc
            
    # print continue class accuracy and other classes accuracy
    unlearned_cls_acc_test = cls_accs_dict_test.get(args.previous_unlearn_idx, None)
    other_cls_total_test = sum([total for idx, total in cls_total_dict_test.items() if idx != args.previous_unlearn_idx])
    other_cls_correct_test = sum([correct for idx, correct in cls_correct_dict_test.items() if idx != args.previous_unlearn_idx])
    other_cls_acc_test = other_cls_correct_test / other_cls_total_test
            

    test_acc = test_correct / test_total
        
    return test_acc, cls_accs_dict_test, unlearned_cls_acc_test, other_cls_acc_test

orig_overall_acc, orig_class_acc, orig_unlearn_acc, orig_remain_acc = evalute_model(model, test_loader, args)


# 1. Loading model parameters into dictionary
weights = {name: param.detach().cpu() for name, param in model.named_parameters() if 'weight' in name}
 
# 2. Defining conductance parameters
conductance_min = 20  # Minimum conductance value
conductance_max = 80  # Maximum conductance value
conductance_zero = 50  # Conductance value corresponding to zero weight
# 3. Initializing record variables
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

def evaluate_model_RRAM(model, test_loader, args,
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
        
        # Update model weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in noisy_weights:
                    param.copy_(noisy_weights[name])
    
    # ========== Model evaluation ==========
    model.eval()
    
    # Overall accuracy calculation
    overall_acc, class_acc, unlearn_acc, remain_acc = evalute_model(model, test_loader, args)
    
    # ========== Restoring original weights ==========
    if conductance and scale_dict and noise_std > 0:
        model.load_state_dict(original_state_dict)
    
    # Return overall accuracy, per-class accuracy list, forgetting metric, and retention metric
    return overall_acc, class_acc, unlearn_acc, remain_acc
# Get baseline values for the original model
model_copy = copy.deepcopy(model).to(device)
orig_overall_acc, orig_class_acc, orig_unlearn_acc, orig_remain_acc = evaluate_model_RRAM(
    model_copy, test_loader, args,
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

# Create results storage list
results = []

# Create CSV file and write header
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_filename = f"noise_sensitivity_results_cl.csv"

# Build dynamic header
fieldnames = ['std', 'run', 'noisy_accuracy', 'unlearn_accuracy', 'remain_accuracy']
for cls_idx, cls_acc in orig_class_acc.items():
    fieldnames.append(f'class_{cls_idx}_accuracy')
    fieldnames.append(f'delta_class_{cls_idx}')
fieldnames.extend(['delta_noisy', 'delta_unlearn', 'delta_remain'])

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Noise level range setting
noise_levels = list(range(1,11))

# Progress bar initialization
pbar = tqdm(total=len(noise_levels) * 100, desc="Noise Sensitivity Analysis")
# Traverse all noise levels
for std in noise_levels:
    # Results list for the current noise level
    std_noisy_results = []
    std_unlearn_results = []
    std_remain_results = []
    std_class_results = [[] for i, (cls_idx, cls_acc) in enumerate(orig_class_acc.items())]  # Results list for each class
    
    # Run 100 times for each noise level
    for run in range(100):
        # Create a copy of the model for noise evaluation
        model_copy = copy.deepcopy(model).to(device)
        
        # Evaluate the noisy model
        noisy_overall_acc, class_acc, unlearn_acc, remain_acc = evaluate_model_RRAM(
            model_copy, 
            test_loader, 
            args,
            conductance=conductance,
            scale_dict=scale_dict,
            noise_std=std
        )
        
        # Calculate accuracy changes
        delta_noisy = noisy_overall_acc - orig_overall_acc
        delta_unlearn = unlearn_acc - orig_unlearn_acc
        delta_remain = remain_acc - orig_remain_acc
        
        # Calculate changes for each class
        class_deltas = [cls_acc - orig_class_acc[cls_idx] for cls_idx, cls_acc in class_acc.items()]
        
        # Record results
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
        
        # Add accuracy and changes for each class
        for i, (cls_idx, cls_acc) in enumerate(class_acc.items()):
            result_entry[f'class_{cls_idx}_accuracy'] = class_acc[cls_idx]
            result_entry[f'delta_class_{cls_idx}'] = class_deltas[i]
        
        results.append(result_entry)
        
        # Save current run results
        std_noisy_results.append(noisy_overall_acc)
        std_unlearn_results.append(unlearn_acc)
        std_remain_results.append(remain_acc)
        for i, (cls_idx, cls_acc) in enumerate(class_acc.items()):
            std_class_results[i].append(class_acc[cls_idx])
        
        # Save to CSV file (real-time writing)
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result_entry)
        
        # Clean up resources
        del model_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Update progress bar
        pbar.update(1)
    
    # Calculate statistics for the current noise level
    def calc_stats(values):
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        std_dev = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
        return mean_val, min_val, max_val, std_dev
    
    mean_noisy, min_noisy, max_noisy, std_noisy = calc_stats(std_noisy_results)
    mean_unlearn, min_unlearn, max_unlearn, std_unlearn = calc_stats(std_unlearn_results)
    mean_remain, min_remain, max_remain, std_remain = calc_stats(std_remain_results)
    
    # Calculate statistics for each class
    class_stats = []
    for i, (cls_idx, cls_acc) in enumerate(orig_class_acc.items()):
        mean_val, min_val, max_val, std_val = calc_stats(std_class_results[i])
        class_stats.append((i, mean_val, min_val, max_val, std_val))
    
    # Print summary information for the current noise level
    print(f"\nNoise level {std} completed:")
    print(f"  Overall accuracy: Mean={mean_noisy:.4f}±{std_noisy:.4f}, Range=[{min_noisy:.4f}, {max_noisy:.4f}]")
    print(f"  Unlearn class error rate: Mean={mean_unlearn:.4f}±{std_unlearn:.4f}, Range=[{min_unlearn:.4f}, {max_unlearn:.4f}]")
    print(f"  Remaining class accuracy: Mean={mean_remain:.4f}±{std_remain:.4f}, Range=[{min_remain:.4f}, {max_remain:.4f}]")
    
    # Print statistics for each class
    print("\n  Accuracy statistics for each class:")
    for i, mean_val, min_val, max_val, std_val in class_stats:
        print(f"    Class {i}: Mean={mean_val:.4f}±{std_val:.4f}, Range=[{min_val:.4f}, {max_val:.4f}]")

# Close progress bar
pbar.close()

# Create DataFrame for further analysis
results_df = pd.DataFrame(results)

# Calculate statistics for each noise level
def create_summary(df, prefix):
    return df.groupby('std').agg({
        f'{prefix}_accuracy': ['mean', 'min', 'max', 'std'],
        f'delta_{prefix}': ['mean', 'min', 'max', 'std']
    })

# Create summaries for each metric
noisy_summary = create_summary(results_df, 'noisy')
unlearn_summary = create_summary(results_df, 'unlearn')
remain_summary = create_summary(results_df, 'remain')

# Create summaries for each class
class_summaries = {}
for i, (cls_idx, cls_acc) in enumerate(orig_class_acc.items()):
    class_df = results_df.rename(columns={
        f'class_{cls_idx}_accuracy': f'class_{cls_idx}_accuracy',
        f'delta_class_{cls_idx}': f'delta_class_{cls_idx}'
    })
    class_summaries[f'class_{cls_idx}'] = create_summary(class_df, f'class_{cls_idx}')

# Combine all summaries
summary_dfs = [noisy_summary, unlearn_summary, remain_summary]
summary_dfs.extend(class_summaries.values())
summary_df = pd.concat(summary_dfs, axis=1).reset_index()

# Save summary results
summary_filename = f"./noise_sensitivity_summary_cl.csv"
summary_df.to_csv(summary_filename, index=False)

print(f"\nAll noise level tests completed! Detailed results saved to: {csv_filename}")
print(f"Summary statistics saved to: {summary_filename}")

# Print final summary information
print("\nNoise sensitivity analysis summary:")
print(summary_df)

# Analyze the impact of noise on each class
print("\nNoise impact analysis for each class:")
for i, (cls_idx, cls_acc) in enumerate(orig_class_acc.items()):
    class_impact = results_df.groupby('std')[f'delta_class_{cls_idx}'].mean()
    robust_threshold = class_impact[class_impact > -0.05].index.max() if i != 2 else class_impact[class_impact < 0.05].index.max()
    impact_type = "Forgetting class" if i == 2 else f"Class {i}"
    
    print(f"  {impact_type}:")
    print(f"    Mean accuracy change: {class_impact.mean():.4f}")
    print(f"    Maximum accuracy drop: {class_impact.min():.4f}")
    print(f"    Robustness threshold: std ≤ {robust_threshold}")
    print("    " + "-"*40)

